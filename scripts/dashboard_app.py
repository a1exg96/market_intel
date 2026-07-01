from __future__ import annotations

import time
import base64
import os
import secrets
import binascii
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from scripts.cache import redis_status
from scripts.common import RAW_DIR, LAB_CONFIG, read_parquet
from scripts.db import init_db, latest_dashboard_snapshot
from scripts.paper_execution import (
    read_active_positions,
    read_audit,
    read_signals,
    read_trades,
    records,
    stats_snapshot,
    update_open_positions,
)

STARTED_AT = time.time()
KYIV_TZ = ZoneInfo("Europe/Kyiv")
TIME_FIELDS = {"timestamp", "ts", "opened_at", "closed_at", "last_update", "updated_at", "generated_at"}
app = FastAPI(title="Market Intelligence Dashboard", version="0.1", docs_url=None, redoc_url=None, openapi_url=None)
security = HTTPBasic()
PUBLIC_PATHS = {"/health"}


def _dashboard_user() -> str:
    return os.getenv("DASHBOARD_USERNAME", "admin")


def _dashboard_password() -> str:
    return os.getenv("DASHBOARD_PASSWORD", "")


def _allowed_dashboard_ips() -> set[str]:
    raw = os.getenv("DASHBOARD_ALLOWED_IPS", "")
    return {item.strip() for item in raw.split(",") if item.strip()}


def _dashboard_auth_error(status_code: int, detail: str) -> HTTPException:
    headers = {"WWW-Authenticate": "Basic"} if status_code == status.HTTP_401_UNAUTHORIZED else None
    return HTTPException(status_code=status_code, detail=detail, headers=headers)


def _check_dashboard_access(request: Request, username: str, password_value: str) -> str:
    password = _dashboard_password()
    if not password:
        raise _dashboard_auth_error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Dashboard access is not configured. Set DASHBOARD_PASSWORD in .env.",
        )

    allowed_ips = _allowed_dashboard_ips()
    client_ip = request.client.host if request.client else ""
    if allowed_ips and client_ip not in allowed_ips:
        raise _dashboard_auth_error(status.HTTP_403_FORBIDDEN, "IP address is not allowed.")

    username_ok = secrets.compare_digest(username, _dashboard_user())
    password_ok = secrets.compare_digest(password_value, password)
    if not (username_ok and password_ok):
        raise _dashboard_auth_error(status.HTTP_401_UNAUTHORIZED, "Invalid dashboard credentials.")
    return username


def require_dashboard_access(request: Request, credentials: HTTPBasicCredentials = Depends(security)) -> str:
    return _check_dashboard_access(request, credentials.username, credentials.password)


def _basic_credentials_from_header(authorization: str | None) -> tuple[str, str] | None:
    if not authorization or not authorization.lower().startswith("basic "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    try:
        decoded = base64.b64decode(token, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return None
    if ":" not in decoded:
        return None
    username, password_value = decoded.split(":", 1)
    return username, password_value


@app.middleware("http")
async def dashboard_auth_middleware(request: Request, call_next: Any) -> Any:
    if request.url.path in PUBLIC_PATHS:
        return await call_next(request)

    credentials = _basic_credentials_from_header(request.headers.get("authorization"))
    if credentials is None:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Dashboard credentials are required."},
            headers={"WWW-Authenticate": "Basic"},
        )

    try:
        _check_dashboard_access(request, credentials[0], credentials[1])
    except HTTPException as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=exc.headers or {},
        )

    return await call_next(request)


def _to_kyiv_time(value: Any) -> Any:
    if value in (None, ""):
        return value
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(KYIV_TZ).isoformat()


def _with_kyiv_times(item: dict[str, Any]) -> dict[str, Any]:
    return {key: _to_kyiv_time(value) if key in TIME_FIELDS else value for key, value in item.items()}


def _records_kyiv(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_with_kyiv_times(row) for row in rows]


def _market_price_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(RAW_DIR.glob(f"*_{LAB_CONFIG.timeframe}_candles.parquet")):
        try:
            candles = read_parquet(path)
        except Exception:
            continue
        if candles.empty:
            continue
        latest = candles.iloc[-1]
        rows.append(
            {
                "symbol": str(latest.get("symbol", path.name.split("_")[0])),
                "price": float(latest["close"]),
                "source": str(latest.get("source", "local")),
                "timestamp": latest.get("timestamp"),
                "updated_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc),
            }
        )
    return _records_kyiv(rows)


@app.on_event("startup")
def startup() -> None:
    try:
        init_db()
    except Exception:
        pass


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        stats = stats_snapshot()
        csv_state = "ok"
    except Exception as exc:
        stats = {}
        csv_state = f"error: {exc}"
    return {
        "collector_status": "see local logs",
        "research_status": "see local logs",
        "csv_state": csv_state,
        "redis_status": redis_status(),
        "uptime_seconds": int(time.time() - STARTED_AT),
        "last_update": datetime.now(KYIV_TZ).isoformat(),
        "stats": _with_kyiv_times(stats),
    }


@app.get("/api/stats", dependencies=[Depends(require_dashboard_access)])
def api_stats() -> dict[str, Any]:
    return _with_kyiv_times(stats_snapshot())


@app.get("/api/active-positions", dependencies=[Depends(require_dashboard_access)])
def api_active_positions() -> list[dict[str, Any]]:
    stats_snapshot()
    return _records_kyiv(records(read_active_positions(open_only=True)))


@app.get("/api/trades", dependencies=[Depends(require_dashboard_access)])
def api_trades() -> list[dict[str, Any]]:
    return _records_kyiv(records(read_trades().tail(100).iloc[::-1].reset_index(drop=True)))


@app.get("/api/signals", dependencies=[Depends(require_dashboard_access)])
def api_signals() -> list[dict[str, Any]]:
    return _records_kyiv(records(read_signals(limit=100).iloc[::-1].reset_index(drop=True)))


@app.get("/api/signal-execution-audit", dependencies=[Depends(require_dashboard_access)])
def api_signal_execution_audit() -> list[dict[str, Any]]:
    return _records_kyiv(records(read_audit().tail(100).iloc[::-1].reset_index(drop=True)))


@app.get("/api/market-prices", dependencies=[Depends(require_dashboard_access)])
def api_market_prices() -> list[dict[str, Any]]:
    return _market_price_rows()


@app.get("/api/summary", dependencies=[Depends(require_dashboard_access)])
def summary() -> dict[str, Any]:
    signals = api_signals()
    audit = api_signal_execution_audit()
    try:
        db_snapshot = latest_dashboard_snapshot()
        logs = db_snapshot.get("logs", [])
    except Exception:
        logs = []
    return {
        "stats": api_stats(),
        "last_signal": signals[0] if signals else None,
        "active_positions": api_active_positions(),
        "trades": api_trades(),
        "signals": signals,
        "signal_execution_audit": audit,
        "market_prices": api_market_prices(),
        "logs": logs,
    }


@app.get("/", response_class=HTMLResponse, dependencies=[Depends(require_dashboard_access)])
def index() -> str:
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Market Intelligence Dashboard</title>
  <style>
    :root { color-scheme: dark; --bg:#101419; --panel:#171c22; --line:#2b323b; --muted:#93a4b7; --text:#eef2f6; --green:#4ade80; --red:#fb7185; --blue:#60a5fa; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:Arial, sans-serif; background:var(--bg); color:var(--text); }
    header { padding:8px 14px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; gap:12px; align-items:center; font-size:11px; }
    main { padding:10px 14px; display:grid; gap:10px; }
    .grid { display:grid; grid-template-columns:repeat(9, minmax(96px, 1fr)); gap:8px; }
    .twoCol { display:grid; grid-template-columns:minmax(240px, 0.82fr) minmax(340px, 1.18fr); gap:10px; align-items:start; }
    .card, .panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:9px; }
    .label { color:var(--muted); font-size:10px; }
    .value { font-size:11px; margin-top:3px; overflow-wrap:anywhere; font-weight:700; }
    h3 { margin:0 0 7px; font-size:14px; }
    table { width:100%; border-collapse:collapse; font-size:11px; table-layout:auto; }
    th, td { border-bottom:1px solid var(--line); padding:5px 6px; text-align:left; white-space:nowrap; }
    .tableWrap { overflow-x:auto; }
    .compactPanel { padding:8px 9px; }
    .compactTable { font-size:10px; }
    .compactTable th, .compactTable td { padding:4px 5px; }
    .shortPanel { padding:6px 8px; }
    .shortPanel h3 { margin-bottom:4px; font-size:12px; }
    .shortPanel table { font-size:10px; }
    .shortPanel th, .shortPanel td { padding:3px 5px; }
    .pricesList { display:grid; gap:5px; }
    .priceRow { display:grid; grid-template-columns:minmax(76px, 1fr) auto; gap:8px; align-items:center; border:1px solid var(--line); border-radius:8px; padding:6px 8px; background:#11161c; }
    .priceSymbol { color:var(--muted); font-size:10px; }
    .priceValue { font-size:11px; text-align:right; font-weight:700; }
    .priceTime { grid-column:1 / -1; color:var(--muted); font-size:9px; display:flex; justify-content:space-between; gap:8px; }
    .scroll { max-height:210px; overflow-y:auto; }
    .scrollSmall { max-height:92px; overflow-y:auto; }
    .profit { color:var(--green); }
    .loss { color:var(--red); }
    .sideLong { color:var(--blue); font-weight:700; }
    .sideShort { color:var(--red); font-weight:700; }
    .signalBox { display:grid; grid-template-columns:repeat(5, minmax(82px, 1fr)); gap:7px; }
    .signalBox .signalItem { min-height:48px; }
    .signalItem { border:1px solid var(--line); border-radius:8px; padding:7px; background:#11161c; font-size:10px; }
    .signalWide { grid-column:span 2; }
    .decisionItem { grid-column:span 2; }
    .badge { display:inline-block; padding:3px 6px; border-radius:999px; font-weight:700; font-size:10px; }
    .badgeLong { background:#12351f; color:var(--green); }
    .badgeShort { background:#3d1420; color:var(--red); }
    .badgeFlat { background:#263241; color:#cbd5e1; }
    @media (max-width:1200px) { .grid { grid-template-columns:repeat(3, minmax(0, 1fr)); } .twoCol { grid-template-columns:1fr; } }
    @media (max-width:760px) { header { align-items:flex-start; flex-direction:column; } main { padding:8px; } .grid, .signalBox { grid-template-columns:repeat(2, minmax(0, 1fr)); } .value { font-size:11px; } }
  </style>
</head>
<body>
  <header>
    <strong>Лабораторія паперової торгівлі</strong>
    <span id="updated">завантаження...</span>
  </header>
  <main>
    <section class="grid">
      <div class="card"><div class="label">Баланс</div><div class="value" id="balance">-</div></div>
      <div class="card"><div class="label">Капітал</div><div class="value" id="equityValue">-</div></div>
      <div class="card"><div class="label">Зафіксований PnL</div><div class="value" id="realized">-</div></div>
      <div class="card"><div class="label">Нереалізований PnL</div><div class="value" id="unrealized">-</div></div>
      <div class="card"><div class="label">Угоди</div><div class="value" id="tradesCount">-</div></div>
      <div class="card"><div class="label">Відкриті позиції</div><div class="value" id="openPositions">-</div></div>
      <div class="card"><div class="label">Winrate</div><div class="value" id="winrate">-</div></div>
      <div class="card"><div class="label">Profit Factor</div><div class="value" id="pf">-</div></div>
      <div class="card"><div class="label">Макс. просадка</div><div class="value" id="maxdd">-</div></div>
    </section>
    <section class="twoCol">
      <div class="panel">
        <h3>Ринкові ціни</h3>
        <div id="priceRows" class="pricesList"></div>
      </div>
      <div class="panel">
        <h3>Останній сигнал</h3>
        <div id="signal" class="signalBox"></div>
      </div>
    </section>
    <section class="panel compactPanel">
      <h3>Активні позиції</h3>
      <div class="tableWrap"><table class="compactTable"><thead><tr><th>ID</th><th>Відкрито</th><th>Пара</th><th>Сторона</th><th>Вхід</th><th>Поточна</th><th>Розмір</th><th>Поточна сума</th><th>Впевн.</th><th>PnL USD</th><th>PnL %</th><th>Статус</th></tr></thead><tbody id="positionRows"></tbody></table></div>
    </section>
    <section class="panel shortPanel">
      <h3>Закриті угоди</h3>
      <div class="tableWrap scrollSmall"><table><thead><tr><th>Відкрито</th><th>Закрито</th><th>Пара</th><th>Сторона</th><th>Вхід</th><th>Вихід</th><th>Розмір</th><th>Впевн.</th><th>PnL USD</th><th>PnL %</th><th>Баланс</th><th>Причина</th></tr></thead><tbody id="tradeRows"></tbody></table></div>
    </section>
    <section class="panel shortPanel">
      <h3>Останні сигнали</h3>
      <div class="tableWrap scrollSmall"><table><thead><tr><th>Свічка</th><th>Згенеровано</th><th>Пара</th><th>Режим</th><th>Сигнал</th><th>Впевн.</th><th>Ціна</th><th>Статус</th></tr></thead><tbody id="signalRows"></tbody></table></div>
    </section>
  </main>
<script>
async function refresh() {
  const [stats, positions, trades, signals, audit, prices] = await Promise.all([
    fetch('/api/stats').then(r => r.json()),
    fetch('/api/active-positions').then(r => r.json()),
    fetch('/api/trades').then(r => r.json()),
    fetch('/api/signals').then(r => r.json()),
    fetch('/api/signal-execution-audit').then(r => r.json()),
    fetch('/api/market-prices').then(r => r.json())
  ]);
  setText('balance', money(stats.balance));
  setText('equityValue', money(stats.equity));
  setText('realized', money(stats.realized_pnl));
  setText('unrealized', money(stats.unrealized_pnl));
  setText('tradesCount', stats.trades_count || 0);
  setText('openPositions', stats.open_positions_count || 0);
  setText('winrate', pct(stats.winrate));
  setText('pf', Number.isFinite(stats.profit_factor) ? fmt(stats.profit_factor, 2) : 'inf');
  setText('maxdd', pct(stats.max_drawdown));
  renderSignal(signals[0] || null, audit[0] || null);
  renderPositions(positions || []);
  renderPrices(prices || []);
  renderTrades(trades || []);
  renderSignals(signals || []);
  setText('updated', displayTime(stats.last_update || ''));
}
function setText(id, value) { document.getElementById(id).textContent = value; }
function fmt(v, d = 4) { return Number(v || 0).toFixed(d); }
function money(v) { return Number(v || 0).toFixed(2); }
function pct(v) { return (Number(v || 0) * 100).toFixed(1) + '%'; }
function pctRaw(v) { return Number(v || 0).toFixed(2) + '%'; }
function displayTime(v) { return String(v || '-').replace('T', ' ').replace(/\\.\\d+/, '').replace(/\\+\\d\\d:\\d\\d$/, ' Kyiv'); }
function esc(v) { return String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function signalBadge(signal) {
  const normalized = String(signal || 'NO_TRADE').toUpperCase();
  const cls = normalized.includes('LONG') || normalized === 'UP' ? 'badgeLong' : normalized.includes('SHORT') || normalized === 'DOWN' ? 'badgeShort' : 'badgeFlat';
  const label = normalized === 'FLAT' || normalized === 'NO_TRADE' ? 'NO TRADE' : normalized.replaceAll('_', ' ');
  const translated = label === 'LONG' ? 'LONG' : label === 'SHORT' || label === 'DOWN' ? 'SHORT' : label;
  return `<span class="badge ${cls}">${esc(translated)}</span>`;
}
function sideClass(side) { return String(side).toUpperCase() === 'SHORT' ? 'sideShort' : 'sideLong'; }
function pnlClass(v) { return Number(v || 0) >= 0 ? 'profit' : 'loss'; }
function renderSignal(signal, audit) {
  const root = document.getElementById('signal');
  if (!signal) {
    root.innerHTML = '<div class="signalItem signalWide"><div class="label">Статус</div><div class="value">Очікування</div></div>';
    return;
  }
  const action = signal.signal || signal.predicted_direction || 'NO_TRADE';
  const price = signal.entry_price || signal.price || signal.close || null;
  const normalized = String(action).toUpperCase();
  const executable = normalized.includes('LONG') || normalized.includes('SHORT') || normalized === 'DOWN';
  const auditReason = audit && String(audit.executed).toLowerCase() !== 'true' ? `Виконання заблоковано: ${audit.reason || 'UNKNOWN'}.` : '';
  const reason = auditReason || (executable ? `Виконуваний паперовий сигнал із впевненістю ${pct(signal.confidence)}.` : `Без угоди. Впевненість ${pct(signal.confidence)}.`);
  root.innerHTML = `
    <div class="signalItem"><div class="label">Дія</div><div class="value">${signalBadge(action)}</div></div>
    <div class="signalItem"><div class="label">Впевненість</div><div class="value">${pct(signal.confidence)}</div></div>
    <div class="signalItem"><div class="label">Пара</div><div class="value">${esc(signal.symbol || '-')}</div></div>
    <div class="signalItem"><div class="label">Режим</div><div class="value">${esc(signal.regime || 'UNKNOWN')}</div></div>
    <div class="signalItem"><div class="label">Свічка</div><div>${esc(displayTime(signal.timestamp || signal.ts || '-'))}</div></div>
    <div class="signalItem"><div class="label">Згенеровано</div><div>${esc(displayTime(signal.generated_at || '-'))}</div></div>
    <div class="signalItem"><div class="label">Ціна</div><div>${price ? fmt(price) : '-'}</div></div>
    <div class="signalItem decisionItem"><div class="label">Рішення</div><div>${esc(reason)}</div></div>
  `;
}
function renderPrices(rows) {
  document.getElementById('priceRows').innerHTML = rows.length ? rows.map(p => `
    <div class="priceRow"><div class="priceSymbol">${esc(p.symbol)}</div><div class="priceValue">${fmt(p.price, 2)}</div><div class="priceTime"><span>Свічка ${esc(displayTime(p.timestamp))}</span><span>Оновлено ${esc(displayTime(p.updated_at))}</span></div></div>
  `).join('') : '<div class="priceRow"><div class="priceSymbol">Немає локальної ціни</div><div class="priceTime">Очікуємо collector</div></div>';
}
function renderPositions(rows) {
  document.getElementById('positionRows').innerHTML = rows.map(p => `
    <tr><td>${esc(p.position_id)}</td><td>${esc(displayTime(p.opened_at))}</td><td>${esc(p.symbol)}</td><td class="${sideClass(p.side)}">${esc(p.side)}</td><td>${fmt(p.entry_price)}</td><td>${fmt(p.current_price)}</td><td>${fmt(p.position_size, 2)}</td><td>${fmt(Number(p.position_size || 0) + Number(p.unrealized_pnl_usd || 0), 2)}</td><td>${pct(p.confidence)}</td><td class="${pnlClass(p.unrealized_pnl_usd)}">${fmt(p.unrealized_pnl_usd, 2)}</td><td class="${pnlClass(p.unrealized_pnl_pct)}">${pctRaw(p.unrealized_pnl_pct)}</td><td>${esc(p.status)}</td></tr>
  `).join('');
}
function renderTrades(rows) {
  document.getElementById('tradeRows').innerHTML = rows.map(t => `
    <tr><td>${esc(displayTime(t.opened_at))}</td><td>${esc(displayTime(t.closed_at))}</td><td>${esc(t.symbol)}</td><td class="${sideClass(t.side)}">${esc(t.side)}</td><td>${fmt(t.entry_price)}</td><td>${fmt(t.exit_price)}</td><td>${fmt(t.position_size, 2)}</td><td>${pct(t.confidence)}</td><td class="${pnlClass(t.pnl_usd)}">${fmt(t.pnl_usd, 2)}</td><td class="${pnlClass(t.pnl_pct)}">${pctRaw(t.pnl_pct)}</td><td>${money(t.balance_after)}</td><td>${esc(t.reason)}</td></tr>
  `).join('');
}
function renderSignals(rows) {
  document.getElementById('signalRows').innerHTML = rows.map(s => `
    <tr><td>${esc(displayTime(s.timestamp || s.ts))}</td><td>${esc(displayTime(s.generated_at || '-'))}</td><td>${esc(s.symbol)}</td><td>${esc(s.regime || 'UNKNOWN')}</td><td>${signalBadge(s.signal || s.predicted_direction)}</td><td>${pct(s.confidence)}</td><td>${fmt(s.entry_price || s.price || s.close)}</td><td>${esc(s.status || s.decision || '')}</td></tr>
  `).join('');
}
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""
