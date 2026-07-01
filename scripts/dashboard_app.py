from __future__ import annotations

import time
import hashlib
import hmac
import os
import secrets
from urllib.parse import parse_qs
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from scripts.cache import redis_status
from scripts.common import RAW_DIR, LAB_CONFIG, read_parquet
from scripts.db import init_db, latest_dashboard_snapshot
from scripts.paper_execution import (
    close_position_manually,
    paper_trading_settings,
    read_active_positions,
    read_audit,
    read_signals,
    read_trades,
    records,
    stats_snapshot,
    update_open_positions,
    update_paper_trading_settings,
)

STARTED_AT = time.time()
KYIV_TZ = ZoneInfo("Europe/Kyiv")
TIME_FIELDS = {"timestamp", "ts", "opened_at", "closed_at", "last_update", "updated_at", "generated_at"}
app = FastAPI(title="Market Intelligence Dashboard", version="0.1", docs_url=None, redoc_url=None, openapi_url=None)
PUBLIC_PATHS = {"/health", "/login"}
DASHBOARD_SESSION_COOKIE = "market_intel_dashboard_session"


def _dashboard_user() -> str:
    return os.getenv("DASHBOARD_USERNAME", "admin")


def _dashboard_password() -> str:
    return os.getenv("DASHBOARD_PASSWORD", "")


def _dashboard_session_secret() -> str:
    return os.getenv("DASHBOARD_SESSION_SECRET") or _dashboard_password()


def _dashboard_session_seconds() -> int:
    try:
        return int(os.getenv("DASHBOARD_SESSION_SECONDS", "86400"))
    except ValueError:
        return 86400


def _dashboard_cookie_secure() -> bool:
    return os.getenv("DASHBOARD_COOKIE_SECURE", "").lower() in {"1", "true", "yes", "on"}


def _allowed_dashboard_ips() -> set[str]:
    raw = os.getenv("DASHBOARD_ALLOWED_IPS", "")
    return {item.strip() for item in raw.split(",") if item.strip()}


def _dashboard_auth_error(status_code: int, detail: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail=detail)


def _require_dashboard_password_configured() -> str:
    configured_password = _dashboard_password()
    if not configured_password:
        raise _dashboard_auth_error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Dashboard access is not configured. Set DASHBOARD_PASSWORD in .env.",
        )
    return configured_password


def _check_dashboard_ip(request: Request) -> None:
    allowed_ips = _allowed_dashboard_ips()
    client_ip = request.client.host if request.client else ""
    if allowed_ips and client_ip not in allowed_ips:
        raise _dashboard_auth_error(status.HTTP_403_FORBIDDEN, "IP address is not allowed.")


def _check_dashboard_access(request: Request, username: str, password_value: str) -> str:
    password = _require_dashboard_password_configured()
    _check_dashboard_ip(request)
    username_ok = secrets.compare_digest(username, _dashboard_user())
    password_ok = secrets.compare_digest(password_value, password)
    if not (username_ok and password_ok):
        raise _dashboard_auth_error(status.HTTP_401_UNAUTHORIZED, "Invalid dashboard credentials.")
    return username


def _session_signature(message: str) -> str:
    secret = _dashboard_session_secret()
    return hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()


def _create_session_token(username: str, issued_at: int | None = None) -> str:
    _require_dashboard_password_configured()
    issued = int(issued_at if issued_at is not None else time.time())
    message = f"{username}:{issued}"
    return f"{message}:{_session_signature(message)}"


def _session_username(token: str | None, now: int | None = None) -> str | None:
    if not token:
        return None
    parts = token.split(":")
    if len(parts) != 3:
        return None
    username, issued_raw, signature = parts
    try:
        issued = int(issued_raw)
    except ValueError:
        return None

    message = f"{username}:{issued}"
    if not secrets.compare_digest(signature, _session_signature(message)):
        return None

    session_seconds = _dashboard_session_seconds()
    current_time = int(now if now is not None else time.time())
    if session_seconds > 0 and current_time - issued > session_seconds:
        return None
    return username


def _auth_failure_response(request: Request, status_code: int, detail: str) -> JSONResponse | RedirectResponse:
    if request.url.path.startswith("/api/"):
        return JSONResponse(status_code=status_code, content={"detail": detail})
    return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)


@app.middleware("http")
async def dashboard_auth_middleware(request: Request, call_next: Any) -> Any:
    if request.url.path in PUBLIC_PATHS:
        return await call_next(request)

    try:
        _require_dashboard_password_configured()
        _check_dashboard_ip(request)
    except HTTPException as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=exc.headers or {},
        )

    username = _session_username(request.cookies.get(DASHBOARD_SESSION_COOKIE))
    if username != _dashboard_user():
        return _auth_failure_response(request, status.HTTP_401_UNAUTHORIZED, "Dashboard login is required.")

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
    return {
        "collector_status": "see local logs",
        "research_status": "see local logs",
        "csv_state": "not_checked",
        "redis_status": redis_status(),
        "uptime_seconds": int(time.time() - STARTED_AT),
        "last_update": datetime.now(KYIV_TZ).isoformat(),
    }


@app.get("/login", response_class=HTMLResponse)
def login_page(error: str = "") -> str:
    error_html = "<div class=\"error\">Невірний логін або пароль</div>" if error else ""
    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Вхід до dashboard</title>
  <style>
    :root {{ color-scheme: dark; --bg:#101419; --panel:#171c22; --line:#2b323b; --muted:#93a4b7; --text:#eef2f6; --red:#fb7185; --blue:#60a5fa; }}
    * {{ box-sizing:border-box; }}
    body {{ min-height:100vh; margin:0; display:grid; place-items:center; font-family:Arial, sans-serif; background:var(--bg); color:var(--text); }}
    form {{ width:min(360px, calc(100vw - 28px)); background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:18px; display:grid; gap:10px; }}
    h1 {{ margin:0 0 4px; font-size:18px; }}
    p {{ margin:0 0 8px; color:var(--muted); font-size:12px; }}
    label {{ color:var(--muted); font-size:11px; }}
    input {{ width:100%; border:1px solid var(--line); border-radius:6px; background:#11161c; color:var(--text); padding:9px 10px; font-size:14px; }}
    button {{ border:0; border-radius:6px; background:var(--blue); color:#06111f; padding:9px 10px; font-weight:700; cursor:pointer; }}
    .error {{ border:1px solid #6d1d2a; background:#35131a; color:var(--red); border-radius:6px; padding:8px; font-size:12px; }}
  </style>
</head>
<body>
  <form method="post" action="/login" autocomplete="off">
    <h1>Вхід</h1>
    <p>Після закриття браузера сесія dashboard завершиться.</p>
    {error_html}
    <label for="username">Логін</label>
    <input id="username" name="username" required autofocus />
    <label for="password">Пароль</label>
    <input id="password" name="password" type="password" required />
    <button type="submit">Увійти</button>
  </form>
</body>
</html>
"""


@app.post("/login")
async def login(request: Request) -> RedirectResponse:
    body = (await request.body()).decode("utf-8")
    form = parse_qs(body)
    username = form.get("username", [""])[0]
    password = form.get("password", [""])[0]
    try:
        _check_dashboard_access(request, username, password)
    except HTTPException:
        return RedirectResponse(url="/login?error=1", status_code=status.HTTP_303_SEE_OTHER)

    response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(
        DASHBOARD_SESSION_COOKIE,
        _create_session_token(username),
        httponly=True,
        secure=_dashboard_cookie_secure(),
        samesite="lax",
    )
    return response


@app.get("/logout")
def logout() -> RedirectResponse:
    response = RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie(DASHBOARD_SESSION_COOKIE)
    return response


@app.get("/api/stats")
def api_stats() -> dict[str, Any]:
    return _with_kyiv_times(stats_snapshot())


@app.get("/api/active-positions")
def api_active_positions() -> list[dict[str, Any]]:
    stats_snapshot()
    return _records_kyiv(records(read_active_positions(open_only=True)))


@app.get("/api/trades")
def api_trades() -> list[dict[str, Any]]:
    return _records_kyiv(records(read_trades().tail(100).iloc[::-1].reset_index(drop=True)))


@app.get("/api/signals")
def api_signals() -> list[dict[str, Any]]:
    return _records_kyiv(records(read_signals(limit=100).iloc[::-1].reset_index(drop=True)))


@app.get("/api/signal-execution-audit")
def api_signal_execution_audit() -> list[dict[str, Any]]:
    return _records_kyiv(records(read_audit().tail(100).iloc[::-1].reset_index(drop=True)))


@app.get("/api/market-prices")
def api_market_prices() -> list[dict[str, Any]]:
    return _market_price_rows()


@app.get("/api/settings")
def api_settings() -> dict[str, Any]:
    return paper_trading_settings()


@app.post("/api/settings")
async def api_update_settings(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("Settings payload must be an object.")
        return update_paper_trading_settings(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@app.post("/api/positions/{position_id}/close")
def api_close_position(position_id: str) -> dict[str, Any]:
    try:
        return _with_kyiv_times(close_position_manually(position_id))
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Open position not found.") from exc


@app.get("/api/summary")
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
        "settings": api_settings(),
        "logs": logs,
    }


@app.get("/", response_class=HTMLResponse)
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
    header a { color:var(--muted); text-decoration:none; border:1px solid var(--line); border-radius:6px; padding:4px 7px; }
    header a:hover { color:var(--text); }
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
    .settingsGrid { display:grid; grid-template-columns:repeat(5, minmax(120px, 1fr)); gap:8px; align-items:end; }
    .field { display:grid; gap:4px; }
    .field label { color:var(--muted); font-size:10px; }
    .field input { width:100%; border:1px solid var(--line); border-radius:6px; background:#11161c; color:var(--text); padding:7px 8px; font-size:12px; }
    .actionBtn { border:0; border-radius:6px; background:var(--blue); color:#06111f; padding:8px 10px; font-weight:700; cursor:pointer; }
    .closeBtn { width:24px; height:24px; border:1px solid #6d1d2a; border-radius:6px; background:#35131a; color:var(--red); font-weight:700; cursor:pointer; line-height:1; }
    .closeBtn:disabled, .actionBtn:disabled { opacity:.55; cursor:wait; }
    .settingsStatus { color:var(--muted); font-size:10px; min-height:13px; }
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
    @media (max-width:1200px) { .grid { grid-template-columns:repeat(3, minmax(0, 1fr)); } .twoCol { grid-template-columns:1fr; } .settingsGrid { grid-template-columns:repeat(2, minmax(0, 1fr)); } }
    @media (max-width:760px) { header { align-items:flex-start; flex-direction:column; } main { padding:8px; } .grid, .signalBox, .settingsGrid { grid-template-columns:repeat(2, minmax(0, 1fr)); } .value { font-size:11px; } }
  </style>
</head>
<body>
  <header>
    <strong>Лабораторія паперової торгівлі</strong>
    <span><span id="updated">завантаження...</span> <a href="/logout">Вийти</a></span>
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
    <section class="panel compactPanel">
      <h3>Налаштування ставки</h3>
      <form id="settingsForm" class="settingsGrid">
        <div class="field"><label for="leverageInput">Кредитне плече, x</label><input id="leverageInput" name="leverage" type="number" min="0.1" max="125" step="0.1" required /></div>
        <div class="field"><label for="stakeInput">Ставка, % балансу</label><input id="stakeInput" name="stake_pct" type="number" min="0.01" max="100" step="0.01" required /></div>
        <div class="field"><label for="liqLongInput">Ліквідація LONG, %</label><input id="liqLongInput" name="liquidation_long_pct" type="number" min="0.01" max="100" step="0.01" required /></div>
        <div class="field"><label for="liqShortInput">Ліквідація SHORT, %</label><input id="liqShortInput" name="liquidation_short_pct" type="number" min="0.01" max="100" step="0.01" required /></div>
        <div class="field"><button class="actionBtn" id="settingsSave" type="submit">Зберегти</button><div class="settingsStatus" id="settingsStatus"></div></div>
      </form>
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
      <div class="tableWrap"><table class="compactTable"><thead><tr><th></th><th>ID</th><th>Відкрито</th><th>Пара</th><th>Сторона</th><th>Вхід</th><th>Поточна</th><th>Розмір</th><th>Поточна сума</th><th>Впевн.</th><th>PnL USD</th><th>PnL %</th><th>Статус</th></tr></thead><tbody id="positionRows"></tbody></table></div>
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
  const [stats, positions, trades, signals, audit, prices, settings] = await Promise.all([
    fetch('/api/stats').then(r => r.json()),
    fetch('/api/active-positions').then(r => r.json()),
    fetch('/api/trades').then(r => r.json()),
    fetch('/api/signals').then(r => r.json()),
    fetch('/api/signal-execution-audit').then(r => r.json()),
    fetch('/api/market-prices').then(r => r.json()),
    fetch('/api/settings').then(r => r.json())
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
  renderSettings(settings || stats.settings || {});
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
function setInput(id, value, digits = 2) {
  const input = document.getElementById(id);
  if (document.activeElement === input) return;
  input.value = Number(value || 0).toFixed(digits).replace(/\.?0+$/, '');
}
function renderSettings(settings) {
  setInput('leverageInput', settings.leverage, 2);
  setInput('stakeInput', settings.stake_pct, 2);
  setInput('liqLongInput', settings.liquidation_long_pct, 2);
  setInput('liqShortInput', settings.liquidation_short_pct, 2);
}
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
    <tr><td><button class="closeBtn" type="button" data-position-id="${esc(p.position_id)}" title="Закрити вручну">x</button></td><td>${esc(p.position_id)}</td><td>${esc(displayTime(p.opened_at))}</td><td>${esc(p.symbol)}</td><td class="${sideClass(p.side)}">${esc(p.side)}</td><td>${fmt(p.entry_price)}</td><td>${fmt(p.current_price)}</td><td>${fmt(p.position_size, 2)}</td><td>${fmt(Number(p.position_size || 0) + Number(p.unrealized_pnl_usd || 0), 2)}</td><td>${pct(p.confidence)}</td><td class="${pnlClass(p.unrealized_pnl_usd)}">${fmt(p.unrealized_pnl_usd, 2)}</td><td class="${pnlClass(p.unrealized_pnl_pct)}">${pctRaw(p.unrealized_pnl_pct)}</td><td>${esc(p.status)}</td></tr>
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
document.getElementById('settingsForm').addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = document.getElementById('settingsSave');
  const status = document.getElementById('settingsStatus');
  button.disabled = true;
  status.textContent = 'Збереження...';
  const payload = Object.fromEntries(new FormData(event.currentTarget).entries());
  const response = await fetch('/api/settings', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  button.disabled = false;
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    status.textContent = error.detail || 'Помилка';
    return;
  }
  renderSettings(await response.json());
  status.textContent = 'Збережено';
  refresh();
});
document.getElementById('positionRows').addEventListener('click', async (event) => {
  const button = event.target.closest('.closeBtn');
  if (!button) return;
  button.disabled = true;
  const positionId = button.dataset.positionId;
  const response = await fetch(`/api/positions/${encodeURIComponent(positionId)}/close`, {method: 'POST'});
  if (!response.ok) {
    button.disabled = false;
    alert('Не вдалося закрити позицію');
    return;
  }
  refresh();
});
</script>
</body>
</html>
"""
