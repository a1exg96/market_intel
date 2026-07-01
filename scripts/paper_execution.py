from __future__ import annotations

import hashlib
import math
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from scripts.common import CONFIG_DIR, LAB_CONFIG, PROCESSED_DIR, RAW_DIR, REPORTS_DIR, ensure_dirs, max_drawdown, profit_factor, read_parquet

ACTIVE_POSITION_COLUMNS = [
    "position_id",
    "opened_at",
    "symbol",
    "side",
    "entry_price",
    "current_price",
    "position_size",
    "confidence",
    "regime",
    "model_version",
    "reason",
    "unrealized_pnl_usd",
    "unrealized_pnl_pct",
    "status",
]
TRADE_COLUMNS = [
    "position_id",
    "opened_at",
    "closed_at",
    "symbol",
    "side",
    "entry_price",
    "exit_price",
    "position_size",
    "confidence",
    "pnl_usd",
    "pnl_pct",
    "balance_after",
    "reason",
    "model_version",
]
AUDIT_COLUMNS = ["timestamp", "symbol", "signal", "confidence", "price", "decision", "executed", "reason", "position_id"]

ACTIVE_POSITIONS_PATH = REPORTS_DIR / "active_positions.csv"
TRADES_PATH = REPORTS_DIR / "trades.csv"
AUDIT_PATH = REPORTS_DIR / "signal_execution_audit.csv"
SIGNALS_PATH = REPORTS_DIR / "forward_signals.csv"
LOCK_PATH = REPORTS_DIR / ".paper_execution.lock"
_LOCAL_LOCK = threading.RLock()


@dataclass(frozen=True)
class PaperTradingConfig:
    live_trading: bool = False
    initial_balance: float = LAB_CONFIG.initial_balance
    risk_per_trade: float = LAB_CONFIG.risk_per_trade
    horizon_minutes: int = 240
    take_profit_pct: float = 1.0
    stop_loss_pct: float = 0.7
    fee_pct: float = LAB_CONFIG.fee_pct
    slippage_pct: float = LAB_CONFIG.slippage_pct
    confidence_threshold: float = LAB_CONFIG.confidence_threshold


def load_paper_trading_config(path: Path | None = None) -> PaperTradingConfig:
    config_path = path or CONFIG_DIR / "paper_trading.yaml"
    raw: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
            if not isinstance(loaded, dict):
                raise RuntimeError(f"Invalid paper trading config: {config_path}")
            raw = loaded
    config = PaperTradingConfig(**{k: v for k, v in raw.items() if k in PaperTradingConfig.__dataclass_fields__})
    if config.live_trading:
        raise RuntimeError("ERROR: live_trading=true is forbidden. Paper engine refuses to start.")
    return config


def ensure_paper_files() -> None:
    ensure_dirs()
    _ensure_csv(ACTIVE_POSITIONS_PATH, ACTIVE_POSITION_COLUMNS)
    _ensure_csv(TRADES_PATH, TRADE_COLUMNS)
    _ensure_csv(AUDIT_PATH, AUDIT_COLUMNS)


def _ensure_csv(path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        pd.DataFrame(columns=columns).to_csv(path, index=False)


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    _ensure_csv(path, columns)
    frame = pd.read_csv(path)
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    return frame[columns]


@contextmanager
def _paper_state_lock() -> Any:
    ensure_dirs()
    with _LOCAL_LOCK:
        deadline = time.time() + 10
        while True:
            try:
                LOCK_PATH.mkdir()
                break
            except FileExistsError:
                if time.time() >= deadline:
                    raise RuntimeError(f"Timed out waiting for paper state lock: {LOCK_PATH}")
                time.sleep(0.05)
        try:
            yield
        finally:
            try:
                LOCK_PATH.rmdir()
            except FileNotFoundError:
                pass


def read_active_positions(open_only: bool = False) -> pd.DataFrame:
    frame = _read_csv(ACTIVE_POSITIONS_PATH, ACTIVE_POSITION_COLUMNS)
    if open_only and not frame.empty:
        frame = frame[frame["status"].astype(str).str.upper() == "OPEN"]
    return frame.reset_index(drop=True)


def read_trades() -> pd.DataFrame:
    return _read_csv(TRADES_PATH, TRADE_COLUMNS)


def read_audit() -> pd.DataFrame:
    return _read_csv(AUDIT_PATH, AUDIT_COLUMNS)


def read_signals(limit: int = 100) -> pd.DataFrame:
    if not SIGNALS_PATH.exists() or SIGNALS_PATH.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(SIGNALS_PATH).tail(limit).reset_index(drop=True)


def _append_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    current = _read_csv(path, columns)
    updated = pd.concat([current, pd.DataFrame(rows)], ignore_index=True)
    _write_csv(path, updated[columns])


def _write_active_positions(frame: pd.DataFrame) -> None:
    _write_csv(ACTIVE_POSITIONS_PATH, frame[ACTIVE_POSITION_COLUMNS])


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    frame.to_csv(tmp_path, index=False)
    last_error: Exception | None = None
    for _ in range(20):
        try:
            tmp_path.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.05)
    raise RuntimeError(f"Could not replace {path}") from last_error


def _normalize_action(signal: dict[str, Any]) -> str:
    raw = signal.get("signal", signal.get("action", signal.get("decision", signal.get("predicted_direction", "NO_TRADE"))))
    value = str(raw or "NO_TRADE").upper()
    if value in {"UP", "BUY"}:
        return "LONG"
    if value in {"DOWN", "SELL"}:
        return "SHORT"
    if value == "FLAT":
        return "NO_TRADE"
    return value


def _is_executable(signal: dict[str, Any], action: str) -> bool:
    explicit = signal.get("executable")
    if explicit is not None:
        return bool(explicit)
    decision = str(signal.get("decision", signal.get("status", ""))).upper()
    if decision in {"NO_TRADE", "BLOCKED", "REJECTED"}:
        return False
    return action in {"LONG", "SHORT"}


def _signal_price(signal: dict[str, Any]) -> float | None:
    for key in ("price", "entry_price", "close", "current_price"):
        value = signal.get(key)
        if value is None or value == "":
            continue
        try:
            price = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(price) and price > 0:
            return price
    return None


def _execution_price(symbol: str, signal: dict[str, Any]) -> float | None:
    market_price = latest_local_prices().get(symbol)
    if market_price is not None:
        return market_price
    return _signal_price(signal)


def _signal_execution_key(signal: dict[str, Any], action: str | None = None) -> tuple[str, str, str, str]:
    normalized_action = action or _normalize_action(signal)
    price = _signal_price(signal)
    price_key = "" if price is None else f"{price:.8f}"
    confidence = signal.get("confidence", 0.0)
    try:
        confidence_key = f"{float(confidence):.8f}"
    except (TypeError, ValueError):
        confidence_key = str(confidence)
    return (
        str(signal.get("symbol") or LAB_CONFIG.raw_symbol),
        normalized_action,
        confidence_key,
        price_key,
    )


def _audit_execution_keys(audit: pd.DataFrame) -> set[tuple[str, str, str, str]]:
    keys: set[tuple[str, str, str, str]] = set()
    if audit.empty:
        return keys
    for _, row in audit.iterrows():
        if str(row.get("executed", "")).lower() != "true":
            continue
        price = row.get("price")
        try:
            price_key = "" if pd.isna(price) else f"{float(price):.8f}"
        except (TypeError, ValueError):
            price_key = str(price)
        confidence = row.get("confidence", 0.0)
        try:
            confidence_key = f"{float(confidence):.8f}"
        except (TypeError, ValueError):
            confidence_key = str(confidence)
        keys.add((str(row.get("symbol", "")), str(row.get("signal", "")), confidence_key, price_key))
    return keys


def _position_id(symbol: str, side: str, opened_at: str, price: float) -> str:
    digest = hashlib.sha1(f"{symbol}|{side}|{opened_at}|{price:.8f}".encode("utf-8")).hexdigest()[:12]
    return f"pos_{digest}"


def _realized_balance(config: PaperTradingConfig) -> float:
    trades = read_trades()
    if trades.empty:
        return float(config.initial_balance)
    balances = pd.to_numeric(trades["balance_after"], errors="coerce").dropna()
    return float(balances.iloc[-1]) if not balances.empty else float(config.initial_balance)


def _audit(signal: dict[str, Any], action: str, executed: bool, reason: str, position_id: str = "") -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    price = _signal_price(signal)
    _append_csv(
        AUDIT_PATH,
        AUDIT_COLUMNS,
        [
            {
                "timestamp": timestamp,
                "symbol": signal.get("symbol", ""),
                "signal": action,
                "confidence": signal.get("confidence", 0.0),
                "price": price,
                "decision": signal.get("decision", signal.get("status", "")),
                "executed": bool(executed),
                "reason": reason,
                "position_id": position_id,
            }
        ],
    )


def open_position_from_signal(signal: dict[str, Any], config: PaperTradingConfig | None = None) -> dict[str, Any] | None:
    with _paper_state_lock():
        ensure_paper_files()
        config = config or load_paper_trading_config()
        action = _normalize_action(signal)
        symbol = str(signal.get("symbol") or LAB_CONFIG.raw_symbol)
        confidence = float(signal.get("confidence") or 0.0)

        if action == "NO_TRADE":
            _audit(signal, action, False, "NO_TRADE")
            return None
        if action not in {"LONG", "SHORT"} or not _is_executable(signal, action):
            _audit(signal, action, False, "RISK_BLOCKED")
            return None
        if confidence < config.confidence_threshold:
            _audit(signal, action, False, "LOW_CONFIDENCE")
            return None
        price = _execution_price(symbol, signal)
        if price is None:
            _audit(signal, action, False, "MISSING_PRICE")
            return None

        active = read_active_positions(open_only=True)
        if not active.empty and symbol in set(active["symbol"].astype(str)):
            _audit(signal, action, False, "ALREADY_OPEN_POSITION")
            return None

        if _signal_execution_key(signal, action) in _audit_execution_keys(read_audit()):
            return None

        opened_at = datetime.now(timezone.utc).isoformat()
        balance = _realized_balance(config)
        position_size = min(balance, balance * config.risk_per_trade)
        row = {
            "position_id": _position_id(symbol, action, opened_at, price),
            "opened_at": opened_at,
            "symbol": symbol,
            "side": action,
            "entry_price": price,
            "current_price": price,
            "position_size": position_size,
            "confidence": confidence,
            "regime": signal.get("regime", "UNKNOWN"),
            "model_version": signal.get("model_version", "baseline_v0.1"),
            "reason": signal.get("reason", f"paper_{action.lower()}_confidence_{confidence:.3f}"),
            "unrealized_pnl_usd": 0.0,
            "unrealized_pnl_pct": 0.0,
            "status": "OPEN",
        }
        positions = read_active_positions()
        positions = pd.concat([positions, pd.DataFrame([row])], ignore_index=True)
        _write_active_positions(positions)
        _audit(signal, action, True, "OPENED", row["position_id"])
        return row


def _unrealized_return(side: str, entry_price: float, current_price: float) -> float:
    if entry_price <= 0 or current_price <= 0:
        return 0.0
    if side.upper() == "SHORT":
        return current_price and (entry_price / current_price) - 1
    return (current_price / entry_price) - 1


def latest_local_prices() -> dict[str, float]:
    prices: dict[str, float] = {}
    raw_path = RAW_DIR / f"{LAB_CONFIG.raw_symbol}_{LAB_CONFIG.timeframe}_candles.parquet"
    if raw_path.exists():
        try:
            candles = read_parquet(raw_path)
            if not candles.empty:
                symbol = str(candles["symbol"].iloc[-1]) if "symbol" in candles.columns else LAB_CONFIG.raw_symbol
                prices[symbol] = float(candles["close"].iloc[-1])
                return prices
        except Exception:
            pass
    try:
        features = read_parquet(PROCESSED_DIR / f"{LAB_CONFIG.raw_symbol}_{LAB_CONFIG.timeframe}_features.parquet")
    except FileNotFoundError:
        return prices
    if features.empty:
        return prices
    symbol = str(features["symbol"].iloc[-1]) if "symbol" in features.columns else LAB_CONFIG.raw_symbol
    prices[symbol] = float(features["close"].iloc[-1])
    return prices


def update_open_positions(
    price_by_symbol: dict[str, float] | None = None,
    config: PaperTradingConfig | None = None,
) -> pd.DataFrame:
    with _paper_state_lock():
        ensure_paper_files()
        config = config or load_paper_trading_config()
        prices = price_by_symbol or latest_local_prices()
        positions = read_active_positions()
        if positions.empty:
            return positions

        trades = read_trades()
        existing_trade_ids = set(trades["position_id"].astype(str)) if not trades.empty else set()
        balance = _realized_balance(config)
        closed_rows: list[dict[str, Any]] = []
        now = datetime.now(timezone.utc)
        for idx, row in positions.iterrows():
            if str(row["status"]).upper() != "OPEN":
                continue
            symbol = str(row["symbol"])
            current_price = float(prices.get(symbol, row["current_price"]))
            entry_price = float(row["entry_price"])
            side = str(row["side"]).upper()
            unrealized = float(_unrealized_return(side, entry_price, current_price))
            position_size = float(row["position_size"])
            positions.at[idx, "current_price"] = current_price
            positions.at[idx, "unrealized_pnl_usd"] = position_size * unrealized
            positions.at[idx, "unrealized_pnl_pct"] = unrealized * 100

            opened_at = pd.Timestamp(row["opened_at"]).to_pydatetime()
            if opened_at.tzinfo is None:
                opened_at = opened_at.replace(tzinfo=timezone.utc)
            exit_reason = ""
            if current_price == entry_price:
                exit_reason = ""
            elif unrealized >= config.take_profit_pct / 100:
                exit_reason = "TAKE_PROFIT"
            elif unrealized <= -(config.stop_loss_pct / 100):
                exit_reason = "STOP_LOSS"
            elif now - opened_at >= timedelta(minutes=config.horizon_minutes):
                exit_reason = "HORIZON"
            if not exit_reason:
                continue

            position_id = str(row["position_id"])
            if position_id in existing_trade_ids:
                positions.at[idx, "status"] = "CLOSED"
                continue
            net_return = unrealized - config.fee_pct * 2 - config.slippage_pct
            pnl_usd = position_size * net_return
            balance += pnl_usd
            closed_rows.append(
                {
                    "position_id": position_id,
                    "opened_at": row["opened_at"],
                    "closed_at": now.isoformat(),
                    "symbol": symbol,
                    "side": side,
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "position_size": position_size,
                    "confidence": row["confidence"],
                    "pnl_usd": pnl_usd,
                    "pnl_pct": net_return * 100,
                    "balance_after": balance,
                    "reason": exit_reason,
                    "model_version": row["model_version"],
                }
            )
            existing_trade_ids.add(position_id)
            positions.at[idx, "status"] = "CLOSED"

        if closed_rows:
            _append_csv(TRADES_PATH, TRADE_COLUMNS, closed_rows)
        _write_active_positions(positions)
        return positions


def execute_latest_unaudited_signal() -> dict[str, Any] | None:
    signals = read_signals(limit=100)
    if signals.empty:
        return None
    audited_keys = _audit_execution_keys(read_audit())
    row = signals.iloc[-1].to_dict()
    action = _normalize_action(row)
    if _signal_execution_key(row, action) in audited_keys:
        return None
    return open_position_from_signal(row)


def stats_snapshot() -> dict[str, Any]:
    config = load_paper_trading_config()
    update_open_positions(config=config)
    active = read_active_positions(open_only=True)
    trades = read_trades()
    balance = _realized_balance(config)
    unrealized_pnl = float(pd.to_numeric(active["unrealized_pnl_usd"], errors="coerce").fillna(0).sum()) if not active.empty else 0.0
    realized_pnl = balance - config.initial_balance
    pnl_values = pd.to_numeric(trades["pnl_usd"], errors="coerce") if not trades.empty else pd.Series(dtype="float64")
    wins = int((pnl_values > 0).sum()) if not pnl_values.empty else 0
    losses = int((pnl_values <= 0).sum()) if not pnl_values.empty else 0
    trade_returns = pd.to_numeric(trades["pnl_pct"], errors="coerce") / 100 if not trades.empty else pd.Series(dtype="float64")
    balances = [config.initial_balance] + pd.to_numeric(trades["balance_after"], errors="coerce").dropna().astype(float).tolist()
    return {
        "balance": float(balance),
        "equity": float(balance + unrealized_pnl),
        "realized_pnl": float(realized_pnl),
        "unrealized_pnl": float(unrealized_pnl),
        "trades_count": int(len(trades)),
        "open_positions_count": int(len(active)),
        "wins": wins,
        "losses": losses,
        "winrate": float(wins / len(trades)) if len(trades) else 0.0,
        "profit_factor": profit_factor(trade_returns),
        "max_drawdown": max_drawdown(balances),
        "last_update": datetime.now(timezone.utc).isoformat(),
    }


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    clean = frame.where(pd.notna(frame), None)
    return clean.to_dict(orient="records")
