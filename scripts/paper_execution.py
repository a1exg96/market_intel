from __future__ import annotations

import hashlib
import math
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from scripts.common import CONFIG_DIR, LAB_CONFIG, PROCESSED_DIR, RAW_DIR, REPORTS_DIR, ensure_dirs, market_symbols, max_drawdown, profit_factor, read_parquet
from scripts.protections import evaluate_protections

MAX_CONFIG_RISK_PER_TRADE = 0.005

ACTIVE_POSITION_COLUMNS = [
    "position_id",
    "opened_at",
    "symbol",
    "side",
    "entry_price",
    "current_price",
    "position_size",
    "confidence",
    "expected_return",
    "expected_risk",
    "risk_reward",
    "stop_loss",
    "take_profit",
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
    "expected_return",
    "expected_risk",
    "risk_reward",
    "stop_loss",
    "take_profit",
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
LOCK_TIMEOUT_SECONDS = 10
STALE_LOCK_SECONDS = 120


@dataclass(frozen=True)
class PaperTradingConfig:
    live_trading: bool = False
    initial_balance: float = LAB_CONFIG.initial_balance
    risk_per_trade: float = LAB_CONFIG.risk_per_trade
    leverage: float = 1.0
    horizon_minutes: int = 240
    take_profit_pct: float = 1.0
    stop_loss_pct: float = 0.7
    liquidation_long_pct: float = 0.7
    liquidation_short_pct: float = 0.7
    fee_pct: float = LAB_CONFIG.fee_pct
    slippage_pct: float = LAB_CONFIG.slippage_pct
    confidence_threshold: float = LAB_CONFIG.confidence_threshold
    min_expected_return: float = 0.0005
    min_risk_reward: float = 1.5
    min_entry_quality: float = 0.0
    cooldown_minutes: int = 180
    loss_streak_reduce_after: int = 1
    max_consecutive_losses: int = 2
    max_daily_loss_pct: float = 0.02
    max_daily_trades: int = 6
    symbol_loss_cooldown_minutes: int = 360
    max_drawdown_live_block: float = 0.10


def load_paper_trading_config(path: Path | None = None) -> PaperTradingConfig:
    config_path = path or CONFIG_DIR / "paper_trading.yaml"
    raw: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
            if not isinstance(loaded, dict):
                raise RuntimeError(f"Invalid paper trading config: {config_path}")
            raw = loaded
    if "stake_pct" in raw and "risk_per_trade" not in raw:
        raw["risk_per_trade"] = float(raw["stake_pct"]) / 100
    fields = PaperTradingConfig.__dataclass_fields__
    config = PaperTradingConfig(**{k: v for k, v in raw.items() if k in fields})
    if config.live_trading:
        raise RuntimeError("ERROR: live_trading=true is forbidden. Paper engine refuses to start.")
    if config.leverage <= 0:
        raise RuntimeError("ERROR: leverage must be positive.")
    if config.risk_per_trade <= 0:
        raise RuntimeError("ERROR: risk_per_trade must be positive.")
    if config.risk_per_trade > MAX_CONFIG_RISK_PER_TRADE:
        config = replace(config, risk_per_trade=MAX_CONFIG_RISK_PER_TRADE)
    if config.liquidation_long_pct <= 0 or config.liquidation_short_pct <= 0:
        raise RuntimeError("ERROR: liquidation percentages must be positive.")
    return config


def paper_trading_settings(path: Path | None = None) -> dict[str, Any]:
    config = load_paper_trading_config(path)
    return {
        "leverage": float(config.leverage),
        "stake_pct": float(config.risk_per_trade * 100),
        "liquidation_long_pct": float(config.liquidation_long_pct),
        "liquidation_short_pct": float(config.liquidation_short_pct),
        "take_profit_pct": float(config.take_profit_pct),
        "horizon_minutes": int(config.horizon_minutes),
        "confidence_threshold": float(config.confidence_threshold),
        "min_entry_quality": float(config.min_entry_quality),
        "max_daily_loss_pct": float(config.max_daily_loss_pct),
        "max_daily_trades": int(config.max_daily_trades),
    }


def update_paper_trading_settings(values: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    config_path = path or CONFIG_DIR / "paper_trading.yaml"
    raw: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
            if not isinstance(loaded, dict):
                raise RuntimeError(f"Invalid paper trading config: {config_path}")
            raw = loaded

    def positive_float(key: str, minimum: float, maximum: float | None = None) -> float | None:
        if key not in values:
            return None
        try:
            value = float(values[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a number") from exc
        if value < minimum or (maximum is not None and value > maximum):
            limit = f" between {minimum} and {maximum}" if maximum is not None else f" at least {minimum}"
            raise ValueError(f"{key} must be{limit}")
        return value

    leverage = positive_float("leverage", 0.1, 125)
    if leverage is not None:
        raw["leverage"] = leverage

    stake_pct = positive_float("stake_pct", 0.01, 1)
    if stake_pct is not None:
        raw["risk_per_trade"] = stake_pct / 100

    confidence_threshold = positive_float("confidence_threshold", 0.0, 1.0)
    if confidence_threshold is not None:
        raw["confidence_threshold"] = confidence_threshold

    min_entry_quality = positive_float("min_entry_quality", 0.0, 1.0)
    if min_entry_quality is not None:
        raw["min_entry_quality"] = min_entry_quality

    liquidation_long_pct = positive_float("liquidation_long_pct", 0.01, 100)
    if liquidation_long_pct is not None:
        raw["liquidation_long_pct"] = liquidation_long_pct

    liquidation_short_pct = positive_float("liquidation_short_pct", 0.01, 100)
    if liquidation_short_pct is not None:
        raw["liquidation_short_pct"] = liquidation_short_pct

    config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = config_path.with_name(f".{config_path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)
    tmp_path.replace(config_path)
    return paper_trading_settings(config_path)


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
        deadline = time.time() + LOCK_TIMEOUT_SECONDS
        while True:
            try:
                LOCK_PATH.mkdir()
                break
            except FileExistsError:
                if _lock_is_stale():
                    _remove_stale_lock()
                    continue
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


def _lock_is_stale() -> bool:
    try:
        age = time.time() - LOCK_PATH.stat().st_mtime
    except FileNotFoundError:
        return False
    except OSError:
        return False
    return age > STALE_LOCK_SECONDS


def _remove_stale_lock() -> None:
    try:
        LOCK_PATH.rmdir()
    except FileNotFoundError:
        return
    except OSError:
        return


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
    frame = pd.read_csv(SIGNALS_PATH)
    if frame.empty:
        return frame
    if {"timestamp", "symbol"}.issubset(frame.columns):
        normalized_timestamp = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce", format="mixed")
        frame = frame.assign(
            _timestamp_key=normalized_timestamp.dt.strftime("%Y-%m-%dT%H:%M:%S%z"),
            _symbol_key=frame["symbol"].astype(str),
            _timestamp_sort=normalized_timestamp,
        )
        sort_columns = ["_timestamp_sort"]
        if "generated_at" in frame.columns:
            frame["_generated_sort"] = pd.to_datetime(frame["generated_at"], utc=True, errors="coerce", format="mixed")
            sort_columns.append("_generated_sort")
        frame = frame.sort_values(sort_columns, na_position="last")
        frame = frame.drop_duplicates(["_timestamp_key", "_symbol_key"], keep="last")
        frame = frame.drop(columns=[column for column in frame.columns if column.startswith("_")])
    return frame.tail(limit).reset_index(drop=True)


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


def _signal_float(signal: dict[str, Any], key: str, default: float | None = None) -> float | None:
    value = signal.get(key)
    if value is None or value == "":
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


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


def _risk_multiplier_from_history(trades: pd.DataFrame, config: PaperTradingConfig) -> float:
    if trades.empty:
        return 1.0
    pnl = pd.to_numeric(trades["pnl_usd"], errors="coerce").fillna(0.0)
    streak = 0
    for value in reversed(pnl.tolist()):
        if value < 0:
            streak += 1
        else:
            break
    if streak >= config.loss_streak_reduce_after + 2:
        return 0.25
    if streak >= config.loss_streak_reduce_after:
        return 0.5
    return 1.0


def _stop_distance(side: str, entry_price: float, stop_loss: float) -> float:
    if entry_price <= 0 or stop_loss <= 0:
        return 0.0
    if side == "SHORT":
        return (stop_loss / entry_price) - 1
    return 1 - (stop_loss / entry_price)


def _take_profit_distance(side: str, entry_price: float, take_profit: float) -> float:
    if entry_price <= 0 or take_profit <= 0:
        return 0.0
    if side == "SHORT":
        return 1 - (take_profit / entry_price)
    return (take_profit / entry_price) - 1


def _liquidation_buffer_is_safe(side: str, entry_price: float, stop_loss: float, config: PaperTradingConfig) -> bool:
    leverage = max(float(config.leverage), 1.0)
    liquidation_distance = min(0.95 / leverage, 0.95)
    if side == "SHORT":
        liquidation_price = entry_price * (1 + liquidation_distance)
        return liquidation_price > stop_loss and (liquidation_price - stop_loss) / entry_price >= 0.005
    liquidation_price = entry_price * (1 - liquidation_distance)
    return liquidation_price < stop_loss and (stop_loss - liquidation_price) / entry_price >= 0.005


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

        expected_return = _signal_float(signal, "expected_return")
        expected_risk = _signal_float(signal, "expected_risk")
        risk_reward = _signal_float(signal, "risk_reward")
        stop_loss = _signal_float(signal, "stop_loss")
        take_profit = _signal_float(signal, "take_profit")
        if expected_return is None or expected_return < config.min_expected_return:
            _audit(signal, action, False, "NON_POSITIVE_EXPECTANCY")
            return None
        if risk_reward is None or risk_reward < config.min_risk_reward:
            _audit(signal, action, False, "LOW_RISK_REWARD")
            return None
        entry_quality = _signal_float(signal, "entry_quality")
        if entry_quality is not None and entry_quality < config.min_entry_quality:
            _audit(signal, action, False, "LOW_ENTRY_QUALITY")
            return None
        if stop_loss is None or take_profit is None:
            _audit(signal, action, False, "MISSING_RISK_PLAN")
            return None
        stop_distance = _stop_distance(action, price, stop_loss)
        take_profit_distance = _take_profit_distance(action, price, take_profit)
        if stop_distance <= 0 or take_profit_distance <= 0:
            _audit(signal, action, False, "INVALID_TP_SL")
            return None
        if take_profit_distance / max(stop_distance, 1e-9) < config.min_risk_reward:
            _audit(signal, action, False, "LOW_TP_SL_RISK_REWARD")
            return None
        if not _liquidation_buffer_is_safe(action, price, stop_loss, config):
            _audit(signal, action, False, "STOP_TOO_CLOSE_TO_LIQUIDATION")
            return None

        active = read_active_positions(open_only=True)
        if not active.empty and symbol in set(active["symbol"].astype(str)):
            _audit(signal, action, False, "ALREADY_OPEN_POSITION")
            return None

        if _signal_execution_key(signal, action) in _audit_execution_keys(read_audit()):
            return None

        opened_at = datetime.now(timezone.utc).isoformat()
        balance = _realized_balance(config)
        trades = read_trades()
        protection = evaluate_protections(symbol, trades, config)
        if not protection.allowed:
            _audit(signal, action, False, protection.reason)
            return None
        risk_fraction = min(float(config.risk_per_trade), MAX_CONFIG_RISK_PER_TRADE) * _risk_multiplier_from_history(trades, config)
        risk_budget = balance * risk_fraction
        suggested_size = _signal_float(signal, "position_size", 0.0) or 0.0
        risk_sized_position = risk_budget / max(stop_distance * float(config.leverage), 1e-9)
        position_size = min(balance, risk_sized_position, suggested_size if suggested_size > 0 else balance)
        row = {
            "position_id": _position_id(symbol, action, opened_at, price),
            "opened_at": opened_at,
            "symbol": symbol,
            "side": action,
            "entry_price": price,
            "current_price": price,
            "position_size": position_size,
            "confidence": confidence,
            "expected_return": expected_return,
            "expected_risk": expected_risk,
            "risk_reward": risk_reward,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "regime": signal.get("regime", "UNKNOWN"),
            "model_version": signal.get("model_version", "dual_setup_v0.3_edge_score"),
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


def _leveraged_return(unrealized: float, config: PaperTradingConfig) -> float:
    return unrealized * float(config.leverage)


def _round_trip_cost(config: PaperTradingConfig) -> float:
    return (float(config.fee_pct) * 2 + float(config.slippage_pct)) * float(config.leverage)


def _liquidation_pct(side: str, config: PaperTradingConfig) -> float:
    if side.upper() == "SHORT":
        return float(config.liquidation_short_pct)
    return float(config.liquidation_long_pct)


def latest_local_prices() -> dict[str, float]:
    prices: dict[str, float] = {}
    for symbol in market_symbols():
        raw_path = RAW_DIR / f"{symbol}_{LAB_CONFIG.timeframe}_candles.parquet"
        if not raw_path.exists():
            continue
        try:
            candles = read_parquet(raw_path)
            if not candles.empty:
                candle_symbol = str(candles["symbol"].iloc[-1]) if "symbol" in candles.columns else symbol
                prices[candle_symbol] = float(candles["close"].iloc[-1])
        except Exception:
            pass
    if prices:
        return prices
    for symbol in market_symbols():
        try:
            features = read_parquet(PROCESSED_DIR / f"{symbol}_{LAB_CONFIG.timeframe}_features.parquet")
        except FileNotFoundError:
            continue
        if features.empty:
            continue
        feature_symbol = str(features["symbol"].iloc[-1]) if "symbol" in features.columns else symbol
        prices[feature_symbol] = float(features["close"].iloc[-1])
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
            leveraged_unrealized = _leveraged_return(unrealized, config)
            position_size = float(row["position_size"])
            positions.at[idx, "current_price"] = current_price
            positions.at[idx, "unrealized_pnl_usd"] = position_size * leveraged_unrealized
            positions.at[idx, "unrealized_pnl_pct"] = leveraged_unrealized * 100

            opened_at = pd.Timestamp(row["opened_at"]).to_pydatetime()
            if opened_at.tzinfo is None:
                opened_at = opened_at.replace(tzinfo=timezone.utc)
            exit_reason = ""
            stop_loss = _signal_float(row.to_dict(), "stop_loss", 0.0) or 0.0
            take_profit = _signal_float(row.to_dict(), "take_profit", 0.0) or 0.0
            if side == "SHORT" and take_profit > 0 and current_price <= take_profit:
                exit_reason = "TAKE_PROFIT"
            elif side == "SHORT" and stop_loss > 0 and current_price >= stop_loss:
                exit_reason = "STOP_LOSS"
            elif side != "SHORT" and take_profit > 0 and current_price >= take_profit:
                exit_reason = "TAKE_PROFIT"
            elif side != "SHORT" and stop_loss > 0 and current_price <= stop_loss:
                exit_reason = "STOP_LOSS"
            elif current_price == entry_price:
                exit_reason = ""
            elif leveraged_unrealized >= config.take_profit_pct / 100:
                exit_reason = "TAKE_PROFIT"
            elif leveraged_unrealized <= -(config.stop_loss_pct / 100):
                exit_reason = "STOP_LOSS"
            elif now - opened_at >= timedelta(minutes=config.horizon_minutes):
                exit_reason = "HORIZON"
            if not exit_reason:
                continue

            position_id = str(row["position_id"])
            if position_id in existing_trade_ids:
                positions.at[idx, "status"] = "CLOSED"
                continue
            net_return = leveraged_unrealized - _round_trip_cost(config)
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
                    "expected_return": row.get("expected_return", 0.0),
                    "expected_risk": row.get("expected_risk", 0.0),
                    "risk_reward": row.get("risk_reward", 0.0),
                    "stop_loss": row.get("stop_loss", 0.0),
                    "take_profit": row.get("take_profit", 0.0),
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


def close_position_manually(position_id: str, config: PaperTradingConfig | None = None) -> dict[str, Any]:
    with _paper_state_lock():
        ensure_paper_files()
        config = config or load_paper_trading_config()
        positions = read_active_positions()
        if positions.empty:
            raise KeyError(position_id)

        matches = positions.index[
            (positions["position_id"].astype(str) == str(position_id))
            & (positions["status"].astype(str).str.upper() == "OPEN")
        ].tolist()
        if not matches:
            raise KeyError(position_id)

        idx = matches[0]
        row = positions.loc[idx]
        symbol = str(row["symbol"])
        current_price = float(latest_local_prices().get(symbol, row["current_price"]))
        entry_price = float(row["entry_price"])
        side = str(row["side"]).upper()
        position_size = float(row["position_size"])
        leveraged_unrealized = _leveraged_return(_unrealized_return(side, entry_price, current_price), config)
        net_return = leveraged_unrealized - _round_trip_cost(config)
        pnl_usd = position_size * net_return
        balance_after = _realized_balance(config) + pnl_usd
        closed_at = datetime.now(timezone.utc).isoformat()
        trade = {
            "position_id": str(position_id),
            "opened_at": row["opened_at"],
            "closed_at": closed_at,
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "exit_price": current_price,
            "position_size": position_size,
            "confidence": row["confidence"],
            "expected_return": row.get("expected_return", 0.0),
            "expected_risk": row.get("expected_risk", 0.0),
            "risk_reward": row.get("risk_reward", 0.0),
            "stop_loss": row.get("stop_loss", 0.0),
            "take_profit": row.get("take_profit", 0.0),
            "pnl_usd": pnl_usd,
            "pnl_pct": net_return * 100,
            "balance_after": balance_after,
            "reason": "MANUAL",
            "model_version": row["model_version"],
        }
        positions.at[idx, "current_price"] = current_price
        positions.at[idx, "unrealized_pnl_usd"] = position_size * leveraged_unrealized
        positions.at[idx, "unrealized_pnl_pct"] = leveraged_unrealized * 100
        positions.at[idx, "status"] = "CLOSED"
        _append_csv(TRADES_PATH, TRADE_COLUMNS, [trade])
        _write_active_positions(positions)
        return trade


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
        "settings": paper_trading_settings(),
    }


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict(orient="records")
