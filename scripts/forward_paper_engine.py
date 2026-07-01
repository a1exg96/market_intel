from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scripts.common import LAB_CONFIG, PROCESSED_DIR, REPORTS_DIR, ensure_dirs, market_symbols, read_parquet
from scripts.execution_engine import HORIZON_ROWS
from scripts.paper_execution import execute_latest_unaudited_signal, latest_local_prices, open_position_from_signal, update_open_positions

LOGGER = logging.getLogger(__name__)
SIGNAL_COLUMNS = [
    "timestamp",
    "generated_at",
    "symbol",
    "regime",
    "signal",
    "confidence",
    "expected_return",
    "expected_risk",
    "risk_reward",
    "volatility_state",
    "entry_quality",
    "position_size",
    "stop_loss",
    "take_profit",
    "reason",
    "entry_price",
    "horizon",
    "status",
]
RESULT_COLUMNS = ["timestamp", "symbol", "regime", "signal", "confidence", "entry_price", "exit_price", "future_return", "trade_return", "pnl", "balance"]


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path)
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    return frame[columns]


def _timestamp_key(value: object) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


def _row_key(row: pd.Series | dict) -> tuple[str, str]:
    return str(row.get("symbol", "")), _timestamp_key(row.get("timestamp"))


def _dedupe_by_symbol_time(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty or not {"timestamp", "symbol"}.issubset(frame.columns):
        return frame.reindex(columns=columns)
    clean = frame.copy()
    parsed_timestamp = pd.to_datetime(clean["timestamp"], utc=True, errors="coerce", format="mixed")
    clean["_timestamp_key"] = parsed_timestamp.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    clean["_timestamp_sort"] = parsed_timestamp
    clean["_symbol_key"] = clean["symbol"].astype(str)
    sort_columns = ["_timestamp_sort"]
    if "generated_at" in clean.columns:
        clean["_generated_sort"] = pd.to_datetime(clean["generated_at"], utc=True, errors="coerce", format="mixed")
        sort_columns.append("_generated_sort")
    clean = clean.sort_values(sort_columns, na_position="last")
    clean = clean.drop_duplicates(["_symbol_key", "_timestamp_key"], keep="last")
    clean = clean.drop(columns=[column for column in clean.columns if column.startswith("_")])
    return clean.reindex(columns=columns).reset_index(drop=True)


def _limit_rows(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit <= 0 or frame.empty or len(frame) <= limit:
        return frame
    return frame.tail(limit).reset_index(drop=True)


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, str(default))))
    except ValueError:
        LOGGER.warning("Invalid integer for %s; using %s", name, default)
        return default


def run_forward_paper_engine() -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-shot forward-paper processor.

    In live use this command can be scheduled every candle. It records signals without
    outcome fields first, then settles only signals whose horizon has elapsed.
    """
    ensure_dirs()
    features = pd.concat(
        [read_parquet(PROCESSED_DIR / f"{symbol}_{LAB_CONFIG.timeframe}_features.parquet") for symbol in market_symbols()],
        ignore_index=True,
    )
    predictions = read_parquet(PROCESSED_DIR / "predictions.parquet")
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], utc=True)
    features = features.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    predictions = predictions.sort_values("timestamp").reset_index(drop=True)
    regime_path = PROCESSED_DIR / "regime_labels.parquet"
    regimes = read_parquet(regime_path) if regime_path.exists() else pd.DataFrame(columns=["timestamp", "symbol", "regime"])
    if not regimes.empty:
        regimes["timestamp"] = pd.to_datetime(regimes["timestamp"], utc=True)

    signal_path = REPORTS_DIR / "forward_signals.csv"
    result_path = REPORTS_DIR / "forward_results.csv"
    existing_signals = _dedupe_by_symbol_time(_read_csv(signal_path, SIGNAL_COLUMNS), SIGNAL_COLUMNS)
    existing_results = _dedupe_by_symbol_time(_read_csv(result_path, RESULT_COLUMNS), RESULT_COLUMNS)
    seen = set()
    if not existing_signals.empty:
        for _, existing in existing_signals.iterrows():
            has_score_payload = "reason" in existing and pd.notna(existing.get("reason")) and str(existing.get("reason", "")).strip() != ""
            if has_score_payload:
                seen.add(_row_key(existing))

    feature_index = {(str(row["symbol"]), row["timestamp"]): i for i, row in features.iterrows()}
    new_signals: list[dict] = []
    for _, row in predictions.iterrows():
        ts = row["timestamp"]
        if (str(row.get("symbol", "")), _timestamp_key(ts)) in seen:
            continue
        symbol = str(row.get("symbol", ""))
        idx = feature_index.get((symbol, ts))
        if idx is None:
            continue
        side = str(row.get("side", "") or "").upper()
        direction = str(row["predicted_direction"])
        signal_name = side if side in {"LONG", "SHORT", "NO_TRADE"} else ("LONG" if direction == "up" else ("SHORT" if direction == "down" else "NO_TRADE"))
        market_price = latest_local_prices().get(str(row["symbol"]))
        if signal_name == "NO_TRADE":
            entry_price = float(features.loc[idx, "close"])
        else:
            symbol_features = features[features["symbol"].astype(str) == symbol].reset_index(drop=True)
            symbol_idx = symbol_features.index[symbol_features["timestamp"] == ts]
            if len(symbol_idx) and int(symbol_idx[0]) + 1 < len(symbol_features):
                entry_price = float(symbol_features.loc[int(symbol_idx[0]) + 1, "open"])
            else:
                entry_price = float(market_price if market_price is not None else features.loc[idx, "close"])
        regime = "UNKNOWN"
        if not regimes.empty:
            prior = regimes[(regimes["symbol"].astype(str) == symbol) & (regimes["timestamp"] <= ts)].tail(1)
            if not prior.empty:
                regime = str(prior["regime"].iloc[0])
        new_signals.append(
            {
                "timestamp": ts.isoformat(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "symbol": row["symbol"],
                "regime": regime,
                "signal": signal_name,
                "confidence": float(row["confidence"]),
                "expected_return": float(row.get("expected_return", 0.0) or 0.0),
                "expected_risk": float(row.get("expected_risk", 0.0) or 0.0),
                "risk_reward": float(row.get("risk_reward", 0.0) or 0.0),
                "volatility_state": row.get("volatility_state", "UNKNOWN"),
                "entry_quality": float(row.get("entry_quality", 0.0) or 0.0),
                "position_size": float(row.get("position_size", 0.0) or 0.0),
                "stop_loss": float(row.get("stop_loss", 0.0) or 0.0),
                "take_profit": float(row.get("take_profit", 0.0) or 0.0),
                "reason": row.get("reason", ""),
                "entry_price": entry_price,
                "horizon": row.get("horizon", "4h"),
                "status": "EXECUTABLE" if signal_name in {"LONG", "SHORT"} else "NO_TRADE",
            }
        )

    max_signal_rows = _env_int("MARKET_INTEL_MAX_FORWARD_SIGNAL_ROWS", 750)
    max_result_rows = _env_int("MARKET_INTEL_MAX_FORWARD_RESULT_ROWS", 750)
    signals = _limit_rows(
        _dedupe_by_symbol_time(pd.concat([existing_signals, pd.DataFrame(new_signals)], ignore_index=True), SIGNAL_COLUMNS),
        max_signal_rows,
    )
    if not signals.empty and "reason" in signals:
        missing_reason = signals["reason"].isna() | (signals["reason"].astype(str).str.strip() == "")
        signals.loc[missing_reason, "reason"] = "LEGACY_SIGNAL_NO_SCORING_PAYLOAD"
        signals.loc[missing_reason & (signals["signal"].astype(str).str.upper() == "NO_TRADE"), "status"] = "NO_TRADE"
    result_seen = {_row_key(row) for _, row in existing_results.iterrows()} if not existing_results.empty else set()
    balance = LAB_CONFIG.initial_balance if existing_results.empty else float(existing_results["balance"].iloc[-1])
    new_results: list[dict] = []
    for _, signal in signals.iterrows():
        if str(signal.get("signal", "")).upper() == "NO_TRADE":
            continue
        if _row_key(signal) in result_seen:
            continue
        ts = pd.Timestamp(signal["timestamp"])
        symbol = str(signal.get("symbol", ""))
        idx = feature_index.get((symbol, ts))
        if idx is None:
            continue
        symbol_features = features[features["symbol"].astype(str) == symbol].reset_index(drop=True)
        symbol_idx = symbol_features.index[symbol_features["timestamp"] == ts]
        if not len(symbol_idx):
            continue
        horizon_rows = HORIZON_ROWS.get(str(signal.get("horizon", "4h")), 48)
        exit_idx = int(symbol_idx[0]) + 1 + horizon_rows
        if exit_idx >= len(symbol_features):
            continue
        entry = float(signal["entry_price"])
        exit_price = float(symbol_features.loc[exit_idx, "close"])
        gross = exit_price / entry - 1
        signed = gross if signal["signal"] == "LONG" else -gross
        trade_return = signed - LAB_CONFIG.fee_pct * 2 - LAB_CONFIG.slippage_pct
        pnl = balance * LAB_CONFIG.risk_per_trade * trade_return / max(abs(trade_return), 0.002)
        balance += pnl
        new_results.append(
            {
                "timestamp": signal["timestamp"],
                "symbol": signal["symbol"],
                "regime": signal["regime"],
                "signal": signal["signal"],
                "confidence": float(signal["confidence"]),
                "entry_price": entry,
                "exit_price": exit_price,
                "future_return": gross,
                "trade_return": trade_return,
                "pnl": pnl,
                "balance": balance,
            }
        )
    results = _limit_rows(
        _dedupe_by_symbol_time(pd.concat([existing_results, pd.DataFrame(new_results)], ignore_index=True), RESULT_COLUMNS),
        max_result_rows,
    )
    signals.to_csv(signal_path, index=False)
    results.to_csv(result_path, index=False)
    opened = None
    for signal in new_signals:
        if signal["signal"] == "NO_TRADE":
            continue
        opened = open_position_from_signal({**signal, "decision": "EXECUTABLE"})
        if opened:
            break
    if opened is None:
        execute_latest_unaudited_signal()
    update_open_positions()
    LOGGER.info("Saved forward paper signals=%s results=%s", len(signals), len(results))
    return signals, results


if __name__ == "__main__":
    from scripts.common import setup_logging

    setup_logging()
    run_forward_paper_engine()
