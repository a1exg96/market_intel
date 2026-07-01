from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scripts.common import LAB_CONFIG, PROCESSED_DIR, REPORTS_DIR, ensure_dirs, read_parquet
from scripts.execution_engine import HORIZON_ROWS
from scripts.paper_execution import execute_latest_unaudited_signal, latest_local_prices, open_position_from_signal, update_open_positions

LOGGER = logging.getLogger(__name__)
SIGNAL_COLUMNS = ["timestamp", "generated_at", "symbol", "regime", "signal", "confidence", "entry_price", "horizon", "status"]
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


def run_forward_paper_engine() -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-shot forward-paper processor.

    In live use this command can be scheduled every candle. It records signals without
    outcome fields first, then settles only signals whose horizon has elapsed.
    """
    ensure_dirs()
    features = read_parquet(PROCESSED_DIR / f"{LAB_CONFIG.raw_symbol}_{LAB_CONFIG.timeframe}_features.parquet")
    predictions = read_parquet(PROCESSED_DIR / "predictions.parquet")
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], utc=True)
    features = features.sort_values("timestamp").reset_index(drop=True)
    predictions = predictions.sort_values("timestamp").reset_index(drop=True)
    regime_path = PROCESSED_DIR / "regime_labels.parquet"
    regimes = read_parquet(regime_path) if regime_path.exists() else pd.DataFrame(columns=["timestamp", "symbol", "regime"])
    if not regimes.empty:
        regimes["timestamp"] = pd.to_datetime(regimes["timestamp"], utc=True)

    signal_path = REPORTS_DIR / "forward_signals.csv"
    result_path = REPORTS_DIR / "forward_results.csv"
    existing_signals = _dedupe_by_symbol_time(_read_csv(signal_path, SIGNAL_COLUMNS), SIGNAL_COLUMNS)
    existing_results = _dedupe_by_symbol_time(_read_csv(result_path, RESULT_COLUMNS), RESULT_COLUMNS)
    seen = {_row_key(row) for _, row in existing_signals.iterrows()} if not existing_signals.empty else set()

    feature_index = {ts: i for i, ts in enumerate(features["timestamp"])}
    new_signals: list[dict] = []
    for _, row in predictions.iterrows():
        ts = row["timestamp"]
        if (str(row.get("symbol", "")), _timestamp_key(ts)) in seen:
            continue
        idx = feature_index.get(ts)
        if idx is None:
            continue
        direction = str(row["predicted_direction"])
        signal_name = "LONG" if direction == "up" else ("SHORT" if direction == "down" else "NO_TRADE")
        market_price = latest_local_prices().get(str(row["symbol"]))
        if signal_name == "NO_TRADE":
            entry_price = float(features.loc[idx, "close"])
        elif idx + 1 < len(features):
            entry_price = float(features.loc[idx + 1, "open"])
        else:
            entry_price = float(market_price if market_price is not None else features.loc[idx, "close"])
        regime = "UNKNOWN"
        if not regimes.empty:
            prior = regimes[regimes["timestamp"] <= ts].tail(1)
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
                "entry_price": entry_price,
                "horizon": row.get("horizon", "4h"),
                "status": "PENDING",
            }
        )

    signals = _dedupe_by_symbol_time(pd.concat([existing_signals, pd.DataFrame(new_signals)], ignore_index=True), SIGNAL_COLUMNS)
    result_seen = {_row_key(row) for _, row in existing_results.iterrows()} if not existing_results.empty else set()
    balance = LAB_CONFIG.initial_balance if existing_results.empty else float(existing_results["balance"].iloc[-1])
    new_results: list[dict] = []
    for _, signal in signals.iterrows():
        if str(signal.get("signal", "")).upper() == "NO_TRADE":
            continue
        if _row_key(signal) in result_seen:
            continue
        ts = pd.Timestamp(signal["timestamp"])
        idx = feature_index.get(ts)
        if idx is None:
            continue
        horizon_rows = HORIZON_ROWS.get(str(signal.get("horizon", "4h")), 48)
        exit_idx = idx + 1 + horizon_rows
        if exit_idx >= len(features):
            continue
        entry = float(signal["entry_price"])
        exit_price = float(features.loc[exit_idx, "close"])
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
    results = _dedupe_by_symbol_time(pd.concat([existing_results, pd.DataFrame(new_results)], ignore_index=True), RESULT_COLUMNS)
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
