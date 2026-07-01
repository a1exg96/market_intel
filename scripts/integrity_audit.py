from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Callable

import numpy as np
import pandas as pd

from scripts.baseline_ml import walk_forward
from scripts.common import LAB_CONFIG, PROCESSED_DIR, REPORTS_DIR, ensure_dirs, read_parquet
from scripts.feature_engineering import feature_columns
from scripts.paper_trader import compute_stats, simulate

LOGGER = logging.getLogger(__name__)
HORIZON_TO_ROWS = {"1h": 12, "4h": 48, "24h": 288}


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features_path = PROCESSED_DIR / f"{LAB_CONFIG.raw_symbol}_{LAB_CONFIG.timeframe}_features.parquet"
    predictions_path = PROCESSED_DIR / "predictions.parquet"
    trades_path = REPORTS_DIR / "trades.csv"
    if not features_path.exists():
        raise FileNotFoundError("Missing features parquet. Run `python scripts/main.py features` first.")
    features = read_parquet(features_path)
    predictions = read_parquet(predictions_path) if predictions_path.exists() else pd.DataFrame()
    trades = pd.read_csv(trades_path, parse_dates=["timestamp"]) if trades_path.exists() else pd.DataFrame()
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)
    if not predictions.empty:
        predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], utc=True)
    if not trades.empty:
        trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True)
    return features, predictions, trades


def _future_columns(df: pd.DataFrame) -> list[str]:
    tokens = ("future_", "target_", "long_target_", "short_target_", "max_up", "max_down")
    return [col for col in df.columns if any(token in col for token in tokens)]


def _leakage_checks(df: pd.DataFrame, used_features: list[str]) -> dict[str, object]:
    future_cols = _future_columns(df)
    feature_leaks = [col for col in used_features if col in future_cols or any(token in col for token in ("future", "target", "max_up", "max_down"))]
    target_cols = [col for col in df.columns if col.startswith(("long_target_", "short_target_", "trade_target_"))]
    return {
        "used_features_count": len(used_features),
        "future_columns_present_as_labels": future_cols,
        "target_columns_present": target_cols,
        "feature_leakage_columns": feature_leaks,
        "feature_leakage_detected": bool(feature_leaks),
    }


def _walk_forward_checks(df: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, object]:
    horizon = str(predictions["horizon"].iloc[0]) if not predictions.empty and "horizon" in predictions else "4h"
    horizon_rows = HORIZON_TO_ROWS.get(horizon, 48)
    embargo_ok = LAB_CONFIG.embargo_rows >= horizon_rows
    monotonic = bool(df["timestamp"].is_monotonic_increasing)
    duplicate_timestamps = int(df["timestamp"].duplicated().sum())
    return {
        "horizon": horizon,
        "required_embargo_rows": horizon_rows,
        "configured_embargo_rows": LAB_CONFIG.embargo_rows,
        "embargo_covers_horizon": embargo_ok,
        "timestamps_monotonic": monotonic,
        "duplicate_timestamps": duplicate_timestamps,
    }


def _trade_audit(features: pd.DataFrame, predictions: pd.DataFrame, trades: pd.DataFrame, used_features: list[str]) -> pd.DataFrame:
    if trades.empty or predictions.empty:
        out = pd.DataFrame(
            columns=[
                "signal_timestamp",
                "entry_timestamp",
                "exit_timestamp",
                "symbol",
                "side",
                "all_features_used",
                "future_columns_used",
                "trade_return_formula",
                "raw_pnl_calculation",
                "reported_pnl_usd",
                "recomputed_pnl_usd",
                "pnl_matches",
                "execution_warning",
            ]
        )
        out.to_csv(REPORTS_DIR / "integrity_trade_audit.csv", index=False)
        return out

    merged = trades.merge(
        predictions,
        on=["timestamp", "symbol"],
        how="left",
        suffixes=("_trade", "_signal"),
    )
    rows: list[dict[str, object]] = []
    feature_lookup = features.set_index("timestamp")
    future_used = ["future_return"]
    for _, row in merged.iterrows():
        signal_ts = pd.Timestamp(row["timestamp"])
        horizon = str(row.get("horizon", "4h"))
        horizon_rows = HORIZON_TO_ROWS.get(horizon, 48)
        entry_ts = signal_ts
        exit_ts = signal_ts + pd.Timedelta(minutes=5 * horizon_rows)
        side = str(row["side"])
        entry_price = float(row["entry_price"])
        exit_price = float(row["exit_price"])
        position_size = float(row["position_size"])
        gross_return = float(row.get("future_return", (exit_price / entry_price) - 1))
        signed_return = gross_return if side == "LONG" else -gross_return
        net_return = signed_return - LAB_CONFIG.fee_pct - LAB_CONFIG.slippage_pct
        recomputed_pnl = position_size * net_return
        feature_values_present = signal_ts in feature_lookup.index
        rows.append(
            {
                "signal_timestamp": signal_ts.isoformat(),
                "entry_timestamp": entry_ts.isoformat(),
                "exit_timestamp": exit_ts.isoformat(),
                "symbol": row["symbol"],
                "side": side,
                "all_features_used": json.dumps(used_features),
                "feature_values_present_at_signal": feature_values_present,
                "future_columns_used": json.dumps(future_used),
                "trade_return_formula": "signed_return = future_return for LONG else -future_return; net_return = signed_return - fee_pct - slippage_pct; pnl_usd = position_size * net_return",
                "raw_pnl_calculation": f"{position_size:.10f} * ({signed_return:.10f} - {LAB_CONFIG.fee_pct:.6f} - {LAB_CONFIG.slippage_pct:.6f})",
                "reported_pnl_usd": float(row["pnl_usd"]),
                "recomputed_pnl_usd": recomputed_pnl,
                "pnl_matches": bool(np.isclose(float(row["pnl_usd"]), recomputed_pnl, atol=1e-8)),
                "execution_warning": "ENTRY_AT_SIGNAL_CLOSE_SAME_BAR" if entry_ts == signal_ts else "",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS_DIR / "integrity_trade_audit.csv", index=False)
    return out


def _evaluate_variant(name: str, transform: Callable[[pd.DataFrame, list[str]], pd.DataFrame], base_df: pd.DataFrame, used_features: list[str]) -> dict[str, object]:
    try:
        variant = transform(base_df.copy(), used_features)
        predictions, wf_metrics, _, _, _ = walk_forward(
            variant,
            horizon="4h",
            confidence_threshold=LAB_CONFIG.confidence_threshold,
            target_threshold=0.010,
            feature_set="full",
            balance_method="class_weight",
            calibration_method="sigmoid",
            folds=3,
        )
        trades = simulate(predictions)
        stats = compute_stats(trades)
        return {"test": name, **wf_metrics, **asdict(stats), "status": "ok"}
    except Exception as exc:
        return {"test": name, "status": f"failed: {exc}"}


def _shuffle_labels(df: pd.DataFrame, _: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    target_cols = [col for col in df.columns if col.startswith(("long_target_", "short_target_"))]
    for col in target_cols:
        df[col] = rng.permutation(df[col].to_numpy())
    return df


def _random_features(df: pd.DataFrame, used_features: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(456)
    for col in used_features:
        df[col] = rng.normal(0, 1, len(df))
    return df


def _delayed_features(df: pd.DataFrame, used_features: list[str]) -> pd.DataFrame:
    for col in used_features:
        df[col] = df[col].shift(1)
    return df.dropna().reset_index(drop=True)


def _remove_top_features(df: pd.DataFrame, used_features: list[str]) -> pd.DataFrame:
    importance_path = REPORTS_DIR / "feature_importance.csv"
    if importance_path.exists():
        top = pd.read_csv(importance_path).head(5)["feature"].tolist()
    else:
        top = used_features[:5]
    for col in top:
        if col in df.columns:
            df[col] = 0.0
    return df


def _sanity_tests(df: pd.DataFrame, used_features: list[str]) -> pd.DataFrame:
    tests = [
        ("shuffle_labels", _shuffle_labels),
        ("random_features", _random_features),
        ("delayed_features", _delayed_features),
        ("remove_top_features", _remove_top_features),
    ]
    results = [_evaluate_variant(name, fn, df, used_features) for name, fn in tests]
    out = pd.DataFrame(results)
    out.to_csv(REPORTS_DIR / "integrity_sanity_tests.csv", index=False)
    return out


def run_integrity_audit() -> str:
    ensure_dirs()
    features, predictions, trades = _load_inputs()
    used_features = feature_columns(features, feature_set="full")
    leakage = _leakage_checks(features, used_features)
    wf = _walk_forward_checks(features, predictions)
    trade_audit = _trade_audit(features, predictions, trades, used_features)
    sanity = _sanity_tests(features, used_features)

    shuffle_row = sanity[sanity["test"] == "shuffle_labels"]
    shuffle_invalid = False
    if not shuffle_row.empty and shuffle_row.iloc[0].get("status") == "ok":
        pf = float(shuffle_row.iloc[0].get("profit_factor", 0) or 0)
        avg = float(shuffle_row.iloc[0].get("avg_return_per_trade", 0) or 0)
        trades_count = int(shuffle_row.iloc[0].get("number_of_trades", 0) or 0)
        shuffle_invalid = bool((pf > 1.2 or avg > 0) and trades_count > 5)

    pnl_mismatches = int((~trade_audit["pnl_matches"]).sum()) if "pnl_matches" in trade_audit and not trade_audit.empty else 0
    same_bar_entries = int((trade_audit.get("execution_warning", pd.Series(dtype=str)) == "ENTRY_AT_SIGNAL_CLOSE_SAME_BAR").sum()) if not trade_audit.empty else 0
    future_feature_leak = bool(leakage["feature_leakage_detected"])
    walk_forward_contamination = not bool(wf["embargo_covers_horizon"])
    invalid_backtest = shuffle_invalid or pnl_mismatches > 0 or future_feature_leak or walk_forward_contamination or same_bar_entries > 0

    report = f"""# Integrity Audit

## Verdict

{"BACKTEST INVALID OR NOT TRUSTWORTHY" if invalid_backtest else "No critical integrity failure detected"}

This audit checks mechanics. It does not validate the trading edge.

## Leakage Checks

- Used features: {leakage["used_features_count"]}
- Feature leakage detected: {leakage["feature_leakage_detected"]}
- Feature leakage columns: {leakage["feature_leakage_columns"]}
- Future/target columns present only as labels: {len(leakage["future_columns_present_as_labels"])}

## Walk-Forward And Look-Ahead

- Horizon: {wf["horizon"]}
- Required embargo rows: {wf["required_embargo_rows"]}
- Configured embargo rows: {wf["configured_embargo_rows"]}
- Embargo covers horizon: {wf["embargo_covers_horizon"]}
- Timestamps monotonic: {wf["timestamps_monotonic"]}
- Duplicate timestamps: {wf["duplicate_timestamps"]}

## Execution And PnL

- Trades audited: {len(trade_audit)}
- Same-bar entry fills: {same_bar_entries}
- PnL mismatches: {pnl_mismatches}
- Current paper trader entry logic: entry at signal candle close.
- Integrity note: same-bar close fills are optimistic unless the signal is known before that close is tradable. Prefer next-candle open/close execution for future runs.

Detailed per-trade audit: `data/reports/integrity_trade_audit.csv`

## Sanity Tests

{sanity.to_string(index=False)}

## Shuffle Labels Test

If shuffled labels remain profitable, the backtest is invalid. Result: {"FAILED - shuffled labels still produced apparent profitability" if shuffle_invalid else "passed or inconclusive"}

## Raw Trade Formula

`signed_return = future_return for LONG else -future_return`

`net_return = signed_return - fee_pct - slippage_pct`

`pnl_usd = position_size * net_return`

## Final Notes

- Future columns used for trade outcome are allowed only after signal generation for evaluation, not as model features.
- `integrity_trade_audit.csv` records `signal_timestamp`, `entry_timestamp`, `exit_timestamp`, feature list, future outcome columns, formula, and raw PnL calculation for every trade.
"""
    (REPORTS_DIR / "integrity_audit.md").write_text(report, encoding="utf-8")
    LOGGER.info("Saved integrity audit invalid_backtest=%s", invalid_backtest)
    return report


if __name__ == "__main__":
    from scripts.common import setup_logging

    setup_logging()
    print(run_integrity_audit())

