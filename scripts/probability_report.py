from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from scripts.common import LAB_CONFIG, PROCESSED_DIR, REPORTS_DIR, ensure_dirs, profit_factor, read_parquet

LOGGER = logging.getLogger(__name__)
BUCKETS = [0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00]


def _empty_report(reason: str) -> pd.DataFrame:
    ensure_dirs()
    df = pd.DataFrame(
        columns=[
            "bucket",
            "predictions_count",
            "predicted_up",
            "predicted_down",
            "predicted_no_trade",
            "actual_winrate",
            "avg_future_return",
            "profit_factor_if_traded",
            "avg_trade_return_after_costs",
        ]
    )
    df.to_csv(REPORTS_DIR / "probability_buckets.csv", index=False)
    (REPORTS_DIR / "probability_report.md").write_text(f"# Probability Report\n\nInsufficient data: {reason}\n", encoding="utf-8")
    return df


def build_probability_report() -> pd.DataFrame:
    ensure_dirs()
    predictions_path = PROCESSED_DIR / "predictions.parquet"
    if not predictions_path.exists():
        return _empty_report("predictions.parquet not found. Run train first.")
    predictions = read_parquet(predictions_path)
    if predictions.empty or "confidence" not in predictions:
        return _empty_report("predictions are empty or missing confidence.")

    df = predictions.copy()
    df["predicted_class"] = df.get("predicted_label", df.get("predicted_direction", "")).astype(str)
    df["bucket"] = pd.cut(df["confidence"].astype(float), BUCKETS, include_lowest=True, right=False)
    rows: list[dict[str, object]] = []
    for bucket, part in df.groupby("bucket", observed=False):
        if part.empty:
            rows.append(
                {
                    "bucket": str(bucket),
                    "predictions_count": 0,
                    "predicted_up": 0,
                    "predicted_down": 0,
                    "predicted_no_trade": 0,
                    "actual_winrate": np.nan,
                    "avg_future_return": np.nan,
                    "profit_factor_if_traded": np.nan,
                    "avg_trade_return_after_costs": np.nan,
                }
            )
            continue
        predicted_up = part["predicted_class"].isin(["LONG_SETUP", "big_move_up", "up"])
        predicted_down = part["predicted_class"].isin(["SHORT_SETUP", "big_move_down", "down"])
        predicted_no_trade = part["predicted_class"].isin(["NO_TRADE", "no_trade", "flat", ""])
        signed_return = pd.Series(np.nan, index=part.index, dtype="float64")
        if "future_return" in part:
            signed_return.loc[predicted_up] = part.loc[predicted_up, "future_return"].astype(float)
            signed_return.loc[predicted_down] = -part.loc[predicted_down, "future_return"].astype(float)
        trade_returns = (signed_return - LAB_CONFIG.fee_pct - LAB_CONFIG.slippage_pct).dropna()
        rows.append(
            {
                "bucket": str(bucket),
                "predictions_count": int(len(part)),
                "predicted_up": int(predicted_up.sum()),
                "predicted_down": int(predicted_down.sum()),
                "predicted_no_trade": int(predicted_no_trade.sum()),
                "actual_winrate": float((trade_returns > 0).mean()) if not trade_returns.empty else np.nan,
                "avg_future_return": float(part["future_return"].mean()) if "future_return" in part else np.nan,
                "profit_factor_if_traded": profit_factor(trade_returns) if not trade_returns.empty else np.nan,
                "avg_trade_return_after_costs": float(trade_returns.mean()) if not trade_returns.empty else np.nan,
            }
        )

    output = pd.DataFrame(rows)
    output.to_csv(REPORTS_DIR / "probability_buckets.csv", index=False)
    non_empty = output[output["predictions_count"] > 0].copy()
    important = non_empty.sort_values(["predictions_count", "bucket"], ascending=[False, False]).head(5)
    report = f"""# Probability Report

This report checks whether high confidence correlates with profitability. It does not change thresholds automatically.

## Non-Empty Buckets

{non_empty.to_string(index=False) if not non_empty.empty else "No predictions available."}

## Most Important Buckets

{important.to_string(index=False) if not important.empty else "No populated confidence buckets."}
"""
    (REPORTS_DIR / "probability_report.md").write_text(report, encoding="utf-8")
    LOGGER.info("Saved probability report buckets=%s", len(output))
    return output


if __name__ == "__main__":
    from scripts.common import setup_logging

    setup_logging()
    build_probability_report()
