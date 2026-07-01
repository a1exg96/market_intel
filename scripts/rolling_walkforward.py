from __future__ import annotations

import logging

import pandas as pd

from scripts.baseline_ml import walk_forward
from scripts.common import LAB_CONFIG, PROCESSED_DIR, REPORTS_DIR, ensure_dirs, read_parquet
from scripts.paper_trader import compute_stats, simulate

LOGGER = logging.getLogger(__name__)


def run_rolling_walkforward(train_days: int = 90, test_days: int = 30) -> pd.DataFrame:
    ensure_dirs()
    df = read_parquet(PROCESSED_DIR / f"{LAB_CONFIG.raw_symbol}_{LAB_CONFIG.timeframe}_features.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    start = df["timestamp"].min()
    end = df["timestamp"].max()
    rows: list[dict] = []
    fold = 0
    cursor = start
    while cursor + pd.Timedelta(days=train_days + test_days) <= end:
        train_end = cursor + pd.Timedelta(days=train_days)
        test_end = train_end + pd.Timedelta(days=test_days)
        window = df[(df["timestamp"] >= cursor) & (df["timestamp"] < test_end)].reset_index(drop=True)
        if len(window) < 150:
            cursor += pd.Timedelta(days=test_days)
            continue
        try:
            predictions, _, _, _, _ = walk_forward(window, folds=3)
            trades = simulate(predictions)
            stats = compute_stats(trades)
            rows.append({"fold": fold, "train_start": cursor, "train_end": train_end, "test_end": test_end, **stats.__dict__})
            fold += 1
        except Exception as exc:
            rows.append({"fold": fold, "train_start": cursor, "train_end": train_end, "test_end": test_end, "error": str(exc)})
            fold += 1
        cursor += pd.Timedelta(days=test_days)
    if not rows:
        try:
            predictions, _, _, _, _ = walk_forward(df, folds=4)
            trades = simulate(predictions)
            stats = compute_stats(trades)
            rows.append({"fold": 0, "train_start": start, "train_end": end, "test_end": end, "note": "insufficient history for 90d/30d; used available sample", **stats.__dict__})
        except Exception as exc:
            rows.append({"fold": 0, "error": str(exc), "note": "insufficient history"})
    result = pd.DataFrame(rows)
    if "profit_factor" in result:
        result["stability_score"] = result[["profit_factor", "winrate", "avg_return_per_trade"]].replace([float("inf")], 10).mean(axis=1) - result["max_drawdown"].abs()
    result.to_csv(REPORTS_DIR / "rolling_walkforward.csv", index=False)
    report = "# Rolling Walk-Forward Report\n\n" + result.to_string(index=False) + "\n"
    (REPORTS_DIR / "rolling_walkforward.md").write_text(report, encoding="utf-8")
    LOGGER.info("Saved rolling walk-forward rows=%s", len(result))
    return result


if __name__ == "__main__":
    from scripts.common import setup_logging

    setup_logging()
    run_rolling_walkforward()

