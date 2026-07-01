from __future__ import annotations

import logging

import pandas as pd

from scripts.common import LAB_CONFIG, PROCESSED_DIR, REPORTS_DIR, ensure_dirs, market_symbols, read_parquet

LOGGER = logging.getLogger(__name__)
MOVE_THRESHOLDS = [0.005, 0.010, 0.015, 0.020, 0.030]
TARGET_THRESHOLDS = [0.005, 0.010, 0.015, 0.020]
HORIZONS = ["1h", "4h", "24h"]


def _suffix(threshold: float) -> str:
    return f"{int(threshold * 1000):03d}"


def run_target_audit(symbol: str | None = None, interval: str = LAB_CONFIG.timeframe) -> pd.DataFrame:
    ensure_dirs()
    symbols = [symbol] if symbol else market_symbols()
    feature_paths = [PROCESSED_DIR / f"{item}_{interval}_features.parquet" for item in symbols]
    if any(not path.exists() for path in feature_paths):
        report = "# Target Audit\n\nInsufficient data: features file not found. Run `features` first.\n"
        (REPORTS_DIR / "target_audit.md").write_text(report, encoding="utf-8")
        empty = pd.DataFrame()
        empty.to_csv(REPORTS_DIR / "target_distribution.csv", index=False)
        return empty

    df = pd.concat([read_parquet(path) for path in feature_paths], ignore_index=True)
    rows: list[dict[str, float | int | str | bool]] = []
    for horizon in HORIZONS:
        up_col = f"future_max_up_{horizon}"
        down_col = f"future_max_down_{horizon}"
        if up_col not in df.columns or down_col not in df.columns:
            continue
        for threshold in TARGET_THRESHOLDS:
            suffix = _suffix(threshold)
            long_col = f"long_target_{horizon}_{suffix}"
            short_col = f"short_target_{horizon}_{suffix}"
            long_count = int(df[long_col].sum()) if long_col in df.columns else int((df[up_col] > threshold).sum())
            short_count = int(df[short_col].sum()) if short_col in df.columns else int((df[down_col] < -threshold).sum())
            none_count = int(len(df) - long_count - short_count)
            minority = max(1, min(long_count, short_count))
            majority = max(long_count, short_count, none_count)
            rows.append(
                {
                    "horizon": horizon,
                    "target_threshold": threshold,
                    "examples": int(len(df)),
                    "long_positive": long_count,
                    "short_positive": short_count,
                    "neither_setup": max(0, none_count),
                    "long_share": long_count / max(len(df), 1),
                    "short_share": short_count / max(len(df), 1),
                    "imbalance_ratio": majority / minority,
                    "severe_imbalance": majority / minority > 10,
                }
            )

    for horizon in HORIZONS:
        up_col = f"future_max_up_{horizon}"
        down_col = f"future_max_down_{horizon}"
        if up_col not in df.columns or down_col not in df.columns:
            continue
        for threshold in MOVE_THRESHOLDS:
            rows.append(
                {
                    "horizon": horizon,
                    "target_threshold": threshold,
                    "examples": int(len(df)),
                    "long_positive": int((df[up_col] > threshold).sum()),
                    "short_positive": int((df[down_col] < -threshold).sum()),
                    "neither_setup": int(((df[up_col] <= threshold) & (df[down_col] >= -threshold)).sum()),
                    "long_share": float((df[up_col] > threshold).mean()),
                    "short_share": float((df[down_col] < -threshold).mean()),
                    "imbalance_ratio": float("nan"),
                    "severe_imbalance": False,
                }
            )

    output = pd.DataFrame(rows).drop_duplicates(subset=["horizon", "target_threshold", "examples", "long_positive", "short_positive"])
    output.to_csv(REPORTS_DIR / "target_distribution.csv", index=False)
    summary = output[output["target_threshold"].isin(TARGET_THRESHOLDS)].copy()
    report = f"""# Target Audit

The model no longer uses `no_trade` as a training class. It audits separate binary LONG/SHORT setup labels.

## Target Distribution

{summary.to_string(index=False) if not summary.empty else "No target rows available."}

## Diagnosis

- If `long_positive` or `short_positive` is very small, the setup task is sparse and needs more data or a less strict target threshold.
- If `imbalance_ratio` is above 10, balancing methods should be tested; do not solve it by blindly lowering confidence thresholds.
- These counts are labels only. Future columns remain blocked from feature training.
"""
    (REPORTS_DIR / "target_audit.md").write_text(report, encoding="utf-8")
    LOGGER.info("Saved target audit rows=%s", len(output))
    return output


if __name__ == "__main__":
    from scripts.common import setup_logging

    setup_logging()
    run_target_audit()
