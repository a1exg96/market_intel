from __future__ import annotations

import logging

import pandas as pd

from scripts.common import LAB_CONFIG, PROCESSED_DIR, REPORTS_DIR, ensure_dirs, read_parquet

LOGGER = logging.getLogger(__name__)


def build_label_distribution_report(
    symbol: str = LAB_CONFIG.raw_symbol,
    interval: str = LAB_CONFIG.timeframe,
    horizon: str = "4h",
    target_threshold: float = 0.010,
) -> str:
    ensure_dirs()
    features_path = PROCESSED_DIR / f"{symbol}_{interval}_features.parquet"
    if not features_path.exists():
        report = "# Label Distribution\n\nInsufficient data: features file not found. Run features first.\n"
        (REPORTS_DIR / "label_distribution.md").write_text(report, encoding="utf-8")
        return report

    df = read_parquet(features_path)
    suffix = f"{int(target_threshold * 1000):03d}"
    long_col = f"long_target_{horizon}_{suffix}"
    short_col = f"short_target_{horizon}_{suffix}"
    if long_col not in df.columns or short_col not in df.columns:
        report = "# Label Distribution\n\nInsufficient data: no supported long/short target columns found.\n"
        (REPORTS_DIR / "label_distribution.md").write_text(report, encoding="utf-8")
        return report

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    counts = pd.Series(
        {
            "LONG_SETUP": int(df[long_col].sum()),
            "SHORT_SETUP": int(df[short_col].sum()),
            "NO_SETUP": int(((df[long_col] == 0) & (df[short_col] == 0)).sum()),
        }
    )
    shares = counts / max(len(df), 1)
    dominant_share = float(shares.max()) if not shares.empty else 1.0
    big_move_mask = (df[long_col] == 1) | (df[short_col] == 1)
    by_day = df.assign(is_big_move=big_move_mask).groupby(df["timestamp"].dt.date)["is_big_move"].sum()
    by_week = df.assign(is_big_move=big_move_mask).groupby(df["timestamp"].dt.isocalendar().week)["is_big_move"].sum()

    report = f"""# Label Distribution

Target columns: `{long_col}`, `{short_col}`

Target threshold: {target_threshold:.2%}

## Counts

{counts.to_string()}

## Shares

{shares.to_string()}

Strong class imbalance: {dominant_share > 0.70}

Big moves per day:

{by_day.to_string()}

Big moves per ISO week:

{by_week.to_string()}

Target strictness diagnosis: {"likely too strict for this sample" if dominant_share > 0.80 else "not obviously too strict"}

The current big-move target is useful as a safety filter, but if it produces almost all `no_trade`, the next honest experiment is to compare 1.0%, 1.5%, and 2.0% move thresholds on walk-forward data.
"""
    (REPORTS_DIR / "label_distribution.md").write_text(report, encoding="utf-8")
    LOGGER.info("Saved label distribution report targets=%s,%s", long_col, short_col)
    return report


if __name__ == "__main__":
    from scripts.common import setup_logging

    setup_logging()
    build_label_distribution_report()
