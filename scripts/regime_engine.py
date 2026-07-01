from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from scripts.common import LAB_CONFIG, PROCESSED_DIR, ensure_dirs, market_symbols, read_parquet, save_parquet

LOGGER = logging.getLogger(__name__)

REGIME_ORDER = ["TREND_UP", "TREND_DOWN", "RANGE", "PANIC", "EUPHORIA"]


def classify_regimes(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    required = [
        "timestamp",
        "symbol",
        "return_24h",
        "realized_volatility",
        "funding_rate",
        "funding_zscore",
        "open_interest",
        "oi_change_24h",
        "atr",
        "volume_zscore",
        "liquidation_imbalance",
        "momentum_24h",
    ]
    missing = [col for col in required if col not in work.columns]
    if missing:
        raise ValueError(f"Cannot classify regimes; missing columns: {missing}")

    vol_q80 = work["realized_volatility"].quantile(0.80)
    atr_q80 = work["atr"].quantile(0.80)
    volume_q80 = work["volume_zscore"].quantile(0.80)

    panic = (
        (work["return_24h"] < -0.02)
        & ((work["realized_volatility"] > vol_q80) | (work["atr"] > atr_q80))
        & ((work["volume_zscore"] > volume_q80) | (work["liquidation_imbalance"] < -0.01))
    )
    euphoria = (
        (work["return_24h"] > 0.02)
        & ((work["funding_zscore"] > 1.0) | (work["funding_rate"] > work["funding_rate"].quantile(0.75)))
        & ((work["oi_change_24h"] > 0) | (work["volume_zscore"] > volume_q80))
    )
    trend_up = (work["momentum_24h"] > 0.01) & (work["return_24h"] > 0)
    trend_down = (work["momentum_24h"] < -0.01) & (work["return_24h"] < 0)

    work["regime"] = np.select(
        [panic, euphoria, trend_up, trend_down],
        ["PANIC", "EUPHORIA", "TREND_UP", "TREND_DOWN"],
        default="RANGE",
    )
    work["regime_confidence"] = np.select(
        [panic, euphoria, trend_up | trend_down],
        [0.80, 0.75, 0.65],
        default=0.55,
    )
    return work[["timestamp", "symbol", "regime", "regime_confidence"]]


def build_regime_labels(symbol: str | None = None, interval: str = LAB_CONFIG.timeframe) -> pd.DataFrame:
    ensure_dirs()
    symbols = [symbol] if symbol else market_symbols()
    features = pd.concat(
        [read_parquet(PROCESSED_DIR / f"{item}_{interval}_features.parquet") for item in symbols],
        ignore_index=True,
    ).sort_values(["timestamp", "symbol"])
    regimes = classify_regimes(features)
    save_parquet(regimes, PROCESSED_DIR / "regime_labels.parquet")
    LOGGER.info("Saved regime labels rows=%s distribution=%s", len(regimes), regimes["regime"].value_counts().to_dict())
    return regimes


if __name__ == "__main__":
    from scripts.common import setup_logging

    setup_logging()
    build_regime_labels()
