from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from scripts.common import LAB_CONFIG, PROCESSED_DIR, RAW_DIR, ensure_dirs, read_parquet, save_parquet

LOGGER = logging.getLogger(__name__)
STEPS_PER_HOUR = 12
TARGET_COLUMNS = {
    "future_return_1h",
    "future_return_4h",
    "future_return_24h",
    "future_max_up_1h",
    "future_max_up_4h",
    "future_max_up_24h",
    "future_max_down_1h",
    "future_max_down_4h",
    "future_max_down_24h",
    "long_target_1h_005",
    "long_target_1h_010",
    "long_target_1h_015",
    "long_target_1h_020",
    "long_target_4h_005",
    "long_target_4h_010",
    "long_target_4h_015",
    "long_target_4h_020",
    "long_target_24h_005",
    "long_target_24h_010",
    "long_target_24h_015",
    "long_target_24h_020",
    "short_target_1h_005",
    "short_target_1h_010",
    "short_target_1h_015",
    "short_target_1h_020",
    "short_target_4h_005",
    "short_target_4h_010",
    "short_target_4h_015",
    "short_target_4h_020",
    "short_target_24h_005",
    "short_target_24h_010",
    "short_target_24h_015",
    "short_target_24h_020",
}


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(5, window // 4)).mean()
    std = series.rolling(window, min_periods=max(5, window // 4)).std()
    return (series - mean) / std.replace(0, np.nan)


def _regime(row: pd.Series) -> str:
    if row["return_24h"] > 0.02 and row["vol_24h"] > 0:
        return "bull_momentum"
    if row["return_24h"] < -0.02:
        return "bear_momentum"
    if row["range_pct"] > 0.01:
        return "high_volatility"
    return "neutral"


def _future_max_up(high: pd.Series, close: pd.Series, horizon: int) -> pd.Series:
    future_high = high.shift(-1).rolling(horizon, min_periods=horizon).max().shift(-(horizon - 1))
    return future_high / close - 1


def _future_max_down(low: pd.Series, close: pd.Series, horizon: int) -> pd.Series:
    future_low = low.shift(-1).rolling(horizon, min_periods=horizon).min().shift(-(horizon - 1))
    return future_low / close - 1


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def build_features(symbol: str = LAB_CONFIG.raw_symbol, interval: str = LAB_CONFIG.timeframe) -> pd.DataFrame:
    ensure_dirs()
    candles = read_parquet(RAW_DIR / f"{symbol}_{interval}_candles.parquet").sort_values("timestamp")
    futures = read_parquet(RAW_DIR / f"{symbol}_{interval}_futures_context.parquet").sort_values("timestamp")
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)
    futures["timestamp"] = pd.to_datetime(futures["timestamp"], utc=True)
    df = pd.merge_asof(candles, futures, on="timestamp", direction="backward")

    df["return_1h"] = df["close"].pct_change(STEPS_PER_HOUR)
    df["return_4h"] = df["close"].pct_change(STEPS_PER_HOUR * 4)
    df["return_24h"] = df["close"].pct_change(STEPS_PER_HOUR * 24)
    df["future_return_1h"] = df["close"].shift(-STEPS_PER_HOUR) / df["close"] - 1
    df["future_return_4h"] = df["close"].shift(-STEPS_PER_HOUR * 4) / df["close"] - 1
    df["future_return_24h"] = df["close"].shift(-STEPS_PER_HOUR * 24) / df["close"] - 1
    for horizon_name, horizon_rows in {"1h": 12, "4h": 48, "24h": 288}.items():
        df[f"future_max_up_{horizon_name}"] = _future_max_up(df["high"], df["close"], horizon_rows)
        df[f"future_max_down_{horizon_name}"] = _future_max_down(df["low"], df["close"], horizon_rows)
        for suffix, threshold in {"005": 0.005, "010": 0.010, "015": 0.015, "020": 0.020}.items():
            df[f"long_target_{horizon_name}_{suffix}"] = (df[f"future_max_up_{horizon_name}"] > threshold).astype(int)
            df[f"short_target_{horizon_name}_{suffix}"] = (df[f"future_max_down_{horizon_name}"] < -threshold).astype(int)
    df["volume_zscore"] = _zscore(df["volume"], STEPS_PER_HOUR * 24)
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["body_pct"] = (df["close"] - df["open"]).abs() / df["open"]
    df["vol_24h"] = df["volume"].rolling(STEPS_PER_HOUR * 24, min_periods=STEPS_PER_HOUR).sum()
    df["atr"] = _atr(df) / df["close"]
    df["realized_volatility"] = df["close"].pct_change().rolling(STEPS_PER_HOUR * 24, min_periods=STEPS_PER_HOUR).std()
    btc_return = df["close"].pct_change()
    df["rolling_beta_BTC"] = (
        btc_return.rolling(STEPS_PER_HOUR * 24, min_periods=STEPS_PER_HOUR).cov(btc_return)
        / btc_return.rolling(STEPS_PER_HOUR * 24, min_periods=STEPS_PER_HOUR).var().replace(0, np.nan)
    ).fillna(1.0)
    df["momentum_12h"] = df["close"].pct_change(STEPS_PER_HOUR * 12)
    df["momentum_24h"] = df["close"].pct_change(STEPS_PER_HOUR * 24)
    df["funding_rate"] = df["funding_rate"].ffill().fillna(0.0)
    df["funding_zscore"] = _zscore(df["funding_rate"], STEPS_PER_HOUR * 24).fillna(0.0)
    df["open_interest"] = df["open_interest"].ffill().bfill().fillna(0.0)
    df["oi_change_1h"] = df["open_interest"].pct_change(STEPS_PER_HOUR).fillna(0.0)
    df["oi_change_4h"] = df["open_interest"].pct_change(STEPS_PER_HOUR * 4).fillna(0.0)
    df["oi_change_24h"] = df["open_interest"].pct_change(STEPS_PER_HOUR * 24).fillna(0.0)
    df["funding_acceleration"] = df["funding_rate"].diff(STEPS_PER_HOUR).fillna(0.0)
    df["oi_acceleration"] = df["oi_change_1h"].diff(STEPS_PER_HOUR).fillna(0.0)
    df["long_short_ratio"] = df["long_short_ratio"].ffill().fillna(1.0)
    df["liquidation_imbalance"] = df["liquidation_imbalance"].ffill().fillna(0.0)
    signed_volume = np.sign(df["close"] - df["open"]).replace(0, 1) * df["volume"]
    df["cumulative_delta"] = signed_volume.rolling(STEPS_PER_HOUR * 4, min_periods=STEPS_PER_HOUR).sum() / df["vol_24h"].replace(0, np.nan)
    df["orderbook_imbalance"] = ((df["close"] - df["open"]) / (df["high"] - df["low"]).replace(0, np.nan)).clip(-1, 1)
    df["sentiment_score"] = 0.0
    df["event_type"] = "none"
    df["event_type_none"] = True
    df["news_count_1h"] = 0
    df["news_count_24h"] = 0
    df["similar_event_success_rate"] = 0.5
    df["similar_event_avg_return"] = 0.0
    df["regime_label"] = df.apply(_regime, axis=1)
    df = pd.get_dummies(df, columns=["regime_label"], prefix="regime")
    feature_like = [
        "return_1h",
        "return_4h",
        "return_24h",
        "volume_zscore",
        "range_pct",
        "body_pct",
        "vol_24h",
        "atr",
        "realized_volatility",
        "rolling_beta_BTC",
        "momentum_12h",
        "momentum_24h",
        "funding_rate",
        "funding_zscore",
        "open_interest",
        "oi_change_1h",
        "oi_change_4h",
        "oi_change_24h",
        "long_short_ratio",
        "liquidation_imbalance",
        "funding_acceleration",
        "oi_acceleration",
        "orderbook_imbalance",
        "cumulative_delta",
        "sentiment_score",
        "news_count_1h",
        "news_count_24h",
        "similar_event_success_rate",
        "similar_event_avg_return",
        "future_return_1h",
        "future_return_4h",
        "future_return_24h",
        "future_max_up_1h",
        "future_max_up_4h",
        "future_max_up_24h",
        "future_max_down_1h",
        "future_max_down_4h",
        "future_max_down_24h",
    ]
    df = df.replace([np.inf, -np.inf], np.nan)
    live_safe_required = [
        col
        for col in feature_like
        if col in df.columns and not col.startswith("future_") and not col.startswith(("long_target_", "short_target_"))
    ]
    df = df.dropna(subset=live_safe_required).reset_index(drop=True)
    save_parquet(df, PROCESSED_DIR / f"{symbol}_{interval}_features.parquet")
    LOGGER.info("Saved features rows=%s", len(df))
    return df


def feature_columns(df: pd.DataFrame, feature_set: str = "full") -> list[str]:
    blocked_tokens = ("future", "max_up", "max_down", "direction")
    non_features = {"timestamp", "symbol", "source", "context_source", "generated_at", "open", "high", "low", "close", "event_type"}
    non_features.update({col for col in df.columns if col.startswith(("trade_target_", "long_target_", "short_target_"))})
    cols = [c for c in df.columns if c not in non_features and not any(token in c for token in blocked_tokens)]
    if feature_set == "basic":
        allowed = {"return_1h", "return_4h", "return_24h", "volume_zscore", "range_pct", "body_pct", "vol_24h", "atr", "realized_volatility", "momentum_12h", "momentum_24h"}
        return [c for c in cols if c in allowed]
    if feature_set == "futures":
        allowed = {
            "funding_rate",
            "funding_zscore",
            "open_interest",
            "oi_change_1h",
            "oi_change_4h",
            "oi_change_24h",
            "long_short_ratio",
            "liquidation_imbalance",
            "funding_acceleration",
            "oi_acceleration",
        }
        return [c for c in cols if c in allowed]
    return cols


if __name__ == "__main__":
    from scripts.common import setup_logging

    setup_logging()
    build_features()
