from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from scripts.common import LAB_CONFIG, RAW_DIR, ensure_dirs, save_parquet
from scripts.futures_context import build_context
from scripts.privacy_audit import checked_request

LOGGER = logging.getLogger(__name__)
SPOT_BASE = "https://api.binance.com"


def _fetch_binance_klines(symbol: str, interval: str, limit: int) -> list[Any]:
    response = checked_request(
        "GET",
        f"{SPOT_BASE}/api/v3/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def synthetic_candles(symbol: str, periods: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    timestamps = pd.date_range(end=datetime.now(timezone.utc), periods=periods, freq="5min")
    wave = np.array([math.sin(i / 35) for i in range(periods)]) * 0.012
    drift = np.linspace(0.0, 0.05, periods)
    noise = rng.normal(0, 0.002, periods).cumsum() / 8
    close = 65_000 * (1 + drift + wave + noise)
    open_ = np.roll(close, 1)
    open_[0] = close[0] * 0.999
    high = np.maximum(open_, close) * (1 + rng.uniform(0.0005, 0.003, periods))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.0005, 0.003, periods))
    volume = rng.lognormal(mean=3.3, sigma=0.4, size=periods)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": symbol,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "source": "synthetic",
        }
    )


def collect(symbol: str = LAB_CONFIG.raw_symbol, interval: str = LAB_CONFIG.timeframe, limit: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    try:
        payload = _fetch_binance_klines(symbol, interval, min(limit, 1000))
        cols = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ]
        candles = pd.DataFrame(payload, columns=cols)
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            candles[col] = pd.to_numeric(candles[col], errors="coerce")
        candles["timestamp"] = pd.to_datetime(candles["open_time"], unit="ms", utc=True)
        candles["symbol"] = symbol
        candles["source"] = "binance"
        candles = candles[["timestamp", "symbol", "open", "high", "low", "close", "volume", "quote_volume", "source"]]
    except Exception as exc:
        LOGGER.warning("Market data unavailable, using synthetic demo candles: %s", exc)
        candles = synthetic_candles(symbol=symbol, periods=limit)

    futures = build_context(candles["timestamp"], symbol=symbol)
    save_parquet(candles, RAW_DIR / f"{symbol}_{interval}_candles.parquet")
    save_parquet(futures, RAW_DIR / f"{symbol}_{interval}_futures_context.parquet")
    LOGGER.info("Saved local market data rows=%s futures_rows=%s", len(candles), len(futures))
    return candles, futures


if __name__ == "__main__":
    from scripts.common import setup_logging

    setup_logging()
    collect()

