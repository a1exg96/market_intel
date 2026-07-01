from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from scripts.common import LAB_CONFIG
from scripts.privacy_audit import checked_request

LOGGER = logging.getLogger(__name__)
FAPI_BASE = "https://fapi.binance.com"


def _json_get(path: str, params: dict[str, Any]) -> Any:
    response = checked_request("GET", f"{FAPI_BASE}{path}", params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def fetch_funding_rate(symbol: str = LAB_CONFIG.raw_symbol, limit: int = 1000) -> pd.DataFrame:
    payload = _json_get("/fapi/v1/fundingRate", {"symbol": symbol, "limit": limit})
    df = pd.DataFrame(payload)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    return df[["timestamp", "funding_rate"]].sort_values("timestamp")


def fetch_open_interest(symbol: str = LAB_CONFIG.raw_symbol, period: str = "5m", limit: int = 500) -> pd.DataFrame:
    payload = _json_get("/futures/data/openInterestHist", {"symbol": symbol, "period": period, "limit": limit})
    df = pd.DataFrame(payload)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "open_interest"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["open_interest"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
    return df[["timestamp", "open_interest"]].sort_values("timestamp")


def synthetic_context(timestamps: pd.Series) -> pd.DataFrame:
    ts = pd.to_datetime(timestamps, utc=True)
    idx = range(len(ts))
    return pd.DataFrame(
        {
            "timestamp": ts,
            "funding_rate": [0.0001 + 0.00004 * math.sin(i / 80) for i in idx],
            "open_interest": [3_000_000_000 + 25_000_000 * math.sin(i / 35) for i in idx],
            "long_short_ratio": [1.0 + 0.08 * math.sin(i / 50) for i in idx],
            "liquidation_imbalance": [0.02 * math.sin(i / 20) for i in idx],
            "context_source": "synthetic",
            "generated_at": datetime.now(timezone.utc),
        }
    )


def build_context(timestamps: pd.Series, symbol: str = LAB_CONFIG.raw_symbol) -> pd.DataFrame:
    try:
        funding = fetch_funding_rate(symbol)
        oi = fetch_open_interest(symbol)
        if funding.empty or oi.empty:
            raise RuntimeError("empty futures context")
        base = pd.DataFrame({"timestamp": pd.to_datetime(timestamps, utc=True)}).sort_values("timestamp")
        out = pd.merge_asof(base, funding.sort_values("timestamp"), on="timestamp", direction="backward")
        out = pd.merge_asof(out, oi.sort_values("timestamp"), on="timestamp", direction="backward")
        out["long_short_ratio"] = 1.0
        out["liquidation_imbalance"] = 0.0
        out["context_source"] = "binance_futures"
        return out
    except Exception as exc:
        LOGGER.warning("Futures context unavailable, using synthetic demo context: %s", exc)
        return synthetic_context(timestamps)

