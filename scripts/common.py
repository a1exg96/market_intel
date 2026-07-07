from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"
KB_DIR = DATA_DIR / "knowledge_base"
MODELS_DIR = ROOT / "models"
DEFAULT_MARKET_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


@dataclass(frozen=True)
class LabConfig:
    symbol: str = "BTC/USDT"
    raw_symbol: str = "BTCUSDT"
    timeframe: str = "5m"
    initial_balance: float = 1000.0
    risk_per_trade: float = 0.005
    confidence_threshold: float = 0.70
    fee_pct: float = 0.0004
    slippage_pct: float = 0.0003
    flat_band: float = 0.0015
    embargo_rows: int = 48


LAB_CONFIG = LabConfig()


def market_symbols() -> list[str]:
    raw = os.getenv("MARKET_INTEL_SYMBOLS")
    if not raw:
        return list(DEFAULT_MARKET_SYMBOLS)
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def ensure_dirs() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, REPORTS_DIR, KB_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def save_parquet(df: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    df.to_parquet(tmp_path, index=False)
    last_error: Exception | None = None
    for _ in range(20):
        try:
            tmp_path.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.05)
    raise RuntimeError(f"Could not replace {path}") from last_error


def read_parquet(path: Path) -> Any:
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_parquet(path)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def assert_dry_run_config(path: Path) -> None:
    config = load_json(path)
    if config.get("dry_run") is not True:
        raise RuntimeError(f"Refusing live trading config: {path} must set dry_run=true")
    exchange = config.get("exchange", {})
    if exchange.get("key") or exchange.get("secret"):
        raise RuntimeError(f"Refusing config with API keys in file: {path}")


def profit_factor(returns: Iterable[float]) -> float:
    import pandas as pd

    series = pd.Series(list(returns), dtype="float64").dropna()
    wins = series[series > 0].sum()
    losses = series[series < 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / abs(losses))


def max_drawdown(equity: Iterable[float]) -> float:
    import pandas as pd

    series = pd.Series(list(equity), dtype="float64").dropna()
    if series.empty:
        return 0.0
    peak = series.cummax()
    return float(((series - peak) / peak).min())


def sharpe_like(returns: Iterable[float]) -> float:
    import numpy as np
    import pandas as pd

    series = pd.Series(list(returns), dtype="float64").dropna()
    std = series.std(ddof=0)
    if series.empty or std == 0:
        return 0.0
    return float(np.sqrt(len(series)) * series.mean() / std)


def sortino_like(returns: Iterable[float]) -> float:
    import numpy as np
    import pandas as pd

    series = pd.Series(list(returns), dtype="float64").dropna()
    downside = series[series < 0].std(ddof=0)
    if series.empty or pd.isna(downside) or downside == 0:
        return 0.0
    return float(np.sqrt(len(series)) * series.mean() / downside)
