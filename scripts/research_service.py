from __future__ import annotations

import logging
import os
import signal
import time

import pandas as pd

from scripts.baseline_ml import train
from scripts.cache import publish_event
from scripts.collector import collect
from scripts.common import LAB_CONFIG, RAW_DIR, REPORTS_DIR, market_symbols, setup_logging
from scripts.db import init_db, insert_daily_report, insert_signals, log_event, prune_runtime_tables
from scripts.feature_engineering import build_features
from scripts.forward_paper_engine import run_forward_paper_engine
from scripts.regime_engine import build_regime_labels
from scripts.stats_report import generate_report
from scripts.target_audit import run_target_audit

LOGGER = logging.getLogger(__name__)
RUNNING = True


def _stop(_: int, __: object) -> None:
    global RUNNING
    RUNNING = False


def _safe_read(path: str) -> pd.DataFrame:
    full = REPORTS_DIR / path
    if not full.exists() or full.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(full)


def _ensure_seed_market_data() -> None:
    for symbol in market_symbols():
        candles_path = RAW_DIR / f"{symbol}_{LAB_CONFIG.timeframe}_candles.parquet"
        futures_path = RAW_DIR / f"{symbol}_{LAB_CONFIG.timeframe}_futures_context.parquet"
        if candles_path.exists() and futures_path.exists():
            continue
        LOGGER.info("Seed market data missing; collecting %s before research cycle.", symbol)
        collect(symbol=symbol, interval=LAB_CONFIG.timeframe, limit=1000)


def run_research_service() -> None:
    setup_logging()
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    interval = int(os.getenv("RESEARCH_INTERVAL_SECONDS", "60"))
    while RUNNING:
        try:
            init_db()
            _ensure_seed_market_data()
            for symbol in market_symbols():
                build_features(symbol=symbol)
            run_target_audit()
            build_regime_labels()
            predictions = train()
            insert_signals(predictions)
            trades_count = 0
            if os.getenv("ENABLE_HISTORICAL_PAPER_TRADES", "false").lower() == "true":
                from scripts.db import insert_paper_trades
                from scripts.paper_trader import run_paper

                trades = run_paper()
                trades_count = len(trades)
                insert_paper_trades(trades)
            run_forward_paper_engine()
            report = generate_report()
            insert_daily_report(report)
            prune_runtime_tables()
            log_event("research", "INFO", "research cycle completed", {"report_chars": len(report)})
            publish_event(
                "market_intel.research",
                {"predictions": len(predictions), "trades": trades_count, "report_chars": len(report)},
            )
        except Exception as exc:
            LOGGER.exception("research loop failed: %s", exc)
            log_event("research", "ERROR", str(exc))
        time.sleep(interval)
    log_event("research", "INFO", "research stopped")


if __name__ == "__main__":
    run_research_service()
