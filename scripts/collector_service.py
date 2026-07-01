from __future__ import annotations

import logging
import os
import signal
import time

from scripts.collector import collect
from scripts.cache import publish_event
from scripts.common import LAB_CONFIG, market_symbols, setup_logging
from scripts.db import init_db, log_event, prune_runtime_tables, upsert_candles, upsert_futures_context

LOGGER = logging.getLogger(__name__)
RUNNING = True


def _stop(_: int, __: object) -> None:
    global RUNNING
    RUNNING = False


def run_collector_service() -> None:
    setup_logging()
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    interval = int(os.getenv("COLLECTOR_INTERVAL_SECONDS", "30"))
    symbols = market_symbols()
    while RUNNING:
        try:
            init_db()
            for symbol in symbols:
                candles, futures = collect(symbol=symbol, interval=LAB_CONFIG.timeframe, limit=1000)
                candle_rows = upsert_candles(candles, timeframe=LAB_CONFIG.timeframe)
                context_rows = upsert_futures_context(futures, symbol=symbol)
                message = f"collector stored symbol={symbol} candles={candle_rows} context={context_rows}"
                LOGGER.info(message)
                log_event("collector", "INFO", message)
                publish_event(
                    "market_intel.collector",
                    {"symbol": symbol, "candles": candle_rows, "context": context_rows},
                )
            prune_runtime_tables()
        except Exception as exc:
            LOGGER.exception("collector loop failed: %s", exc)
            log_event("collector", "ERROR", str(exc))
        time.sleep(interval)
    log_event("collector", "INFO", "collector stopped")


if __name__ == "__main__":
    run_collector_service()
