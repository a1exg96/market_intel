from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from scripts.common import LAB_CONFIG, REPORTS_DIR, ensure_dirs
from scripts.paper_trader import compute_stats
from scripts.research_engine import run_research

LOGGER = logging.getLogger(__name__)


def _append_log(text: str) -> None:
    path = REPORTS_DIR / "adaptation_log.md"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n\n")


def adapt() -> str:
    ensure_dirs()
    trades_path = REPORTS_DIR / "trades.csv"
    if not trades_path.exists():
        message = "No paper trades found. Run paper trading before adaptation."
        _append_log(f"## {datetime.now(timezone.utc).isoformat()}\n{message}")
        return message
    trades = pd.read_csv(trades_path, parse_dates=["timestamp"])
    if trades.empty:
        message = "No paper trades found. Risk remains unchanged."
        _append_log(f"## {datetime.now(timezone.utc).isoformat()}\n{message}")
        return message
    cutoff = trades["timestamp"].max() - pd.Timedelta(days=7)
    recent = trades[trades["timestamp"] >= cutoff]
    stats = compute_stats(recent if not recent.empty else trades)
    trigger = stats.total_return < 0 or stats.profit_factor < 1.0 or stats.max_drawdown < -0.10
    if not trigger:
        message = (
            f"Adaptive check OK. total_return={stats.total_return:.2%}, "
            f"profit_factor={stats.profit_factor:.2f}, max_drawdown={stats.max_drawdown:.2%}. "
            "Risk remains capped at 1%; live trading remains disabled."
        )
        _append_log(f"## {datetime.now(timezone.utc).isoformat()}\n{message}")
        return message

    research = run_research()
    if research.empty:
        suggestion = "Research produced no candidate. Keep current config and do not increase risk."
    else:
        best = research.iloc[0]
        safe_threshold = max(float(best["confidence_threshold"]), LAB_CONFIG.confidence_threshold + 0.05)
        suggestion = (
            f"Suggested paper-only config: horizon={best['horizon']}, "
            f"feature_set={best['feature_set']}, confidence_threshold={safe_threshold:.2f}."
        )
    message = (
        f"Adaptive trigger fired. total_return={stats.total_return:.2%}, "
        f"profit_factor={stats.profit_factor:.2f}, max_drawdown={stats.max_drawdown:.2%}. "
        "Do not increase risk. Do not enable live trading. " + suggestion
    )
    _append_log(f"## {datetime.now(timezone.utc).isoformat()}\n{message}")
    LOGGER.warning(message)
    return message


if __name__ == "__main__":
    from scripts.common import setup_logging

    setup_logging()
    print(adapt())

