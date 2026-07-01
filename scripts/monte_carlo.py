from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from scripts.common import LAB_CONFIG, REPORTS_DIR, ensure_dirs, max_drawdown, profit_factor
from scripts.execution_engine import ExecutionConfig, simulate_realistic_execution

LOGGER = logging.getLogger(__name__)


def _simulate_equity(base_trades: pd.DataFrame, rng: np.random.Generator) -> dict[str, float]:
    if base_trades.empty:
        return {"total_return": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0, "trades": 0}
    trades = base_trades.sample(frac=1.0, replace=False, random_state=int(rng.integers(0, 1_000_000))).copy()
    keep = rng.random(len(trades)) > rng.uniform(0.0, 0.20)
    trades = trades[keep]
    balance = LAB_CONFIG.initial_balance
    equity = [balance]
    returns = []
    for _, row in trades.iterrows():
        jitter = rng.normal(0, 0.0005)
        ret = float(row["pnl_pct"]) + jitter
        pnl = balance * ret
        balance += pnl
        returns.append(ret)
        equity.append(balance)
    return {
        "total_return": balance / LAB_CONFIG.initial_balance - 1,
        "profit_factor": profit_factor(returns),
        "max_drawdown": max_drawdown(equity),
        "trades": int(len(trades)),
    }


def run_monte_carlo(n: int = 5000) -> pd.DataFrame:
    ensure_dirs()
    rng = np.random.default_rng(2027)
    base = simulate_realistic_execution(ExecutionConfig(mode="next_open_random_slippage", fee_pct=0.0008, slippage_pct=0.0010, missed_fill_probability=0.05))
    rows = [{"simulation": i, **_simulate_equity(base, rng)} for i in range(n)]
    result = pd.DataFrame(rows)
    result.to_csv(REPORTS_DIR / "monte_carlo.csv", index=False)
    summary = {
        "median_return": float(result["total_return"].median()) if not result.empty else 0.0,
        "p05_return": float(result["total_return"].quantile(0.05)) if not result.empty else 0.0,
        "p95_return": float(result["total_return"].quantile(0.95)) if not result.empty else 0.0,
        "probability_of_loss": float((result["total_return"] < 0).mean()) if not result.empty else 1.0,
        "probability_pf_gt_1": float((result["profit_factor"] > 1).mean()) if not result.empty else 0.0,
        "probability_pf_gt_1_2": float((result["profit_factor"] > 1.2).mean()) if not result.empty else 0.0,
        "probability_dd_gt_20pct": float((result["max_drawdown"] < -0.20).mean()) if not result.empty else 0.0,
    }
    report = "# Monte Carlo Report\n\n" + "\n".join(f"- {k}: {v:.4f}" for k, v in summary.items()) + "\n"
    (REPORTS_DIR / "monte_carlo.md").write_text(report, encoding="utf-8")
    LOGGER.info("Saved Monte Carlo rows=%s", len(result))
    return result


if __name__ == "__main__":
    from scripts.common import setup_logging

    setup_logging()
    run_monte_carlo()
