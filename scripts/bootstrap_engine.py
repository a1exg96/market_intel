from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from scripts.common import LAB_CONFIG, REPORTS_DIR, ensure_dirs, max_drawdown, profit_factor

LOGGER = logging.getLogger(__name__)


def _source_returns() -> pd.Series:
    for filename, column in [("forward_results.csv", "trade_return"), ("trades.csv", "pnl_pct")]:
        path = REPORTS_DIR / filename
        if path.exists() and path.stat().st_size > 0:
            df = pd.read_csv(path)
            if column in df and not df.empty:
                return df[column].astype(float).dropna()
    return pd.Series(dtype="float64")


def run_bootstrap(n: int = 10000) -> pd.DataFrame:
    ensure_dirs()
    returns = _source_returns()
    rng = np.random.default_rng(2031)
    rows: list[dict[str, float | int]] = []
    if returns.empty:
        result = pd.DataFrame(columns=["simulation", "total_return", "profit_factor", "winrate", "max_drawdown"])
    else:
        values = returns.to_numpy()
        for i in range(n):
            sample = rng.choice(values, size=len(values), replace=True)
            equity = LAB_CONFIG.initial_balance * (1 + pd.Series(sample)).cumprod()
            rows.append(
                {
                    "simulation": i,
                    "total_return": float(equity.iloc[-1] / LAB_CONFIG.initial_balance - 1),
                    "profit_factor": min(profit_factor(sample), 100.0),
                    "winrate": float((sample > 0).mean()),
                    "max_drawdown": max_drawdown([LAB_CONFIG.initial_balance] + equity.tolist()),
                }
            )
        result = pd.DataFrame(rows)
    result.to_csv(REPORTS_DIR / "bootstrap_results.csv", index=False)
    if result.empty:
        summary = pd.DataFrame([{"metric": "insufficient_data", "p05": 0, "median": 0, "p95": 0}])
    else:
        summary = pd.DataFrame(
            [
                {"metric": col, "p05": result[col].quantile(0.05), "median": result[col].median(), "p95": result[col].quantile(0.95)}
                for col in ["total_return", "profit_factor", "winrate", "max_drawdown"]
            ]
        )
    summary.to_csv(REPORTS_DIR / "bootstrap_summary.csv", index=False)
    report = "# Bootstrap Report\n\n" + summary.to_string(index=False) + "\n"
    (REPORTS_DIR / "bootstrap_report.md").write_text(report, encoding="utf-8")
    LOGGER.info("Saved bootstrap rows=%s", len(result))
    return result


if __name__ == "__main__":
    from scripts.common import setup_logging

    setup_logging()
    run_bootstrap()
