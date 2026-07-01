from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from scripts.common import LAB_CONFIG, PROCESSED_DIR, REPORTS_DIR, ensure_dirs, max_drawdown, profit_factor, read_parquet, sharpe_like

LOGGER = logging.getLogger(__name__)


def _metrics(name: str, returns: pd.Series) -> dict[str, float | str | int]:
    returns = returns.dropna().astype(float)
    if returns.empty:
        return {"benchmark": name, "trades": 0, "total_return": 0.0, "winrate": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}
    equity = LAB_CONFIG.initial_balance * (1 + returns).cumprod()
    return {
        "benchmark": name,
        "trades": int(len(returns)),
        "total_return": float(equity.iloc[-1] / LAB_CONFIG.initial_balance - 1),
        "winrate": float((returns > 0).mean()),
        "profit_factor": profit_factor(returns),
        "max_drawdown": max_drawdown([LAB_CONFIG.initial_balance] + equity.tolist()),
        "sharpe": sharpe_like(returns),
    }


def run_benchmarks() -> pd.DataFrame:
    ensure_dirs()
    features = read_parquet(PROCESSED_DIR / f"{LAB_CONFIG.raw_symbol}_{LAB_CONFIG.timeframe}_features.parquet")
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)
    rows: list[dict] = []
    rows.append(_metrics("buy_and_hold_4h", features["future_return_4h"].iloc[::48] - LAB_CONFIG.fee_pct * 2))
    rng = np.random.default_rng(81)
    random_side = rng.choice([-1, 1], size=len(features))
    rows.append(_metrics("random_long_short_4h", pd.Series(random_side * features["future_return_4h"].to_numpy()) - LAB_CONFIG.fee_pct * 2 - LAB_CONFIG.slippage_pct))
    momentum = np.sign(features["momentum_24h"].fillna(0)).replace(0, np.nan)
    rows.append(_metrics("momentum_baseline_4h", momentum * features["future_return_4h"] - LAB_CONFIG.fee_pct * 2 - LAB_CONFIG.slippage_pct))
    breakout = features["range_pct"] > features["range_pct"].rolling(48, min_periods=12).quantile(0.8)
    rows.append(_metrics("volatility_breakout_4h", features.loc[breakout, "future_return_4h"] - LAB_CONFIG.fee_pct * 2 - LAB_CONFIG.slippage_pct))
    result = pd.DataFrame(rows)
    result.to_csv(REPORTS_DIR / "benchmark_results.csv", index=False)
    report = "# Benchmark Report\n\n" + result.to_string(index=False) + "\n"
    (REPORTS_DIR / "benchmark_report.md").write_text(report, encoding="utf-8")
    LOGGER.info("Saved benchmark rows=%s", len(result))
    return result


if __name__ == "__main__":
    from scripts.common import setup_logging

    setup_logging()
    run_benchmarks()
