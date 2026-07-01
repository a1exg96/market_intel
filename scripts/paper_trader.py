from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import pandas as pd

from scripts.common import (
    LAB_CONFIG,
    PROCESSED_DIR,
    REPORTS_DIR,
    ensure_dirs,
    max_drawdown,
    profit_factor,
    read_parquet,
    sharpe_like,
    sortino_like,
)
from scripts.baseline_ml import train

LOGGER = logging.getLogger(__name__)
TRADE_COLUMNS = [
    "timestamp",
    "symbol",
    "side",
    "entry_price",
    "exit_price",
    "position_size",
    "confidence",
    "pnl_usd",
    "pnl_pct",
    "balance_after",
    "reason",
    "model_version",
]


@dataclass
class TradeStats:
    total_return: float
    winrate: float
    profit_factor: float
    max_drawdown: float
    avg_return_per_trade: float
    sharpe_like: float
    sortino_like: float
    number_of_trades: int
    long_count: int
    short_count: int


def load_predictions() -> pd.DataFrame:
    path = PROCESSED_DIR / "predictions.parquet"
    if path.exists():
        return read_parquet(path)
    return train()


def simulate(predictions: pd.DataFrame, initial_balance: float = LAB_CONFIG.initial_balance) -> pd.DataFrame:
    balance = initial_balance
    rows: list[dict] = []
    for _, signal in predictions.iterrows():
        direction = signal["predicted_direction"]
        if direction == "flat":
            continue
        gross_return = float(signal["future_return"])
        signed_return = gross_return if direction == "up" else -gross_return
        net_return = signed_return - LAB_CONFIG.fee_pct - LAB_CONFIG.slippage_pct
        risk_budget = balance * LAB_CONFIG.risk_per_trade
        position_size = min(balance, risk_budget / max(abs(net_return), 0.002))
        pnl_usd = position_size * net_return
        before = balance
        balance += pnl_usd
        rows.append(
            {
                "timestamp": signal["timestamp"],
                "symbol": signal["symbol"],
                "side": "LONG" if direction == "up" else "SHORT",
                "entry_price": float(signal["close"]),
                "exit_price": float(signal["close"]) * (1 + gross_return),
                "position_size": position_size,
                "confidence": float(signal["confidence"]),
                "pnl_usd": pnl_usd,
                "pnl_pct": pnl_usd / max(before, 1e-9),
                "balance_after": balance,
                "reason": f"paper_{direction}_confidence_{float(signal['confidence']):.3f}",
                "model_version": signal.get("model_version", "baseline_v0.1"),
            }
        )
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)


def compute_stats(trades: pd.DataFrame, initial_balance: float = LAB_CONFIG.initial_balance) -> TradeStats:
    if trades.empty:
        return TradeStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    returns = trades["pnl_pct"].astype(float)
    equity = [initial_balance] + trades["balance_after"].astype(float).tolist()
    return TradeStats(
        total_return=float(trades["balance_after"].iloc[-1] / initial_balance - 1),
        winrate=float((trades["pnl_usd"] > 0).mean()),
        profit_factor=profit_factor(returns),
        max_drawdown=max_drawdown(equity),
        avg_return_per_trade=float(returns.mean()),
        sharpe_like=sharpe_like(returns),
        sortino_like=sortino_like(returns),
        number_of_trades=int(len(trades)),
        long_count=int((trades["side"] == "LONG").sum()),
        short_count=int((trades["side"] == "SHORT").sum()),
    )


def run_paper() -> pd.DataFrame:
    ensure_dirs()
    predictions = load_predictions()
    trades = simulate(predictions)
    trades.to_csv(REPORTS_DIR / "trades.csv", index=False)
    pd.DataFrame([asdict(compute_stats(trades))]).to_csv(REPORTS_DIR / "paper_metrics.csv", index=False)
    LOGGER.info("Saved paper trades rows=%s", len(trades))
    return trades


if __name__ == "__main__":
    from scripts.common import setup_logging

    setup_logging()
    run_paper()
