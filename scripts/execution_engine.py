from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from scripts.common import LAB_CONFIG, PROCESSED_DIR, REPORTS_DIR, ensure_dirs, max_drawdown, profit_factor, read_parquet, sharpe_like

LOGGER = logging.getLogger(__name__)
MODE_TO_ENTRY = {"next_open": "open", "next_close": "close", "next_open_random_slippage": "open"}
HORIZON_ROWS = {"1h": 12, "4h": 48, "24h": 288}


@dataclass
class ExecutionConfig:
    mode: str = "next_open"
    fee_pct: float = 0.0004
    spread_pct: float = 0.0002
    slippage_pct: float = 0.0003
    missed_fill_probability: float = 0.0
    random_seed: int = 42


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = read_parquet(PROCESSED_DIR / f"{LAB_CONFIG.raw_symbol}_{LAB_CONFIG.timeframe}_features.parquet")
    predictions = read_parquet(PROCESSED_DIR / "predictions.parquet")
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], utc=True)
    return features.sort_values("timestamp").reset_index(drop=True), predictions.sort_values("timestamp").reset_index(drop=True)


def simulate_realistic_execution(config: ExecutionConfig = ExecutionConfig()) -> pd.DataFrame:
    ensure_dirs()
    features, predictions = _load()
    rng = np.random.default_rng(config.random_seed)
    rows: list[dict] = []
    balance = LAB_CONFIG.initial_balance
    feature_index = {ts: i for i, ts in enumerate(features["timestamp"])}
    for _, signal in predictions.iterrows():
        direction = signal["predicted_direction"]
        if direction == "flat":
            continue
        if rng.random() < config.missed_fill_probability:
            continue
        idx = feature_index.get(signal["timestamp"])
        if idx is None:
            continue
        horizon = str(signal.get("horizon", "4h"))
        horizon_rows = HORIZON_ROWS.get(horizon, 48)
        entry_idx = idx + 1
        exit_idx = entry_idx + horizon_rows
        if exit_idx >= len(features):
            continue
        entry_col = MODE_TO_ENTRY.get(config.mode, "open")
        entry_price = float(features.loc[entry_idx, entry_col])
        exit_price = float(features.loc[exit_idx, "close"])
        random_slip = rng.uniform(0, config.slippage_pct) if config.mode == "next_open_random_slippage" else config.slippage_pct
        side_mult = 1 if direction == "up" else -1
        adverse_entry = entry_price * (1 + side_mult * (config.spread_pct / 2 + random_slip))
        adverse_exit = exit_price * (1 - side_mult * (config.spread_pct / 2 + random_slip))
        gross_return = (adverse_exit / adverse_entry - 1) * side_mult
        net_return = gross_return - config.fee_pct * 2
        before = balance
        risk_budget = balance * LAB_CONFIG.risk_per_trade
        position_size = min(balance, risk_budget / max(abs(net_return), 0.002))
        pnl_usd = position_size * net_return
        balance += pnl_usd
        rows.append(
            {
                "signal_timestamp": signal["timestamp"],
                "entry_timestamp": features.loc[entry_idx, "timestamp"],
                "exit_timestamp": features.loc[exit_idx, "timestamp"],
                "symbol": signal["symbol"],
                "side": "LONG" if direction == "up" else "SHORT",
                "mode": config.mode,
                "fee_pct": config.fee_pct,
                "spread_pct": config.spread_pct,
                "slippage_pct": random_slip,
                "entry_price": adverse_entry,
                "exit_price": adverse_exit,
                "position_size": position_size,
                "confidence": float(signal["confidence"]),
                "pnl_usd": pnl_usd,
                "pnl_pct": pnl_usd / max(before, 1e-9),
                "balance_after": balance,
            }
        )
    return pd.DataFrame(rows)


def execution_metrics(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {"total_return": 0.0, "winrate": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "trades": 0}
    returns = trades["pnl_pct"].astype(float)
    equity = [LAB_CONFIG.initial_balance] + trades["balance_after"].astype(float).tolist()
    return {
        "total_return": float(trades["balance_after"].iloc[-1] / LAB_CONFIG.initial_balance - 1),
        "winrate": float((trades["pnl_usd"] > 0).mean()),
        "profit_factor": profit_factor(returns),
        "max_drawdown": max_drawdown(equity),
        "sharpe": sharpe_like(returns),
        "trades": int(len(trades)),
    }


def run_execution_sweep() -> pd.DataFrame:
    ensure_dirs()
    rows: list[dict] = []
    all_trades: list[pd.DataFrame] = []
    for mode in ["next_open", "next_close", "next_open_random_slippage"]:
        for fee in [0.0004, 0.0008]:
            for slip in [0.0003, 0.0005, 0.0010]:
                config = ExecutionConfig(mode=mode, fee_pct=fee, slippage_pct=slip)
                trades = simulate_realistic_execution(config)
                metrics = execution_metrics(trades)
                rows.append({**asdict(config), **metrics})
                if not trades.empty:
                    all_trades.append(trades.assign(config=f"{mode}_fee{fee}_slip{slip}"))
    result = pd.DataFrame(rows)
    result.to_csv(REPORTS_DIR / "realistic_execution.csv", index=False)
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(REPORTS_DIR / "realistic_execution_trades.csv", index=False)
    best = result.sort_values("profit_factor", ascending=False).head(10) if not result.empty else result
    report = "# Realistic Execution Report\n\n" + (best.to_string(index=False) if not best.empty else "No executable trades.")
    (REPORTS_DIR / "realistic_execution.md").write_text(report + "\n", encoding="utf-8")
    LOGGER.info("Saved realistic execution sweep rows=%s", len(result))
    return result


if __name__ == "__main__":
    from scripts.common import setup_logging

    setup_logging()
    run_execution_sweep()

