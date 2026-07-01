from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from scripts.common import LAB_CONFIG, PROCESSED_DIR, REPORTS_DIR, ensure_dirs, max_drawdown, profit_factor, read_parquet, sharpe_like, sortino_like

LOGGER = logging.getLogger(__name__)


def _read_csv(name: str) -> pd.DataFrame:
    path = REPORTS_DIR / name
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _predictions() -> pd.DataFrame:
    path = PROCESSED_DIR / "predictions.parquet"
    if not path.exists():
        return pd.DataFrame()
    return read_parquet(path)


def _trade_returns(trades: pd.DataFrame) -> pd.Series:
    if trades.empty or "pnl_pct" not in trades:
        return pd.Series(dtype="float64")
    values = pd.to_numeric(trades["pnl_pct"], errors="coerce").dropna()
    if values.abs().median() > 1:
        values = values / 100
    return values


def _trade_metrics(trades: pd.DataFrame) -> dict[str, float | int]:
    returns = _trade_returns(trades)
    pnl = pd.to_numeric(trades.get("pnl_usd", pd.Series(dtype=float)), errors="coerce").dropna()
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    balances = [LAB_CONFIG.initial_balance]
    if "balance_after" in trades:
        balances.extend(pd.to_numeric(trades["balance_after"], errors="coerce").dropna().astype(float).tolist())
    return {
        "trades": int(len(trades)),
        "winrate": float((pnl > 0).mean()) if not pnl.empty else 0.0,
        "average_win": float(wins.mean()) if not wins.empty else 0.0,
        "average_loss": float(losses.mean()) if not losses.empty else 0.0,
        "expectancy": float(pnl.mean()) if not pnl.empty else 0.0,
        "profit_factor": profit_factor(returns),
        "max_drawdown": max_drawdown(balances),
        "sharpe": sharpe_like(returns),
        "sortino": sortino_like(returns),
    }


def _precision(predictions: pd.DataFrame, side: str) -> float:
    if predictions.empty or "side" not in predictions:
        return 0.0
    selected = predictions[predictions["side"].astype(str).str.upper() == side]
    if selected.empty:
        return 0.0
    actual_col = "actual_long" if side == "LONG" else "actual_short"
    if actual_col not in selected:
        return 0.0
    return float(pd.to_numeric(selected[actual_col], errors="coerce").fillna(0).eq(1).mean())


def _brier(predictions: pd.DataFrame, probability_col: str, actual_col: str) -> float:
    if predictions.empty or probability_col not in predictions or actual_col not in predictions:
        return 0.0
    data = predictions[[probability_col, actual_col]].dropna()
    if data.empty:
        return 0.0
    actual = pd.to_numeric(data[actual_col], errors="coerce").fillna(0).clip(0, 1)
    probability = pd.to_numeric(data[probability_col], errors="coerce").fillna(0).clip(0, 1)
    return float(brier_score_loss(actual, probability))


def _performance_table(frame: pd.DataFrame, group_col: str) -> str:
    if frame.empty or group_col not in frame or "pnl_usd" not in frame:
        return "n/a"
    grouped = frame.groupby(group_col, dropna=False).agg(
        trades=("pnl_usd", "size"),
        winrate=("pnl_usd", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
        expectancy=("pnl_usd", lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
    )
    return grouped.to_string()


def _confidence_buckets(predictions: pd.DataFrame) -> str:
    if predictions.empty or "confidence" not in predictions:
        return "n/a"
    work = predictions.copy()
    work["bucket"] = pd.cut(pd.to_numeric(work["confidence"], errors="coerce"), np.linspace(0, 1, 11), include_lowest=True)
    actual_win = None
    if {"side", "actual_long", "actual_short"}.issubset(work.columns):
        actual_win = (
            ((work["side"].astype(str) == "LONG") & (pd.to_numeric(work["actual_long"], errors="coerce") == 1))
            | ((work["side"].astype(str) == "SHORT") & (pd.to_numeric(work["actual_short"], errors="coerce") == 1))
        )
    rows = work.groupby("bucket", observed=False).agg(
        signals=("confidence", "size"),
        avg_confidence=("confidence", "mean"),
        no_trade=("side", lambda s: int((s.astype(str) == "NO_TRADE").sum())) if "side" in work else ("confidence", "size"),
    )
    if actual_win is not None:
        rows["precision"] = actual_win.groupby(work["bucket"], observed=False).mean()
    return rows.to_string()


def build_trading_audit_report() -> str:
    ensure_dirs()
    predictions = _predictions()
    forward_signals = _read_csv("forward_signals.csv")
    audit = _read_csv("signal_execution_audit.csv")
    trades = _read_csv("trades.csv")
    rejections = _read_csv("trade_rejections.csv")
    metrics = _trade_metrics(trades)
    no_trade = int((predictions.get("side", pd.Series(dtype=str)).astype(str) == "NO_TRADE").sum()) if not predictions.empty else 0
    rejected_reasons = "n/a"
    if not rejections.empty and "reason_rejected" in rejections:
        rejected_reasons = rejections["reason_rejected"].value_counts().to_string()
    elif not audit.empty and "reason" in audit:
        rejected_reasons = audit[audit["executed"].astype(str).str.lower() != "true"]["reason"].value_counts().to_string()

    model_errors = "n/a"
    if not predictions.empty and {"side", "actual_long", "actual_short"}.issubset(predictions.columns):
        executable = predictions[predictions["side"].astype(str).isin(["LONG", "SHORT"])].copy()
        if not executable.empty:
            correct = (
                ((executable["side"].astype(str) == "LONG") & (pd.to_numeric(executable["actual_long"], errors="coerce") == 1))
                | ((executable["side"].astype(str) == "SHORT") & (pd.to_numeric(executable["actual_short"], errors="coerce") == 1))
            )
            model_errors = executable.loc[~correct, ["timestamp", "symbol", "side", "confidence", "reason"]].head(20).to_string(index=False)

    report = f"""# Trading Audit Report

## Decision Funnel

- Model predictions: {len(predictions)}
- Forward signals logged: {len(forward_signals)}
- NO_TRADE decisions: {no_trade}
- Paper trades opened/closed: {len(trades)}
- Executed audit rows: {int((audit["executed"].astype(str).str.lower() == "true").sum()) if not audit.empty and "executed" in audit else 0}

## Rejected Signals

{rejected_reasons}

## Trade Quality

- Winrate: {metrics["winrate"]:.2%}
- Average win: {metrics["average_win"]:.4f} USD
- Average loss: {metrics["average_loss"]:.4f} USD
- Expectancy: {metrics["expectancy"]:.4f} USD/trade
- Profit factor: {metrics["profit_factor"]:.3f}
- Max drawdown: {metrics["max_drawdown"]:.2%}
- Sharpe-like: {metrics["sharpe"]:.3f}
- Sortino-like: {metrics["sortino"]:.3f}

## Model Quality

- LONG precision: {_precision(predictions, "LONG"):.2%}
- SHORT precision: {_precision(predictions, "SHORT"):.2%}
- LONG Brier score: {_brier(predictions, "long_probability", "actual_long"):.4f}
- SHORT Brier score: {_brier(predictions, "short_probability", "actual_short"):.4f}

## Performance By Symbol

{_performance_table(trades, "symbol")}

## Performance By Regime

{_performance_table(trades, "regime")}

## Confidence Buckets

{_confidence_buckets(predictions)}

## Model Errors

{model_errors}

## Weak Spots

- Live trading remains disabled by default.
- A live release needs at least 500-1000 statistically valid paper trades with positive expectancy and stable drawdown.
- Confidence is recorded for audit, but position size is controlled by risk-to-stop, not by confidence.
"""
    (REPORTS_DIR / "trading_audit_report.md").write_text(report, encoding="utf-8")
    LOGGER.info("Saved trading audit report.")
    return report


if __name__ == "__main__":
    from scripts.common import setup_logging

    setup_logging()
    print(build_trading_audit_report())
