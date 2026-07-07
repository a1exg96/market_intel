from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from scripts.common import max_drawdown


@dataclass(frozen=True)
class ProtectionDecision:
    allowed: bool
    reason: str = ""


def _closed_at(trades: pd.DataFrame) -> pd.Series:
    if trades.empty or "closed_at" not in trades:
        return pd.Series(dtype="datetime64[ns, UTC]")
    return pd.to_datetime(trades["closed_at"], utc=True, errors="coerce", format="mixed")


def _daily_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "closed_at" not in trades:
        return trades.iloc[0:0]
    start_of_day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return trades[_closed_at(trades).ge(start_of_day)]


def consecutive_losses(trades: pd.DataFrame) -> int:
    if trades.empty or "pnl_usd" not in trades:
        return 0
    pnl = pd.to_numeric(trades["pnl_usd"], errors="coerce").fillna(0.0)
    streak = 0
    for value in reversed(pnl.tolist()):
        if value < 0:
            streak += 1
        else:
            break
    return streak


def daily_trade_limit_hit(trades: pd.DataFrame, max_daily_trades: int) -> bool:
    if max_daily_trades <= 0:
        return False
    return len(_daily_trades(trades)) >= int(max_daily_trades)


def daily_loss_limit_hit(trades: pd.DataFrame, initial_balance: float, max_daily_loss_pct: float) -> bool:
    if trades.empty or max_daily_loss_pct <= 0:
        return False
    today = _daily_trades(trades)
    if today.empty:
        return False
    pnl = pd.to_numeric(today["pnl_usd"], errors="coerce").fillna(0.0).sum()
    return float(pnl) <= -(float(initial_balance) * float(max_daily_loss_pct))


def cooldown_active(symbol: str, trades: pd.DataFrame, cooldown_minutes: int) -> bool:
    if trades.empty or cooldown_minutes <= 0:
        return False
    if not {"symbol", "closed_at", "reason"}.issubset(trades.columns):
        return False
    recent_cutoff = datetime.now(timezone.utc) - timedelta(minutes=cooldown_minutes)
    reasons = trades["reason"].astype(str).str.upper()
    matches = (
        (trades["symbol"].astype(str) == symbol)
        & _closed_at(trades).ge(recent_cutoff)
        & reasons.isin(["STOP_LOSS", "LIQUIDATION", "MANUAL"])
    )
    return bool(matches.any())


def symbol_loss_cooldown_active(symbol: str, trades: pd.DataFrame, cooldown_minutes: int) -> bool:
    if trades.empty or cooldown_minutes <= 0:
        return False
    if not {"symbol", "closed_at", "pnl_usd"}.issubset(trades.columns):
        return False
    recent_cutoff = datetime.now(timezone.utc) - timedelta(minutes=cooldown_minutes)
    pnl = pd.to_numeric(trades["pnl_usd"], errors="coerce").fillna(0.0)
    matches = (trades["symbol"].astype(str) == symbol) & _closed_at(trades).ge(recent_cutoff) & (pnl < 0)
    return bool(matches.any())


def historical_drawdown_hit(trades: pd.DataFrame, initial_balance: float, max_drawdown_live_block: float) -> bool:
    balances = [float(initial_balance)]
    if not trades.empty and "balance_after" in trades:
        balances.extend(pd.to_numeric(trades["balance_after"], errors="coerce").dropna().astype(float).tolist())
    return abs(max_drawdown(balances)) >= float(max_drawdown_live_block)


def evaluate_protections(symbol: str, trades: pd.DataFrame, config: Any) -> ProtectionDecision:
    if daily_trade_limit_hit(trades, int(getattr(config, "max_daily_trades", 0))):
        return ProtectionDecision(False, "DAILY_TRADE_LIMIT")
    if daily_loss_limit_hit(
        trades,
        float(getattr(config, "initial_balance", 0.0)),
        float(getattr(config, "max_daily_loss_pct", 0.0)),
    ):
        return ProtectionDecision(False, "DAILY_LOSS_LIMIT")
    max_consecutive_losses = int(getattr(config, "max_consecutive_losses", 0))
    if max_consecutive_losses > 0 and consecutive_losses(trades) >= max_consecutive_losses:
        return ProtectionDecision(False, "CONSECUTIVE_LOSS_LIMIT")
    if cooldown_active(symbol, trades, int(getattr(config, "cooldown_minutes", 0))):
        return ProtectionDecision(False, "COOLDOWN_ACTIVE")
    if symbol_loss_cooldown_active(symbol, trades, int(getattr(config, "symbol_loss_cooldown_minutes", 0))):
        return ProtectionDecision(False, "SYMBOL_LOSS_COOLDOWN")
    if historical_drawdown_hit(
        trades,
        float(getattr(config, "initial_balance", 0.0)),
        float(getattr(config, "max_drawdown_live_block", 1.0)),
    ):
        return ProtectionDecision(False, "DRAWDOWN_LIVE_BLOCK")
    return ProtectionDecision(True)
