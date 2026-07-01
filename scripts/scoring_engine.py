from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from scripts.common import LAB_CONFIG

MIN_RISK_REWARD = 1.5
MIN_EXPECTED_RETURN = 0.0005
MAX_RISK_PER_TRADE = 0.01
DEFAULT_RISK_PER_TRADE = 0.005
HIGH_VOL_ATR = 0.035
EXTREME_VOL_ATR = 0.060
CALIBRATION_WARN_ECE = 0.10


@dataclass(frozen=True)
class TradeScore:
    symbol: str
    side: str
    confidence: float
    expected_return: float
    expected_risk: float
    risk_reward: float
    regime: str
    volatility_state: str
    entry_quality: float
    position_size: float
    stop_loss: float
    take_profit: float
    reason: str
    executable: bool


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _regime(row: pd.Series) -> str:
    for column in row.index:
        if str(column).startswith("regime_") and bool(row.get(column)):
            return str(column).replace("regime_", "", 1)
    return str(row.get("regime", "UNKNOWN") or "UNKNOWN")


def _volatility_state(atr: float, realized_volatility: float) -> str:
    vol = max(atr, realized_volatility)
    if vol >= EXTREME_VOL_ATR:
        return "EXTREME"
    if vol >= HIGH_VOL_ATR:
        return "HIGH"
    if vol <= 0.006:
        return "LOW"
    return "NORMAL"


def _price_levels(side: str, entry_price: float, stop_distance: float, take_profit_distance: float) -> tuple[float, float]:
    if side == "SHORT":
        return entry_price * (1 + stop_distance), entry_price * (1 - take_profit_distance)
    return entry_price * (1 - stop_distance), entry_price * (1 + take_profit_distance)


def _liquidation_is_safe(side: str, entry_price: float, stop_loss: float, leverage: float) -> bool:
    leverage = max(leverage, 1.0)
    liquidation_distance = min(0.95 / leverage, 0.95)
    if side == "SHORT":
        liquidation_price = entry_price * (1 + liquidation_distance)
        return liquidation_price > stop_loss and (liquidation_price - stop_loss) / entry_price >= 0.005
    liquidation_price = entry_price * (1 - liquidation_distance)
    return liquidation_price < stop_loss and (stop_loss - liquidation_price) / entry_price >= 0.005


def _base_reason(reasons: list[str]) -> str:
    return "; ".join(reasons) if reasons else "positive_expectancy_risk_managed_setup"


def build_trade_score(
    row: pd.Series | dict[str, Any],
    long_probability: float,
    short_probability: float,
    *,
    confidence_threshold: float = LAB_CONFIG.confidence_threshold,
    target_threshold: float = 0.010,
    balance: float = LAB_CONFIG.initial_balance,
    risk_per_trade: float = DEFAULT_RISK_PER_TRADE,
    leverage: float = 1.0,
    calibration_ece: float | None = None,
) -> TradeScore:
    source = pd.Series(row)
    symbol = str(source.get("symbol", LAB_CONFIG.raw_symbol))
    entry_price = _finite(source.get("entry_price", source.get("close", source.get("price"))), 0.0)
    confidence = max(_finite(long_probability), _finite(short_probability))
    side = "LONG" if long_probability >= short_probability else "SHORT"
    side_probability = _finite(long_probability if side == "LONG" else short_probability)
    reasons: list[str] = []

    if entry_price <= 0:
        reasons.append("MISSING_ENTRY_PRICE")
        return TradeScore(symbol, "NO_TRADE", confidence, 0.0, 0.0, 0.0, _regime(source), "UNKNOWN", 0.0, 0.0, 0.0, 0.0, _base_reason(reasons), False)

    atr = max(_finite(source.get("atr"), 0.0), 0.002)
    realized_volatility = max(_finite(source.get("realized_volatility"), 0.0), 0.0)
    volatility_state = _volatility_state(atr, realized_volatility)
    regime = _regime(source)
    cost = float(LAB_CONFIG.fee_pct * 2 + LAB_CONFIG.slippage_pct)
    stop_distance = min(max(atr * 1.6, 0.005), 0.035)
    if volatility_state == "HIGH":
        stop_distance = min(max(stop_distance, atr * 1.2), 0.045)
    take_profit_distance = max(stop_distance * 1.8, target_threshold * 1.5, cost * 6)
    expected_return = side_probability * take_profit_distance - (1 - side_probability) * stop_distance - cost
    expected_risk = stop_distance + cost
    risk_reward = take_profit_distance / max(stop_distance, 1e-9)
    entry_quality = max(0.0, min(1.0, (side_probability - 0.5) * 2 + max(expected_return, 0.0) * 20))
    risk_fraction = min(max(float(risk_per_trade), 0.0), MAX_RISK_PER_TRADE)

    if calibration_ece is not None and calibration_ece > CALIBRATION_WARN_ECE:
        risk_fraction = min(risk_fraction, DEFAULT_RISK_PER_TRADE)
        reasons.append("CONFIDENCE_NOT_WELL_CALIBRATED")
    if volatility_state == "HIGH":
        risk_fraction *= 0.5
        reasons.append("HIGH_VOL_REDUCED_SIZE")
    if volatility_state == "EXTREME":
        reasons.append("EXTREME_VOLATILITY")
    if side_probability < confidence_threshold:
        reasons.append("LOW_CONFIDENCE")
    if expected_return <= MIN_EXPECTED_RETURN:
        reasons.append("NON_POSITIVE_EXPECTANCY")
    if risk_reward < MIN_RISK_REWARD:
        reasons.append("LOW_RISK_REWARD")
    if regime.upper() in {"PANIC_BLOCKED"}:
        reasons.append("REGIME_BLOCKED")

    stop_loss, take_profit = _price_levels(side, entry_price, stop_distance, take_profit_distance)
    if not _liquidation_is_safe(side, entry_price, stop_loss, leverage):
        reasons.append("STOP_TOO_CLOSE_TO_LIQUIDATION")

    executable = not any(
        reason
        in {
            "EXTREME_VOLATILITY",
            "LOW_CONFIDENCE",
            "NON_POSITIVE_EXPECTANCY",
            "LOW_RISK_REWARD",
            "REGIME_BLOCKED",
            "STOP_TOO_CLOSE_TO_LIQUIDATION",
            "MISSING_ENTRY_PRICE",
        }
        for reason in reasons
    )
    final_side = side if executable else "NO_TRADE"
    risk_budget = max(balance, 0.0) * risk_fraction
    position_size = 0.0 if final_side == "NO_TRADE" else min(max(balance, 0.0), risk_budget / max(stop_distance, 1e-9))
    return TradeScore(
        symbol=symbol,
        side=final_side,
        confidence=confidence,
        expected_return=float(expected_return),
        expected_risk=float(expected_risk),
        risk_reward=float(risk_reward),
        regime=regime,
        volatility_state=volatility_state,
        entry_quality=float(entry_quality),
        position_size=float(position_size),
        stop_loss=float(stop_loss),
        take_profit=float(take_profit),
        reason=_base_reason(reasons),
        executable=executable,
    )


def score_to_dict(score: TradeScore) -> dict[str, Any]:
    return asdict(score)
