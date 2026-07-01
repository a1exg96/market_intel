# Model Plugin Contract

This service is the data collector, risk gate, paper/live executor, and audit owner.
Models are replaceable plugins. A model receives a standard market snapshot, computes
its own probabilities or scores, and returns a standard decision object. The executor
must be able to consume the output without knowing which model produced it.

## Service Responsibilities

- Collect raw candles, futures context, and optional external context.
- Build or load normalized features.
- Call one or more model plugins with the same input contract.
- Validate every model output against the standard schema.
- Apply scoring, risk management, cooldowns, drawdown gates, and paper/live policy.
- Log all accepted and rejected decisions with reasons.
- Execute only paper trades by default.

## Model Responsibilities

- Never fetch private credentials or submit orders.
- Never write directly to paper/live trade ledgers.
- Never bypass service risk management.
- Use only data provided in the input object or explicitly declared local artifacts.
- Return exactly one decision per symbol/timeframe snapshot.
- Return `NO_TRADE` when the model cannot prove a usable edge.

## Input Schema

The service passes one model input object per symbol/timeframe decision point.

```json
{
  "schema_version": "model_input_v1",
  "request_id": "uuid-or-stable-id",
  "generated_at": "2026-07-02T00:00:00+00:00",
  "symbol": "BTCUSDT",
  "timeframe": "5m",
  "timestamp": "2026-07-02T00:00:00+00:00",
  "market": {
    "open": 60000.0,
    "high": 60100.0,
    "low": 59850.0,
    "close": 60050.0,
    "volume": 1234.5
  },
  "features": {
    "return_1h": 0.002,
    "return_4h": -0.004,
    "return_24h": 0.018,
    "volume_zscore": 1.2,
    "range_pct": 0.004,
    "atr": 0.008,
    "realized_volatility": 0.011,
    "momentum_12h": 0.006,
    "momentum_24h": 0.018,
    "funding_rate": 0.0001,
    "funding_zscore": 0.3,
    "open_interest": 1000000.0,
    "oi_change_1h": 0.001,
    "oi_change_4h": 0.006,
    "oi_change_24h": 0.012,
    "long_short_ratio": 1.05,
    "liquidation_imbalance": -0.01,
    "orderbook_imbalance": 0.2,
    "cumulative_delta": 0.04,
    "sentiment_score": 0.0
  },
  "regime": {
    "label": "neutral",
    "confidence": 0.65
  },
  "account": {
    "mode": "paper",
    "balance": 1000.0,
    "equity": 1000.0,
    "open_positions": 0,
    "risk_per_trade_max": 0.01,
    "live_trading_enabled": false
  },
  "execution_policy": {
    "fee_pct": 0.0004,
    "slippage_pct": 0.0003,
    "min_risk_reward": 1.5,
    "min_expected_return": 0.0005,
    "confidence_threshold": 0.6,
    "paper_required": true
  },
  "history": {
    "recent_loss_streak": 0,
    "max_drawdown": 0.0,
    "cooldown_active": false,
    "last_trade_side": null,
    "last_trade_closed_reason": null
  }
}
```

Required top-level fields:

- `schema_version`
- `request_id`
- `generated_at`
- `symbol`
- `timeframe`
- `timestamp`
- `market`
- `features`
- `regime`
- `account`
- `execution_policy`
- `history`

Rules:

- `timestamp` is the candle/snapshot timestamp being scored.
- `generated_at` is when the service requested the model decision.
- `features` must not contain future columns, target columns, realized future returns, or labels.
- Missing optional features should be omitted or set to `null`; models must handle missing values.
- The model may add internal diagnostics, but it must not mutate the input object.

## Output Schema

Each model returns one standard decision object.

```json
{
  "schema_version": "model_output_v1",
  "request_id": "same-as-input",
  "model_id": "my_model",
  "model_version": "my_model_v1.0.0",
  "generated_at": "2026-07-02T00:00:01+00:00",
  "symbol": "BTCUSDT",
  "timeframe": "5m",
  "timestamp": "2026-07-02T00:00:00+00:00",
  "side": "LONG",
  "confidence": 0.67,
  "long_probability": 0.67,
  "short_probability": 0.21,
  "expected_return": 0.0045,
  "expected_risk": 0.0025,
  "risk_reward": 1.8,
  "regime": "neutral",
  "volatility_state": "NORMAL",
  "entry_quality": 0.72,
  "position_size": 240.0,
  "stop_loss": 59450.0,
  "take_profit": 61120.0,
  "reason": "positive_expectancy_risk_managed_setup",
  "diagnostics": {
    "feature_set": "full",
    "calibration_method": "sigmoid",
    "training_window": "walk_forward",
    "known_limitations": []
  }
}
```

Required output fields:

- `schema_version`
- `request_id`
- `model_id`
- `model_version`
- `generated_at`
- `symbol`
- `timeframe`
- `timestamp`
- `side`
- `confidence`
- `expected_return`
- `expected_risk`
- `risk_reward`
- `regime`
- `volatility_state`
- `entry_quality`
- `position_size`
- `stop_loss`
- `take_profit`
- `reason`

Allowed `side` values:

- `LONG`
- `SHORT`
- `NO_TRADE`

Required numeric conventions:

- `confidence`: float from `0.0` to `1.0`.
- `expected_return`: expected net return after fees/slippage as a decimal, not percent.
- `expected_risk`: expected downside as a decimal, not percent.
- `risk_reward`: `expected_reward / expected_risk`.
- `entry_quality`: float from `0.0` to `1.0`.
- `position_size`: suggested notional size in quote currency. The executor may reduce it.
- `stop_loss` and `take_profit`: absolute prices.

Rules:

- For `NO_TRADE`, set `position_size` to `0.0`.
- For `NO_TRADE`, `stop_loss` and `take_profit` may be `0.0`, but `reason` must explain why.
- The model may suggest `position_size`, but the executor is authoritative.
- The executor must reject outputs with missing risk fields, invalid prices, low risk/reward, non-positive expectancy, or unsafe liquidation distance.
- `confidence` must not be used directly to increase position size unless calibration is proven and approved by service policy.

## Minimal Python Plugin Shape

Create a module under `scripts/models/` or another configured model path.

```python
from __future__ import annotations

from typing import Any


MODEL_ID = "example_model"
MODEL_VERSION = "example_model_v1.0.0"
INPUT_SCHEMA_VERSION = "model_input_v1"
OUTPUT_SCHEMA_VERSION = "model_output_v1"


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    features = payload["features"]
    market = payload["market"]
    price = float(market["close"])

    confidence = 0.0
    side = "NO_TRADE"
    reason = "no_statistical_edge"
    expected_return = 0.0
    expected_risk = 0.0
    risk_reward = 0.0
    position_size = 0.0
    stop_loss = 0.0
    take_profit = 0.0

    momentum = float(features.get("momentum_24h") or 0.0)
    atr = max(float(features.get("atr") or 0.0), 0.002)

    if momentum > 0.02:
        side = "LONG"
        confidence = 0.62
        expected_risk = atr * 1.6
        expected_return = expected_risk * 1.8
        risk_reward = expected_return / expected_risk
        stop_loss = price * (1 - expected_risk)
        take_profit = price * (1 + expected_return)
        position_size = 100.0
        reason = "momentum_edge_candidate"

    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "request_id": payload["request_id"],
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "generated_at": payload["generated_at"],
        "symbol": payload["symbol"],
        "timeframe": payload["timeframe"],
        "timestamp": payload["timestamp"],
        "side": side,
        "confidence": confidence,
        "long_probability": confidence if side == "LONG" else 0.0,
        "short_probability": confidence if side == "SHORT" else 0.0,
        "expected_return": expected_return,
        "expected_risk": expected_risk,
        "risk_reward": risk_reward,
        "regime": payload["regime"]["label"],
        "volatility_state": "NORMAL",
        "entry_quality": confidence,
        "position_size": position_size,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "reason": reason,
        "diagnostics": {"feature_set": "custom"},
    }
```

## Validation Checklist

Before a model can be used by `forward-paper` or live execution:

- It accepts `model_input_v1`.
- It returns `model_output_v1`.
- It returns `NO_TRADE` when required fields are missing.
- It never uses future/target columns as features.
- It has walk-forward validation.
- It has separate LONG and SHORT performance metrics.
- It reports calibration quality or marks confidence as uncalibrated.
- It has at least 500-1000 paper trades before live consideration.
- It remains profitable after fees and slippage.
- It passes liquidation safety checks through service risk management.

## Integration Flow

1. Add the model module.
2. Add a loader entry in the service model registry.
3. Convert current features into `model_input_v1`.
4. Call `predict(payload)`.
5. Validate output against `model_output_v1`.
6. Save the raw model decision for audit.
7. Pass the decision to scoring/risk management.
8. Execute only if the service returns an executable paper/live decision.

The model is never the final authority. The service risk manager and executor are.
