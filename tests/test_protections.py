from __future__ import annotations

import unittest
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from scripts.protections import evaluate_protections


@dataclass
class DummyConfig:
    initial_balance: float = 1000.0
    max_daily_trades: int = 6
    max_daily_loss_pct: float = 0.02
    max_consecutive_losses: int = 2
    cooldown_minutes: int = 180
    symbol_loss_cooldown_minutes: int = 360
    max_drawdown_live_block: float = 0.10


class ProtectionsTest(unittest.TestCase):
    def test_blocks_after_consecutive_losses(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        trades = pd.DataFrame(
            [
                {"symbol": "BTCUSDT", "closed_at": now, "pnl_usd": -1.0, "balance_after": 999.0, "reason": "STOP_LOSS"},
                {"symbol": "ETHUSDT", "closed_at": now, "pnl_usd": -1.0, "balance_after": 998.0, "reason": "STOP_LOSS"},
            ]
        )

        decision = evaluate_protections("SOLUSDT", trades, DummyConfig())

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "CONSECUTIVE_LOSS_LIMIT")

    def test_blocks_daily_loss_before_new_entry(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        trades = pd.DataFrame(
            [
                {"symbol": "BTCUSDT", "closed_at": now, "pnl_usd": -25.0, "balance_after": 975.0, "reason": "STOP_LOSS"},
            ]
        )

        decision = evaluate_protections("SOLUSDT", trades, DummyConfig(max_consecutive_losses=3))

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "DAILY_LOSS_LIMIT")


if __name__ == "__main__":
    unittest.main()
