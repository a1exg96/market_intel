from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts import paper_execution as pe
from scripts import telegram_notifier as tn


class TelegramNotifierTest(unittest.TestCase):
    def test_opened_position_sends_one_notification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp)
            original_paths = (
                pe.ACTIVE_POSITIONS_PATH,
                pe.TRADES_PATH,
                pe.AUDIT_PATH,
                pe.SIGNALS_PATH,
                pe.LOCK_PATH,
                tn.NOTIFICATIONS_PATH,
            )
            pe.ACTIVE_POSITIONS_PATH = reports / "active_positions.csv"
            pe.TRADES_PATH = reports / "trades.csv"
            pe.AUDIT_PATH = reports / "signal_execution_audit.csv"
            pe.SIGNALS_PATH = reports / "forward_signals.csv"
            pe.LOCK_PATH = reports / ".paper_execution.lock"
            tn.NOTIFICATIONS_PATH = reports / "telegram_notifications.csv"
            try:
                pd.DataFrame(
                    [
                        {
                            "position_id": "pos_test",
                            "opened_at": "2026-07-01T08:00:00+00:00",
                            "symbol": "BTCUSDT",
                            "side": "SHORT",
                            "entry_price": 60360.0,
                            "current_price": 60360.0,
                            "position_size": 10.0,
                            "confidence": 0.675,
                            "regime": "UNKNOWN",
                            "model_version": "dual_setup_v0.2",
                            "reason": "paper_short_confidence_0.675",
                            "unrealized_pnl_usd": 0.0,
                            "unrealized_pnl_pct": 0.0,
                            "status": "OPEN",
                        }
                    ],
                    columns=pe.ACTIVE_POSITION_COLUMNS,
                ).to_csv(pe.ACTIVE_POSITIONS_PATH, index=False)
                pd.DataFrame(
                    [
                        {
                            "timestamp": "2026-07-01T08:00:00+00:00",
                            "symbol": "BTCUSDT",
                            "signal": "SHORT",
                            "confidence": 0.675,
                            "price": 60360.0,
                            "decision": "Executable paper signal",
                            "executed": True,
                            "reason": "OPENED",
                            "position_id": "pos_test",
                        }
                    ],
                    columns=pe.AUDIT_COLUMNS,
                ).to_csv(pe.AUDIT_PATH, index=False)

                sent_messages: list[str] = []
                with patch.dict(
                    os.environ,
                    {
                        "TELEGRAM_NOTIFICATIONS_ENABLED": "true",
                        "TELEGRAM_BOT_TOKEN": "token",
                        "TELEGRAM_CHAT_ID": "123",
                        "TELEGRAM_MIN_CONFIDENCE": "0.60",
                    },
                    clear=False,
                ), patch.object(tn, "send_telegram_message", side_effect=sent_messages.append):
                    self.assertEqual(tn.notify_new_strong_signals(), 1)
                    self.assertEqual(tn.notify_new_strong_signals(), 0)

                notifications = tn.read_notifications()
                self.assertEqual(len(sent_messages), 1)
                self.assertIn("BTCUSDT", sent_messages[0])
                self.assertIn("SHORT", sent_messages[0])
                self.assertEqual(len(notifications), 1)
                self.assertEqual(notifications.loc[0, "status"], "SENT")
            finally:
                (
                    pe.ACTIVE_POSITIONS_PATH,
                    pe.TRADES_PATH,
                    pe.AUDIT_PATH,
                    pe.SIGNALS_PATH,
                    pe.LOCK_PATH,
                    tn.NOTIFICATIONS_PATH,
                ) = original_paths


if __name__ == "__main__":
    unittest.main()
