from __future__ import annotations

import unittest

import pandas as pd

from scripts.forward_paper_engine import SIGNAL_COLUMNS, _dedupe_by_symbol_time


class ForwardPaperEngineTest(unittest.TestCase):
    def test_dedupe_by_symbol_time_keeps_latest_generated_signal(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "timestamp": "2026-07-01T10:00:00+00:00",
                    "generated_at": "2026-07-01T10:01:00+00:00",
                    "symbol": "BTCUSDT",
                    "regime": "RANGE",
                    "signal": "NO_TRADE",
                    "confidence": 0.30,
                    "entry_price": 60000,
                    "horizon": "4h",
                    "status": "PENDING",
                },
                {
                    "timestamp": "2026-07-01 10:00:00+00:00",
                    "generated_at": "2026-07-01T10:02:00+00:00",
                    "symbol": "BTCUSDT",
                    "regime": "RANGE",
                    "signal": "SHORT",
                    "confidence": 0.67,
                    "entry_price": 59900,
                    "horizon": "4h",
                    "status": "PENDING",
                },
                {
                    "timestamp": "2026-07-01T10:00:00+00:00",
                    "generated_at": "2026-07-01T10:01:00+00:00",
                    "symbol": "ETHUSDT",
                    "regime": "RANGE",
                    "signal": "LONG",
                    "confidence": 0.61,
                    "entry_price": 2500,
                    "horizon": "4h",
                    "status": "PENDING",
                },
            ]
        )

        deduped = _dedupe_by_symbol_time(frame, SIGNAL_COLUMNS)

        self.assertEqual(len(deduped), 2)
        btc = deduped[deduped["symbol"] == "BTCUSDT"].iloc[0]
        eth = deduped[deduped["symbol"] == "ETHUSDT"].iloc[0]
        self.assertEqual(btc["signal"], "SHORT")
        self.assertEqual(eth["signal"], "LONG")


if __name__ == "__main__":
    unittest.main()
