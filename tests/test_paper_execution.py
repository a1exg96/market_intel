from __future__ import annotations

import tempfile
import unittest
import os
import time
import math
from pathlib import Path

import pandas as pd

from scripts import paper_execution as pe
from scripts.paper_execution import PaperTradingConfig


class PaperExecutionTest(unittest.TestCase):
    def _edge_signal(self, side: str, price: float, confidence: float = 0.675) -> dict[str, object]:
        if side.upper() == "SHORT":
            stop_loss = price * 1.01
            take_profit = price * 0.98
        else:
            stop_loss = price * 0.99
            take_profit = price * 1.02
        return {
            "confidence": confidence,
            "expected_return": 0.004,
            "expected_risk": 0.011,
            "risk_reward": 2.0,
            "position_size": 1000.0,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "reason": "positive_expectancy_risk_managed_setup",
        }

    def test_dashboard_time_fields_are_returned_in_kyiv_time(self) -> None:
        from scripts.dashboard_app import _with_kyiv_times

        row = _with_kyiv_times({"opened_at": "2026-07-01T08:00:00+00:00", "symbol": "BTCUSDT"})

        self.assertEqual(row["opened_at"], "2026-07-01T11:00:00+03:00")
        self.assertEqual(row["symbol"], "BTCUSDT")

    def test_latest_local_prices_prefers_raw_candles_for_live_pnl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_dirs = (pe.RAW_DIR, pe.PROCESSED_DIR)
            pe.RAW_DIR = root / "raw"
            pe.PROCESSED_DIR = root / "processed"
            pe.RAW_DIR.mkdir()
            pe.PROCESSED_DIR.mkdir()
            try:
                pd.DataFrame(
                    [{"timestamp": "2026-07-01T08:10:00+00:00", "symbol": "BTCUSDT", "close": 58704.0}]
                ).to_parquet(pe.RAW_DIR / "BTCUSDT_5m_candles.parquet", index=False)
                pd.DataFrame(
                    [{"timestamp": "2026-06-30T08:10:00+00:00", "symbol": "BTCUSDT", "close": 59431.87}]
                ).to_parquet(pe.PROCESSED_DIR / "BTCUSDT_5m_features.parquet", index=False)

                self.assertEqual(pe.latest_local_prices()["BTCUSDT"], 58704.0)
            finally:
                pe.RAW_DIR, pe.PROCESSED_DIR = original_dirs

    def test_records_converts_nan_to_json_safe_none(self) -> None:
        rows = pe.records(pd.DataFrame([{"symbol": "BTCUSDT", "expected_return": float("nan")}]))

        self.assertIsNone(rows[0]["expected_return"])
        self.assertFalse(any(isinstance(value, float) and math.isnan(value) for value in rows[0].values()))

    def test_read_signals_deduplicates_timestamp_symbol_and_sorts_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp)
            original_path = pe.SIGNALS_PATH
            pe.SIGNALS_PATH = reports / "forward_signals.csv"
            try:
                pd.DataFrame(
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
                            "timestamp": "2026-07-01T10:05:00+00:00",
                            "generated_at": "2026-07-01T10:06:00+00:00",
                            "symbol": "ETHUSDT",
                            "regime": "RANGE",
                            "signal": "LONG",
                            "confidence": 0.61,
                            "entry_price": 2500,
                            "horizon": "4h",
                            "status": "PENDING",
                        },
                    ]
                ).to_csv(pe.SIGNALS_PATH, index=False)

                signals = pe.read_signals(limit=10)

                self.assertEqual(len(signals), 2)
                self.assertEqual(signals.iloc[0]["signal"], "SHORT")
                self.assertEqual(signals.iloc[1]["symbol"], "ETHUSDT")
            finally:
                pe.SIGNALS_PATH = original_path

    def test_stale_paper_state_lock_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp)
            original = (pe.LOCK_PATH, pe.STALE_LOCK_SECONDS)
            pe.LOCK_PATH = reports / ".paper_execution.lock"
            pe.STALE_LOCK_SECONDS = 1
            try:
                pe.LOCK_PATH.mkdir()
                old_time = time.time() - 10
                os.utime(pe.LOCK_PATH, (old_time, old_time))

                with pe._paper_state_lock():
                    self.assertTrue(pe.LOCK_PATH.exists())

                self.assertFalse(pe.LOCK_PATH.exists())
            finally:
                pe.LOCK_PATH, pe.STALE_LOCK_SECONDS = original

    def test_short_signal_opens_position_without_closing_trade(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp)
            original_paths = (
                pe.ACTIVE_POSITIONS_PATH,
                pe.TRADES_PATH,
                pe.AUDIT_PATH,
                pe.SIGNALS_PATH,
                pe.LOCK_PATH,
                pe.latest_local_prices,
            )
            pe.ACTIVE_POSITIONS_PATH = reports / "active_positions.csv"
            pe.TRADES_PATH = reports / "trades.csv"
            pe.AUDIT_PATH = reports / "signal_execution_audit.csv"
            pe.SIGNALS_PATH = reports / "forward_signals.csv"
            pe.LOCK_PATH = reports / ".paper_execution.lock"
            pe.latest_local_prices = lambda: {"BTCUSDT": 60000.0}
            try:
                config = PaperTradingConfig(initial_balance=1000.0, confidence_threshold=0.60)
                opened = pe.open_position_from_signal(
                    {
                        "symbol": "BTCUSDT",
                        "signal": "SHORT",
                        "confidence": 0.647,
                        "entry_price": 60360,
                        "regime": "RANGE",
                        "model_version": "test_v1",
                        "decision": "EXECUTABLE",
                        **self._edge_signal("SHORT", 60000.0, confidence=0.647),
                    },
                    config=config,
                )
                self.assertIsNotNone(opened)

                pe.latest_local_prices = lambda: {"BTCUSDT": 59900.0}
                pe.update_open_positions(config=config)
                active = pe.read_active_positions(open_only=True)
                trades = pe.read_trades()
                audit = pe.read_audit()
                stats = pe.stats_snapshot()

                self.assertEqual(len(active), 1)
                self.assertEqual(active.loc[0, "side"], "SHORT")
                self.assertEqual(len(trades), 0)
                self.assertEqual(stats["balance"], 1000.0)
                self.assertNotEqual(stats["equity"], stats["balance"])
                self.assertTrue(audit["executed"].astype(str).str.lower().eq("true").any())
            finally:
                (
                    pe.ACTIVE_POSITIONS_PATH,
                    pe.TRADES_PATH,
                    pe.AUDIT_PATH,
                    pe.SIGNALS_PATH,
                    pe.LOCK_PATH,
                    pe.latest_local_prices,
                ) = original_paths

    def test_down_executable_signal_opens_short_and_dashboard_api_returns_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp)
            original_paths = (
                pe.ACTIVE_POSITIONS_PATH,
                pe.TRADES_PATH,
                pe.AUDIT_PATH,
                pe.SIGNALS_PATH,
                pe.LOCK_PATH,
                pe.latest_local_prices,
            )
            pe.ACTIVE_POSITIONS_PATH = reports / "active_positions.csv"
            pe.TRADES_PATH = reports / "trades.csv"
            pe.AUDIT_PATH = reports / "signal_execution_audit.csv"
            pe.SIGNALS_PATH = reports / "forward_signals.csv"
            pe.LOCK_PATH = reports / ".paper_execution.lock"
            pe.latest_local_prices = lambda: {"BTCUSDT": 60000.0}
            try:
                config = PaperTradingConfig(initial_balance=1000.0, confidence_threshold=0.60)
                signal = {
                    "symbol": "BTCUSDT",
                    "signal": "DOWN",
                    "confidence": 0.675,
                    "price": 60360,
                    "regime": "UNKNOWN",
                    "model_version": "dual_setup_v0.2",
                    "decision": "Executable paper signal",
                    **self._edge_signal("SHORT", 60000.0),
                }

                opened = pe.open_position_from_signal(signal, config=config)
                self.assertIsNotNone(opened)

                from scripts.dashboard_app import api_active_positions, api_stats

                active = pe.read_active_positions(open_only=True)
                audit = pe.read_audit()
                stats = api_stats()
                api_positions = api_active_positions()

                self.assertEqual(len(active), 1)
                self.assertEqual(active.loc[0, "side"], "SHORT")
                self.assertEqual(active.loc[0, "status"], "OPEN")
                self.assertEqual(float(active.loc[0, "entry_price"]), 60000)
                self.assertEqual(list(audit.columns), pe.AUDIT_COLUMNS)
                self.assertTrue(audit["executed"].astype(str).str.lower().eq("true").any())
                self.assertEqual(stats["open_positions_count"], 1)
                self.assertEqual(len(api_positions), 1)
                self.assertEqual(api_positions[0]["side"], "SHORT")
            finally:
                (
                    pe.ACTIVE_POSITIONS_PATH,
                    pe.TRADES_PATH,
                    pe.AUDIT_PATH,
                    pe.SIGNALS_PATH,
                    pe.LOCK_PATH,
                    pe.latest_local_prices,
                ) = original_paths

    def test_closed_position_stays_closed_trade_and_balance_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp)
            original_paths = (
                pe.ACTIVE_POSITIONS_PATH,
                pe.TRADES_PATH,
                pe.AUDIT_PATH,
                pe.SIGNALS_PATH,
                pe.LOCK_PATH,
                pe.latest_local_prices,
            )
            pe.ACTIVE_POSITIONS_PATH = reports / "active_positions.csv"
            pe.TRADES_PATH = reports / "trades.csv"
            pe.AUDIT_PATH = reports / "signal_execution_audit.csv"
            pe.SIGNALS_PATH = reports / "forward_signals.csv"
            pe.LOCK_PATH = reports / ".paper_execution.lock"
            pe.latest_local_prices = lambda: {"BTCUSDT": 60360.0}
            try:
                config = PaperTradingConfig(
                    initial_balance=1000.0,
                    confidence_threshold=0.60,
                    take_profit_pct=0.1,
                    stop_loss_pct=5.0,
                )
                pe.open_position_from_signal(
                    {
                        "symbol": "BTCUSDT",
                        "signal": "LONG",
                        "confidence": 0.675,
                        "price": 60360,
                        "regime": "UNKNOWN",
                        "model_version": "dual_setup_v0.2",
                        "decision": "Executable paper signal",
                        **self._edge_signal("LONG", 60360.0),
                    },
                    config=config,
                )

                pe.update_open_positions(price_by_symbol={"BTCUSDT": 61000.0}, config=config)
                positions = pe.read_active_positions(open_only=False)
                open_positions = pe.read_active_positions(open_only=True)
                trades = pe.read_trades()
                stats = pe.stats_snapshot()

                self.assertEqual(len(positions), 1)
                self.assertEqual(positions.loc[0, "status"], "CLOSED")
                self.assertEqual(len(open_positions), 0)
                self.assertEqual(len(trades), 1)
                self.assertEqual(trades.loc[0, "position_id"], positions.loc[0, "position_id"])
                self.assertNotEqual(float(trades.loc[0, "balance_after"]), 1000.0)
                self.assertEqual(stats["trades_count"], 1)
                self.assertEqual(stats["balance"], float(trades.loc[0, "balance_after"]))
            finally:
                (
                    pe.ACTIVE_POSITIONS_PATH,
                    pe.TRADES_PATH,
                    pe.AUDIT_PATH,
                    pe.SIGNALS_PATH,
                    pe.LOCK_PATH,
                    pe.latest_local_prices,
                ) = original_paths

    def test_signal_uses_current_market_price_as_execution_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp)
            original_paths = (
                pe.ACTIVE_POSITIONS_PATH,
                pe.TRADES_PATH,
                pe.AUDIT_PATH,
                pe.SIGNALS_PATH,
                pe.LOCK_PATH,
                pe.latest_local_prices,
            )
            pe.ACTIVE_POSITIONS_PATH = reports / "active_positions.csv"
            pe.TRADES_PATH = reports / "trades.csv"
            pe.AUDIT_PATH = reports / "signal_execution_audit.csv"
            pe.SIGNALS_PATH = reports / "forward_signals.csv"
            pe.LOCK_PATH = reports / ".paper_execution.lock"
            pe.latest_local_prices = lambda: {"BTCUSDT": 58704.0}
            try:
                config = PaperTradingConfig(initial_balance=1000.0, confidence_threshold=0.60)
                pe.open_position_from_signal(
                    {
                        "symbol": "BTCUSDT",
                        "signal": "LONG",
                        "confidence": 0.675,
                        "price": 60360,
                        "regime": "UNKNOWN",
                        "model_version": "dual_setup_v0.2",
                        "decision": "Executable paper signal",
                        **self._edge_signal("LONG", 58704.0),
                    },
                    config=config,
                )
                pe.update_open_positions(config=config)
                active = pe.read_active_positions(open_only=True)
                trades = pe.read_trades()

                self.assertEqual(len(active), 1)
                self.assertEqual(float(active.loc[0, "entry_price"]), 58704.0)
                self.assertEqual(float(active.loc[0, "current_price"]), 58704.0)
                self.assertEqual(len(trades), 0)
            finally:
                (
                    pe.ACTIVE_POSITIONS_PATH,
                    pe.TRADES_PATH,
                    pe.AUDIT_PATH,
                    pe.SIGNALS_PATH,
                    pe.LOCK_PATH,
                    pe.latest_local_prices,
                ) = original_paths

    def test_settings_update_persists_leverage_stake_and_liquidation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "paper_trading.yaml"
            config_path.write_text(
                "live_trading: false\ninitial_balance: 1000.0\nrisk_per_trade: 0.01\n",
                encoding="utf-8",
            )

            settings = pe.update_paper_trading_settings(
                {
                    "leverage": 7,
                    "stake_pct": 0.5,
                    "liquidation_long_pct": 3.1,
                    "liquidation_short_pct": 4.2,
                },
                path=config_path,
            )
            config = pe.load_paper_trading_config(config_path)

            self.assertEqual(settings["leverage"], 7.0)
            self.assertEqual(settings["stake_pct"], 0.5)
            self.assertEqual(config.risk_per_trade, 0.005)
            self.assertEqual(config.liquidation_long_pct, 3.1)
            self.assertEqual(config.liquidation_short_pct, 4.2)

    def test_legacy_risk_above_one_percent_is_clamped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "paper_trading.yaml"
            config_path.write_text(
                "live_trading: false\ninitial_balance: 1000.0\nrisk_per_trade: 0.025\n",
                encoding="utf-8",
            )

            config = pe.load_paper_trading_config(config_path)

            self.assertEqual(config.risk_per_trade, 0.01)

    def test_leverage_multiplies_unrealized_pnl_and_stop_loss_prevents_liquidation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp)
            original_paths = (
                pe.ACTIVE_POSITIONS_PATH,
                pe.TRADES_PATH,
                pe.AUDIT_PATH,
                pe.SIGNALS_PATH,
                pe.LOCK_PATH,
                pe.latest_local_prices,
            )
            pe.ACTIVE_POSITIONS_PATH = reports / "active_positions.csv"
            pe.TRADES_PATH = reports / "trades.csv"
            pe.AUDIT_PATH = reports / "signal_execution_audit.csv"
            pe.SIGNALS_PATH = reports / "forward_signals.csv"
            pe.LOCK_PATH = reports / ".paper_execution.lock"
            pe.latest_local_prices = lambda: {"BTCUSDT": 100.0}
            try:
                config = PaperTradingConfig(
                    initial_balance=1000.0,
                    risk_per_trade=0.01,
                    leverage=5.0,
                    confidence_threshold=0.60,
                    take_profit_pct=50.0,
                    liquidation_long_pct=4.0,
                    liquidation_short_pct=4.0,
                )
                pe.open_position_from_signal(
                    {
                        "symbol": "BTCUSDT",
                        "signal": "LONG",
                        "confidence": 0.675,
                        "price": 100,
                        "decision": "EXECUTABLE",
                        **self._edge_signal("LONG", 100.0),
                    },
                    config=config,
                )

                pe.update_open_positions(price_by_symbol={"BTCUSDT": 99.0}, config=config)
                positions = pe.read_active_positions(open_only=False)
                trades = pe.read_trades()

                self.assertEqual(positions.loc[0, "status"], "CLOSED")
                self.assertEqual(trades.loc[0, "reason"], "STOP_LOSS")
                self.assertAlmostEqual(float(positions.loc[0, "unrealized_pnl_usd"]), -10.0, places=6)
            finally:
                (
                    pe.ACTIVE_POSITIONS_PATH,
                    pe.TRADES_PATH,
                    pe.AUDIT_PATH,
                    pe.SIGNALS_PATH,
                    pe.LOCK_PATH,
                    pe.latest_local_prices,
                ) = original_paths

    def test_high_confidence_without_positive_expectancy_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp)
            original_paths = (
                pe.ACTIVE_POSITIONS_PATH,
                pe.TRADES_PATH,
                pe.AUDIT_PATH,
                pe.SIGNALS_PATH,
                pe.LOCK_PATH,
                pe.latest_local_prices,
            )
            pe.ACTIVE_POSITIONS_PATH = reports / "active_positions.csv"
            pe.TRADES_PATH = reports / "trades.csv"
            pe.AUDIT_PATH = reports / "signal_execution_audit.csv"
            pe.SIGNALS_PATH = reports / "forward_signals.csv"
            pe.LOCK_PATH = reports / ".paper_execution.lock"
            pe.latest_local_prices = lambda: {"BTCUSDT": 100.0}
            try:
                config = PaperTradingConfig(initial_balance=1000.0, risk_per_trade=0.01, confidence_threshold=0.60)
                opened = pe.open_position_from_signal(
                    {
                        "symbol": "BTCUSDT",
                        "signal": "LONG",
                        "confidence": 0.95,
                        "price": 100,
                        "decision": "EXECUTABLE",
                    },
                    config=config,
                )
                audit = pe.read_audit()

                self.assertIsNone(opened)
                self.assertEqual(audit.loc[0, "reason"], "NON_POSITIVE_EXPECTANCY")
            finally:
                (
                    pe.ACTIVE_POSITIONS_PATH,
                    pe.TRADES_PATH,
                    pe.AUDIT_PATH,
                    pe.SIGNALS_PATH,
                    pe.LOCK_PATH,
                    pe.latest_local_prices,
                ) = original_paths

    def test_manual_close_moves_open_position_to_trades(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp)
            original_paths = (
                pe.ACTIVE_POSITIONS_PATH,
                pe.TRADES_PATH,
                pe.AUDIT_PATH,
                pe.SIGNALS_PATH,
                pe.LOCK_PATH,
                pe.latest_local_prices,
            )
            pe.ACTIVE_POSITIONS_PATH = reports / "active_positions.csv"
            pe.TRADES_PATH = reports / "trades.csv"
            pe.AUDIT_PATH = reports / "signal_execution_audit.csv"
            pe.SIGNALS_PATH = reports / "forward_signals.csv"
            pe.LOCK_PATH = reports / ".paper_execution.lock"
            pe.latest_local_prices = lambda: {"BTCUSDT": 100.0}
            try:
                config = PaperTradingConfig(initial_balance=1000.0, risk_per_trade=0.01, confidence_threshold=0.60)
                opened = pe.open_position_from_signal(
                    {
                        "symbol": "BTCUSDT",
                        "signal": "SHORT",
                        "confidence": 0.675,
                        "price": 100,
                        "decision": "EXECUTABLE",
                        **self._edge_signal("SHORT", 100.0),
                    },
                    config=config,
                )
                self.assertIsNotNone(opened)

                pe.latest_local_prices = lambda: {"BTCUSDT": 98.0}
                closed = pe.close_position_manually(opened["position_id"], config=config)
                active = pe.read_active_positions(open_only=True)
                trades = pe.read_trades()

                self.assertEqual(len(active), 0)
                self.assertEqual(len(trades), 1)
                self.assertEqual(closed["reason"], "MANUAL")
                self.assertEqual(trades.loc[0, "position_id"], opened["position_id"])
                self.assertGreater(float(trades.loc[0, "pnl_usd"]), 0)
            finally:
                (
                    pe.ACTIVE_POSITIONS_PATH,
                    pe.TRADES_PATH,
                    pe.AUDIT_PATH,
                    pe.SIGNALS_PATH,
                    pe.LOCK_PATH,
                    pe.latest_local_prices,
                ) = original_paths


if __name__ == "__main__":
    unittest.main()
