from __future__ import annotations

import unittest

import pandas as pd

from scripts.feature_engineering import _first_touch_targets


class FeatureEngineeringTargetTest(unittest.TestCase):
    def test_long_target_requires_take_profit_before_stop_loss(self) -> None:
        high = pd.Series([100.0, 100.4, 101.2])
        low = pd.Series([100.0, 98.8, 99.9])
        close = pd.Series([100.0, 100.0, 100.0])

        long_target, short_target = _first_touch_targets(high, low, close, horizon=2, target_threshold=0.01)

        self.assertEqual(long_target.iloc[0], 0)
        self.assertEqual(short_target.iloc[0], 1)

    def test_short_target_requires_take_profit_before_stop_loss(self) -> None:
        high = pd.Series([100.0, 101.2, 100.1])
        low = pd.Series([100.0, 99.7, 98.7])
        close = pd.Series([100.0, 100.0, 100.0])

        long_target, short_target = _first_touch_targets(high, low, close, horizon=2, target_threshold=0.01)

        self.assertEqual(long_target.iloc[0], 1)
        self.assertEqual(short_target.iloc[0], 0)

    def test_same_candle_take_profit_and_stop_loss_is_conservative(self) -> None:
        high = pd.Series([100.0, 101.5, 100.0])
        low = pd.Series([100.0, 98.5, 100.0])
        close = pd.Series([100.0, 100.0, 100.0])

        long_target, short_target = _first_touch_targets(high, low, close, horizon=2, target_threshold=0.01)

        self.assertEqual(long_target.iloc[0], 0)
        self.assertEqual(short_target.iloc[0], 0)


if __name__ == "__main__":
    unittest.main()
