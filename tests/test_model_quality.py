from __future__ import annotations

import unittest

import pandas as pd

from scripts.model_quality import QualityPolicy, apply_model_quality_gate, expected_calibration_error


class ModelQualityTest(unittest.TestCase):
    def test_expected_calibration_error_detects_overconfident_bucket(self) -> None:
        ece = expected_calibration_error(pd.Series([0.9, 0.9, 0.9]), pd.Series([0, 0, 0]), bins=2)

        self.assertGreater(ece, 0.8)

    def test_quality_gate_blocks_overconfident_bad_side(self) -> None:
        predictions = pd.DataFrame(
            [
                {
                    "side": "LONG",
                    "regime": "neutral",
                    "long_probability": 0.9,
                    "short_probability": 0.1,
                    "actual_long": 0,
                    "actual_short": 0,
                    "actual_label": "NO_TRADE",
                    "future_return": -0.01,
                    "reason": "positive_expectancy_risk_managed_setup",
                    "predicted_direction": "up",
                    "position_size": 100.0,
                    "executable": True,
                }
                for _ in range(5)
            ]
        )

        gated, metrics = apply_model_quality_gate(
            predictions,
            QualityPolicy(max_ece=0.1, min_edge_samples=3, min_regime_samples=3),
        )

        self.assertTrue((gated["side"] == "NO_TRADE").all())
        self.assertIn("CALIBRATION_DRIFT", gated["reason"].iloc[0])
        self.assertEqual(metrics["long_quality"], "blocked")

    def test_quality_gate_allows_positive_calibrated_side(self) -> None:
        rows = []
        for _ in range(4):
            rows.append(
                {
                    "side": "SHORT",
                    "regime": "bear",
                    "long_probability": 0.1,
                    "short_probability": 0.75,
                    "actual_long": 0,
                    "actual_short": 1,
                    "actual_label": "SHORT_SETUP",
                    "future_return": -0.02,
                    "reason": "positive_expectancy_risk_managed_setup",
                    "predicted_direction": "down",
                    "position_size": 100.0,
                    "executable": True,
                }
            )
        gated, metrics = apply_model_quality_gate(
            pd.DataFrame(rows),
            QualityPolicy(max_ece=0.30, min_edge_samples=3, min_regime_samples=3),
        )

        self.assertTrue((gated["side"] == "SHORT").all())
        self.assertEqual(metrics["short_quality"], "allowed")

    def test_quality_metrics_count_setup_candidates_even_when_side_is_no_trade(self) -> None:
        predictions = pd.DataFrame(
            [
                {
                    "side": "NO_TRADE",
                    "predicted_class": "LONG_SETUP",
                    "regime": "neutral",
                    "long_probability": 0.8,
                    "short_probability": 0.2,
                    "confidence_threshold": 0.72,
                    "actual_long": 1,
                    "actual_short": 0,
                    "actual_label": "LONG_SETUP",
                    "future_return": 0.02,
                    "reason": "LOW_ENTRY_QUALITY",
                }
                for _ in range(4)
            ]
        )

        _, metrics = apply_model_quality_gate(
            predictions,
            QualityPolicy(max_ece=0.30, min_edge_samples=3, min_regime_samples=3),
        )

        self.assertEqual(metrics["long_samples"], 4.0)
        self.assertEqual(metrics["long_quality"], "allowed")


if __name__ == "__main__":
    unittest.main()
