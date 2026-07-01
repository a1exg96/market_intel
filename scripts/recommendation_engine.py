from __future__ import annotations

import ast
import logging

import pandas as pd

from scripts.common import REPORTS_DIR, ensure_dirs

LOGGER = logging.getLogger(__name__)


def _safe_literal(value: object) -> dict:
    if isinstance(value, dict):
        return value
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def build_recommendations() -> str:
    ensure_dirs()
    diagnostics_path = REPORTS_DIR / "signal_diagnostics.csv"
    probability_path = REPORTS_DIR / "probability_buckets.csv"
    recommendations: list[str] = []

    if not diagnostics_path.exists():
        recommendations.append("Run `python scripts/main.py diagnostics` to generate signal-level diagnostics.")
    else:
        diag = pd.read_csv(diagnostics_path)
        if not diag.empty:
            row = diag.iloc[0]
            total = max(float(row.get("total_predictions", 0)), 1.0)
            rejected_conf = float(row.get("rejected_by_confidence", 0)) / total
            no_trade = float(row.get("no_trade_predictions", 0)) / total
            regime_blocked = float(row.get("rejected_by_regime_filter", 0)) / total
            if rejected_conf > 0.90:
                recommendations.append("More than 90% of predictions are rejected by confidence. Check calibration and run threshold sweeps; do not lower the threshold automatically.")
            if no_trade > 0.80:
                recommendations.append("More than 80% of predictions are `no_trade`. Review target strictness and test big-move thresholds of 1.0%, 1.5%, and 2.0% with walk-forward validation.")
            if regime_blocked > 0.70:
                recommendations.append("Regime filter blocks more than 70% of candidates. Review regime rules and test a soft filter instead of a hard filter.")
            if not recommendations:
                recommendations.append("No single rejection mechanism dominates. Continue with calibration, feature stability, and longer out-of-sample tests.")

    if probability_path.exists():
        buckets = pd.read_csv(probability_path)
        high = buckets[buckets["bucket"].astype(str).str.contains("0.8|0.9", regex=True, na=False)]
        high_with_low_winrate = high[(high["predictions_count"] > 0) & (high["actual_winrate"].fillna(1.0) < 0.50)]
        if not high_with_low_winrate.empty:
            recommendations.append("High-confidence buckets have low realized winrate. Run probability calibration, leakage/drift checks, and feature stability analysis.")

    report = "# Recommendations\n\n" + "\n".join(f"- {item}" for item in recommendations) + "\n"
    (REPORTS_DIR / "recommendations.md").write_text(report, encoding="utf-8")
    LOGGER.info("Saved recommendations count=%s", len(recommendations))
    return report


if __name__ == "__main__":
    from scripts.common import setup_logging

    setup_logging()
    print(build_recommendations())

