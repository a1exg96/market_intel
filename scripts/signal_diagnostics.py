from __future__ import annotations

import logging

import pandas as pd

from scripts.common import LAB_CONFIG, PROCESSED_DIR, REPORTS_DIR, ensure_dirs, read_parquet
from scripts.trade_rejection_report import build_trade_rejection_report

LOGGER = logging.getLogger(__name__)


def _empty_diagnostics(reason: str) -> pd.DataFrame:
    ensure_dirs()
    row = {
        "total_predictions": 0,
        "signals_above_threshold": 0,
        "signals_below_threshold": 0,
        "no_trade_predictions": 0,
        "rejected_by_confidence": 0,
        "rejected_by_regime_filter": 0,
        "rejected_by_risk_engine": 0,
        "rejected_by_adaptation_engine": 0,
        "rejected_by_privacy_or_data_error": 0,
        "final_executable_trades": 0,
        "status": f"insufficient data: {reason}",
    }
    df = pd.DataFrame([row])
    df.to_csv(REPORTS_DIR / "signal_diagnostics.csv", index=False)
    (REPORTS_DIR / "signal_diagnostics.md").write_text(f"# Signal Diagnostics\n\nInsufficient data: {reason}\n", encoding="utf-8")
    return df


def run_signal_diagnostics(threshold: float = LAB_CONFIG.confidence_threshold) -> pd.DataFrame:
    ensure_dirs()
    predictions_path = PROCESSED_DIR / "predictions.parquet"
    if not predictions_path.exists():
        return _empty_diagnostics("predictions.parquet not found. Run train first.")

    predictions = read_parquet(predictions_path)
    if predictions.empty:
        return _empty_diagnostics("predictions.parquet is empty.")

    predictions = predictions.copy()
    predicted_class = predictions.get("predicted_label", predictions.get("predicted_direction", pd.Series(dtype=str))).astype(str)
    actual_class = predictions.get("actual_label", pd.Series([""] * len(predictions))).astype(str)
    confidence = predictions["confidence"].astype(float) if "confidence" in predictions else pd.Series([0.0] * len(predictions))
    trade_like = predicted_class.isin(["LONG_SETUP", "SHORT_SETUP", "big_move_up", "big_move_down", "up", "down"])
    no_trade = predicted_class.isin(["NO_TRADE", "no_trade", "flat", ""])
    above_threshold = trade_like & (confidence >= threshold)
    below_threshold = trade_like & (confidence < threshold)

    rejections = build_trade_rejection_report(threshold=threshold)
    reason_counts = rejections["reason_rejected"].value_counts() if not rejections.empty else pd.Series(dtype=int)
    final_executable = 0
    if not rejections.empty:
        final_executable = int(
            (
                (rejections["reason_rejected"] == "OTHER")
                & (rejections["would_trade_without_filter"].astype(bool))
                & (rejections["confidence"].astype(float) >= threshold)
            ).sum()
        )
    rejected_by_confidence = int(reason_counts.get("LOW_CONFIDENCE", 0))
    rejected_by_regime = int(reason_counts.get("REGIME_BLOCKED", 0))
    rejected_by_data = int(reason_counts.get("DATA_MISSING", 0) + reason_counts.get("PRIVACY_BLOCKED", 0))

    row = {
        "total_predictions": int(len(predictions)),
        "class_distribution_predicted": predicted_class.value_counts().to_dict(),
        "class_distribution_actual": actual_class.value_counts().to_dict() if actual_class.ne("").any() else {},
        "confidence_distribution": confidence.describe().to_dict(),
        "signals_above_threshold": int(above_threshold.sum()),
        "signals_below_threshold": int(below_threshold.sum()),
        "long_opportunities_found": int((predictions.get("long_probability", pd.Series([0] * len(predictions))).astype(float) >= threshold).sum()),
        "short_opportunities_found": int((predictions.get("short_probability", pd.Series([0] * len(predictions))).astype(float) >= threshold).sum()),
        "no_trade_predictions": int(no_trade.sum()),
        "rejected_by_confidence": rejected_by_confidence,
        "rejected_by_regime_filter": rejected_by_regime,
        "rejected_by_risk_engine": 0,
        "rejected_by_adaptation_engine": 0,
        "rejected_by_privacy_or_data_error": rejected_by_data,
        "final_executable_trades": final_executable,
        "status": "ok",
    }
    output = pd.DataFrame([row])
    output.to_csv(REPORTS_DIR / "signal_diagnostics.csv", index=False)

    no_trade_share = row["no_trade_predictions"] / max(row["total_predictions"], 1)
    below_share = row["rejected_by_confidence"] / max(row["total_predictions"], 1)
    threshold_problem = "yes" if row["signals_below_threshold"] > row["signals_above_threshold"] and row["signals_below_threshold"] > 0 else "no"
    target_problem = "likely" if no_trade_share > 0.80 else "not obvious"
    model_confidence_problem = "likely" if row["signals_above_threshold"] == 0 and not no_trade_share > 0.80 else "not primary"
    most_common_rejection = reason_counts.index[0] if not reason_counts.empty else "none"

    report = f"""# Signal Diagnostics

Total predictions: {row["total_predictions"]}

Final executable trades: {final_executable}

## Prediction Classes

Predicted:

{predicted_class.value_counts().to_string()}

Actual:

{actual_class.value_counts().to_string()}

## Threshold And Rejections

- Configured confidence threshold: {threshold:.2f}
- Trade-like predictions above threshold: {row["signals_above_threshold"]}
- Trade-like predictions below threshold: {row["signals_below_threshold"]}
- No-trade predictions: {row["no_trade_predictions"]} ({no_trade_share:.1%})
- Rejected by confidence: {row["rejected_by_confidence"]} ({below_share:.1%})
- Rejected by regime filter: {row["rejected_by_regime_filter"]}
- Rejected by risk engine: {row["rejected_by_risk_engine"]}
- Rejected by adaptation engine: {row["rejected_by_adaptation_engine"]}
- Rejected by privacy/data error: {row["rejected_by_privacy_or_data_error"]}
- Most common rejection reason: {most_common_rejection}

## Human Diagnosis

- Threshold problem: {threshold_problem}
- Target/no-trade class problem: {target_problem}
- Regime filter problem: {"yes" if row["rejected_by_regime_filter"] > 0 else "no"}
- Risk/adaptation engine problem: no
- Model confidence problem: {model_confidence_problem}

No trades were executed because {no_trade_share:.1%} of execution decisions were no-trade and {row["signals_above_threshold"]} setup predictions exceeded the confidence threshold. The correct next step is to inspect target strictness, balancing, and probability calibration, not to lower the threshold automatically.
"""
    (REPORTS_DIR / "signal_diagnostics.md").write_text(report, encoding="utf-8")
    LOGGER.info("Saved signal diagnostics total=%s executable=%s", len(predictions), final_executable)
    return output


if __name__ == "__main__":
    from scripts.common import setup_logging

    setup_logging()
    run_signal_diagnostics()
