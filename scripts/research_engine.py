from __future__ import annotations

import logging
from dataclasses import asdict

import pandas as pd

from scripts.baseline_ml import walk_forward
from scripts.common import LAB_CONFIG, PROCESSED_DIR, REPORTS_DIR, ensure_dirs, read_parquet
from scripts.paper_trader import compute_stats, simulate

LOGGER = logging.getLogger(__name__)
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
HORIZONS = ["1h", "4h", "24h"]
FEATURE_SETS = ["basic", "futures", "full"]
MODEL_PARAM_GRID = [
    {"balance_method": "none", "calibration_method": "none"},
    {"balance_method": "class_weight", "calibration_method": "sigmoid"},
    {"balance_method": "undersample", "calibration_method": "sigmoid"},
    {"balance_method": "oversample", "calibration_method": "sigmoid"},
    {"balance_method": "balanced_sample_weight", "calibration_method": "isotonic"},
]
TARGET_THRESHOLDS = [0.005, 0.010, 0.015, 0.020]


def stability_score(row: pd.Series) -> float:
    return float(row["sharpe_like"] + row["sortino_like"] + row["profit_factor"] * 0.2 - abs(row["max_drawdown"]) * 3)


def run_research() -> pd.DataFrame:
    ensure_dirs()
    df = read_parquet(PROCESSED_DIR / f"{LAB_CONFIG.raw_symbol}_{LAB_CONFIG.timeframe}_features.parquet")
    rows: list[dict] = []
    for horizon in HORIZONS:
        for target_threshold in TARGET_THRESHOLDS:
            for feature_set in FEATURE_SETS:
                for params in MODEL_PARAM_GRID:
                    try:
                        base_predictions, base_metrics, _, _, _ = walk_forward(
                            df,
                            horizon=horizon,
                            confidence_threshold=0.0,
                            target_threshold=target_threshold,
                            feature_set=feature_set,
                            balance_method=params["balance_method"],
                            calibration_method=params["calibration_method"],
                            folds=3,
                        )
                    except Exception as exc:
                        LOGGER.warning(
                            "Research training failed horizon=%s target=%s feature_set=%s balance=%s calibration=%s error=%s",
                            horizon,
                            target_threshold,
                            feature_set,
                            params["balance_method"],
                            params["calibration_method"],
                            exc,
                        )
                        continue
                    for threshold in THRESHOLDS:
                        try:
                            predictions = _apply_confidence_threshold(base_predictions, threshold)
                            trades = simulate(predictions)
                            stats = compute_stats(trades)
                            row = {
                                "horizon": horizon,
                                "target_threshold": target_threshold,
                                "confidence_threshold": threshold,
                                "feature_set": feature_set,
                                "balance_method": params["balance_method"],
                                "calibration_method": params["calibration_method"],
                                **base_metrics,
                                **asdict(stats),
                            }
                            row["passes_success_criteria"] = bool(
                                row["profit_factor"] > 1.2
                                and row["avg_return_per_trade"] > 0
                                and row["max_drawdown"] > -0.15
                                and row["number_of_trades"] > 20
                                and row["always_no_trade"] < 0.95
                            )
                            row["stability_score"] = stability_score(pd.Series(row))
                            rows.append(row)
                        except Exception as exc:
                            LOGGER.warning(
                                "Research threshold failed horizon=%s target=%s threshold=%s feature_set=%s balance=%s calibration=%s error=%s",
                                horizon,
                                target_threshold,
                                threshold,
                                feature_set,
                                params["balance_method"],
                                params["calibration_method"],
                                exc,
                            )
    result = pd.DataFrame(rows).sort_values("stability_score", ascending=False) if rows else pd.DataFrame()
    result.to_csv(REPORTS_DIR / "research_results.csv", index=False)
    LOGGER.info("Saved research results rows=%s", len(result))
    return result


def _apply_confidence_threshold(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = predictions.copy()
    directions: list[str] = []
    classes: list[str] = []
    confidences: list[float] = []
    for _, row in out.iterrows():
        if str(row.get("side", "")).upper() == "NO_TRADE" and "INSUFFICIENT_EMPIRICAL_EDGE" in str(row.get("reason", "")):
            directions.append("flat")
            classes.append("NO_TRADE")
            confidences.append(float(max(row.get("long_probability", 0.0), row.get("short_probability", 0.0))))
            continue
        long_prob = float(row.get("long_probability", 0.0))
        short_prob = float(row.get("short_probability", 0.0))
        if long_prob >= threshold and long_prob >= short_prob:
            directions.append("up")
            classes.append("LONG_SETUP")
            confidences.append(long_prob)
        elif short_prob >= threshold:
            directions.append("down")
            classes.append("SHORT_SETUP")
            confidences.append(short_prob)
        else:
            directions.append("flat")
            classes.append("NO_TRADE")
            confidences.append(max(long_prob, short_prob))
    out["predicted_direction"] = directions
    out["predicted_class"] = classes
    out["predicted_label"] = classes
    out["confidence"] = confidences
    out["probability"] = confidences
    out["confidence_threshold"] = threshold
    return out


if __name__ == "__main__":
    from scripts.common import setup_logging

    setup_logging()
    run_research()
