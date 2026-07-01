from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.utils.class_weight import compute_sample_weight

from scripts.common import LAB_CONFIG, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, ensure_dirs, market_symbols, read_parquet
from scripts.feature_engineering import feature_columns

LOGGER = logging.getLogger(__name__)
TARGET_THRESHOLDS = [0.005, 0.010, 0.015, 0.020]
CONFIDENCE_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
BALANCE_METHODS = ["none", "class_weight", "undersample", "oversample", "balanced_sample_weight"]
CALIBRATION_METHODS = ["none", "sigmoid", "isotonic"]


def threshold_suffix(threshold: float) -> str:
    return f"{int(threshold * 1000):03d}"


@dataclass
class SetupModelResult:
    model: Any
    features: list[str]
    horizon: str
    target_threshold: float
    balance_method: str
    calibration_method: str
    positive_rate: float


class LongSetupModel:
    def __init__(self, result: SetupModelResult):
        self.result = result

    def predict_probability(self, df: pd.DataFrame) -> np.ndarray:
        return _positive_probability(self.result.model, df[self.result.features].fillna(0))


class ShortSetupModel:
    def __init__(self, result: SetupModelResult):
        self.result = result

    def predict_probability(self, df: pd.DataFrame) -> np.ndarray:
        return _positive_probability(self.result.model, df[self.result.features].fillna(0))


def _make_base_model(balance_method: str) -> RandomForestClassifier:
    class_weight = "balanced" if balance_method == "class_weight" else None
    return RandomForestClassifier(
        n_estimators=80,
        max_depth=5,
        min_samples_leaf=5,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
    )


def _resample(x: pd.DataFrame, y: pd.Series, method: str) -> tuple[pd.DataFrame, pd.Series]:
    if method not in {"undersample", "oversample"} or y.nunique() < 2:
        return x, y
    rng = np.random.default_rng(42)
    positives = y[y == 1].index.to_numpy()
    negatives = y[y == 0].index.to_numpy()
    if len(positives) == 0 or len(negatives) == 0:
        return x, y
    if method == "undersample":
        keep_neg = rng.choice(negatives, size=min(len(negatives), len(positives) * 3), replace=False)
        keep = np.concatenate([positives, keep_neg])
    else:
        extra_pos = rng.choice(positives, size=max(0, len(negatives) - len(positives)), replace=True)
        keep = np.concatenate([negatives, positives, extra_pos])
    rng.shuffle(keep)
    return x.loc[keep], y.loc[keep]


def _fit_binary_model(
    x: pd.DataFrame,
    y: pd.Series,
    balance_method: str,
    calibration_method: str,
) -> Any:
    if y.nunique() < 2:
        raise ValueError("Binary setup target has fewer than two classes in training data.")
    x_fit, y_fit = _resample(x, y, balance_method)
    model = _make_base_model(balance_method)
    sample_weight = compute_sample_weight("balanced", y_fit) if balance_method == "balanced_sample_weight" else None
    if calibration_method in {"sigmoid", "isotonic"} and y_fit.value_counts().min() >= 3:
        calibrated = CalibratedClassifierCV(estimator=model, method=calibration_method, cv=3)
        calibrated.fit(x_fit, y_fit, sample_weight=sample_weight)
        return calibrated
    model.fit(x_fit, y_fit, sample_weight=sample_weight)
    return model


def _positive_probability(model: Any, x: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(x)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(x))
    return proba[:, classes.index(1)]


def _target_columns(horizon: str, target_threshold: float) -> tuple[str, str]:
    suffix = threshold_suffix(target_threshold)
    return f"long_target_{horizon}_{suffix}", f"short_target_{horizon}_{suffix}"


def _score_components(row: pd.Series) -> tuple[float, float, float]:
    regime_cols = [c for c in row.index if c.startswith("regime_")]
    regime_known = any(bool(row.get(c, False)) for c in regime_cols)
    regime_score = 0.10 if regime_known else -0.25
    drift_proxy = abs(float(row.get("volume_zscore", 0.0))) + abs(float(row.get("funding_zscore", 0.0)))
    feature_stability_score = 0.10 if drift_proxy < 4.0 else -0.25
    liquidity_score = 0.10 if float(row.get("vol_24h", 0.0)) > 0 else -0.25
    return regime_score, feature_stability_score, liquidity_score


def _decision(
    long_prob: float,
    short_prob: float,
    row: pd.Series,
    confidence_threshold: float,
) -> tuple[str, str, float, float, float, float]:
    regime_score, feature_stability_score, liquidity_score = _score_components(row)
    best_prob = max(long_prob, short_prob)
    signal_score = best_prob + regime_score + feature_stability_score + liquidity_score
    if regime_score < 0:
        return "NO_TRADE", "flat", best_prob, signal_score, regime_score, feature_stability_score
    if feature_stability_score < 0:
        return "NO_TRADE", "flat", best_prob, signal_score, regime_score, feature_stability_score
    if long_prob >= confidence_threshold and long_prob >= short_prob:
        return "LONG_SETUP", "up", long_prob, signal_score, regime_score, feature_stability_score
    if short_prob >= confidence_threshold:
        return "SHORT_SETUP", "down", short_prob, signal_score, regime_score, feature_stability_score
    return "NO_TRADE", "flat", best_prob, signal_score, regime_score, feature_stability_score


def walk_forward(
    df: pd.DataFrame,
    horizon: str = "4h",
    confidence_threshold: float = LAB_CONFIG.confidence_threshold,
    target_threshold: float = 0.010,
    feature_set: str = "full",
    balance_method: str = "class_weight",
    calibration_method: str = "sigmoid",
    folds: int = 4,
) -> tuple[pd.DataFrame, dict[str, float], SetupModelResult, SetupModelResult, list[str]]:
    df = df.reset_index(drop=True).copy()
    features = feature_columns(df, feature_set=feature_set)
    long_col, short_col = _target_columns(horizon, target_threshold)
    missing = [col for col in (long_col, short_col) if col not in df.columns]
    if missing:
        raise ValueError(f"Missing setup target columns: {missing}. Run features first.")
    if not features:
        raise ValueError(f"No features selected for feature_set={feature_set}")

    fold_size = max(1, len(df) // (folds + 1))
    rows: list[dict[str, Any]] = []
    accuracies: list[float] = []
    long_result: SetupModelResult | None = None
    short_result: SetupModelResult | None = None

    for fold in range(1, folds + 1):
        train_end = fold * fold_size
        test_start = min(train_end + LAB_CONFIG.embargo_rows, len(df))
        test_end = min(test_start + fold_size, len(df))
        if test_end <= test_start or train_end < 100:
            continue
        train_idx = df.index[:train_end]
        test_idx = df.index[test_start:test_end]
        x_train = df.loc[train_idx, features].fillna(0)
        x_test = df.loc[test_idx, features].fillna(0)
        y_long = df.loc[train_idx, long_col].astype(int)
        y_short = df.loc[train_idx, short_col].astype(int)
        if y_long.nunique() < 2 and y_short.nunique() < 2:
            LOGGER.warning("Skipping fold=%s because both setup targets have fewer than two classes.", fold)
            continue
        long_model = _fit_binary_model(x_train, y_long, balance_method, calibration_method) if y_long.nunique() >= 2 else None
        short_model = _fit_binary_model(x_train, y_short, balance_method, calibration_method) if y_short.nunique() >= 2 else None
        long_probs = _positive_probability(long_model, x_test) if long_model is not None else np.zeros(len(x_test))
        short_probs = _positive_probability(short_model, x_test) if short_model is not None else np.zeros(len(x_test))

        actual_long = df.loc[test_idx, long_col].astype(int).to_numpy()
        actual_short = df.loc[test_idx, short_col].astype(int).to_numpy()
        actual_any = ((actual_long == 1) | (actual_short == 1)).astype(int)
        predicted_any = ((long_probs >= confidence_threshold) | (short_probs >= confidence_threshold)).astype(int)
        accuracies.append(float(accuracy_score(actual_any, predicted_any)))

        if long_model is not None:
            long_result = SetupModelResult(long_model, features, horizon, target_threshold, balance_method, calibration_method, float(y_long.mean()))
        if short_model is not None:
            short_result = SetupModelResult(short_model, features, horizon, target_threshold, balance_method, calibration_method, float(y_short.mean()))

        for row_idx, long_prob, short_prob, a_long, a_short in zip(test_idx, long_probs, short_probs, actual_long, actual_short):
            source = df.loc[row_idx]
            predicted_class, direction, confidence, signal_score, regime_score, feature_stability_score = _decision(
                float(long_prob),
                float(short_prob),
                source,
                confidence_threshold,
            )
            rows.append(
                {
                    "timestamp": source["timestamp"],
                    "symbol": source["symbol"],
                    "predicted_class": predicted_class,
                    "predicted_label": predicted_class,
                    "predicted_direction": direction,
                    "probability": confidence,
                    "confidence": confidence,
                    "long_probability": float(long_prob),
                    "short_probability": float(short_prob),
                    "signal_score": float(signal_score),
                    "regime_score": float(regime_score),
                    "feature_stability_score": float(feature_stability_score),
                    "liquidity_score": 0.10 if float(source.get("vol_24h", 0.0)) > 0 else -0.25,
                    "close": float(source["close"]),
                    "future_return": float(source[f"future_return_{horizon}"]),
                    "future_max_up": float(source[f"future_max_up_{horizon}"]),
                    "future_max_down": float(source[f"future_max_down_{horizon}"]),
                    "actual_long": int(a_long),
                    "actual_short": int(a_short),
                    "actual_label": "LONG_SETUP" if a_long else ("SHORT_SETUP" if a_short else "NO_TRADE"),
                    "horizon": horizon,
                    "target_threshold": target_threshold,
                    "confidence_threshold": confidence_threshold,
                    "feature_set": feature_set,
                    "balance_method": balance_method,
                    "calibration_method": calibration_method,
                    "model_version": "dual_setup_v0.2",
                }
            )

    if long_result is None:
        long_result = SetupModelResult(_make_base_model(balance_method), features, horizon, target_threshold, balance_method, calibration_method, 0.0)
    if short_result is None:
        short_result = SetupModelResult(_make_base_model(balance_method), features, horizon, target_threshold, balance_method, calibration_method, 0.0)
    predictions = pd.DataFrame(rows)
    metrics = {
        "walk_forward_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "signals": float((predictions["predicted_direction"] != "flat").sum()) if not predictions.empty else 0.0,
        "long_opportunities": float((predictions["long_probability"] >= confidence_threshold).sum()) if not predictions.empty else 0.0,
        "short_opportunities": float((predictions["short_probability"] >= confidence_threshold).sum()) if not predictions.empty else 0.0,
        "always_no_trade": float((predictions["predicted_direction"] == "flat").mean()) if not predictions.empty else 1.0,
    }
    return predictions, metrics, long_result, short_result, features


def _ece(probabilities: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    data = pd.DataFrame({"prob": probabilities, "actual": actual}).dropna()
    if data.empty:
        return 0.0
    data["bucket"] = pd.cut(data["prob"], np.linspace(0, 1, bins + 1), include_lowest=True)
    error = 0.0
    for _, part in data.groupby("bucket", observed=False):
        if not part.empty:
            error += len(part) / len(data) * abs(part["prob"].mean() - part["actual"].mean())
    return float(error)


def save_calibration_report(predictions: pd.DataFrame) -> None:
    if predictions.empty:
        text = "# Calibration Report\n\nNo predictions available.\n"
        (REPORTS_DIR / "calibration_report.md").write_text(text, encoding="utf-8")
        return
    long_ece = _ece(predictions["long_probability"], predictions["actual_long"])
    short_ece = _ece(predictions["short_probability"], predictions["actual_short"])
    buckets = pd.cut(predictions["confidence"], np.linspace(0, 1, 11), include_lowest=True)
    curve = predictions.groupby(buckets, observed=False).agg(
        predictions_count=("confidence", "size"),
        avg_confidence=("confidence", "mean"),
        realized_setup_rate=("actual_label", lambda s: s.isin(["LONG_SETUP", "SHORT_SETUP"]).mean()),
    )
    text = f"""# Calibration Report

Calibration methods supported: `sigmoid` via `CalibratedClassifierCV` and `isotonic` when enough minority examples exist.

- Long setup ECE: {long_ece:.4f}
- Short setup ECE: {short_ece:.4f}

## Reliability Curve

{curve.to_string()}
"""
    (REPORTS_DIR / "calibration_report.md").write_text(text, encoding="utf-8")


def save_feature_importance(model_result: SetupModelResult, filename: str) -> None:
    model = model_result.model
    base_model = getattr(model, "estimator", None)
    if base_model is None:
        base_model = model
    if hasattr(base_model, "feature_importances_"):
        values = base_model.feature_importances_
    elif hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        estimator = model.calibrated_classifiers_[-1].estimator
        values = getattr(estimator, "feature_importances_", np.zeros(len(model_result.features)))
    else:
        values = np.zeros(len(model_result.features))
    pd.DataFrame({"feature": model_result.features, "importance": values}).sort_values("importance", ascending=False).to_csv(
        REPORTS_DIR / filename,
        index=False,
    )


def train(
    symbol: str | None = None,
    interval: str = LAB_CONFIG.timeframe,
    horizon: str = "4h",
    target_threshold: float = 0.010,
    confidence_threshold: float = LAB_CONFIG.confidence_threshold,
    balance_method: str = "class_weight",
    calibration_method: str = "sigmoid",
) -> pd.DataFrame:
    ensure_dirs()
    symbols = [symbol] if symbol else market_symbols()
    frames = [read_parquet(PROCESSED_DIR / f"{item}_{interval}_features.parquet") for item in symbols]
    df = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    outcome_cols = [f"future_return_{horizon}", f"future_max_up_{horizon}", f"future_max_down_{horizon}"]
    train_df = df.dropna(subset=[col for col in outcome_cols if col in df.columns]).reset_index(drop=True)
    predictions, metrics, long_result, short_result, _ = walk_forward(
        train_df,
        horizon=horizon,
        confidence_threshold=confidence_threshold,
        target_threshold=target_threshold,
        balance_method=balance_method,
        calibration_method=calibration_method,
    )
    live_df = df.copy()
    if not predictions.empty:
        live_df = live_df[pd.to_datetime(live_df["timestamp"], utc=True) > pd.to_datetime(predictions["timestamp"], utc=True).max()]
    live_rows: list[dict[str, Any]] = []
    if not live_df.empty:
        long_probs = LongSetupModel(long_result).predict_probability(live_df)
        short_probs = ShortSetupModel(short_result).predict_probability(live_df)
        for (_, source), long_prob, short_prob in zip(live_df.iterrows(), long_probs, short_probs):
            predicted_class, direction, confidence, signal_score, regime_score, feature_stability_score = _decision(
                float(long_prob),
                float(short_prob),
                source,
                confidence_threshold,
            )
            live_rows.append(
                {
                    "timestamp": source["timestamp"],
                    "symbol": source["symbol"],
                    "predicted_class": predicted_class,
                    "predicted_label": predicted_class,
                    "predicted_direction": direction,
                    "probability": confidence,
                    "confidence": confidence,
                    "long_probability": float(long_prob),
                    "short_probability": float(short_prob),
                    "signal_score": float(signal_score),
                    "regime_score": float(regime_score),
                    "feature_stability_score": float(feature_stability_score),
                    "liquidity_score": 0.10 if float(source.get("vol_24h", 0.0)) > 0 else -0.25,
                    "close": float(source["close"]),
                    "future_return": float(source.get(f"future_return_{horizon}", 0.0) or 0.0),
                    "future_max_up": float(source.get(f"future_max_up_{horizon}", 0.0) or 0.0),
                    "future_max_down": float(source.get(f"future_max_down_{horizon}", 0.0) or 0.0),
                    "actual_long": 0,
                    "actual_short": 0,
                    "actual_label": "UNKNOWN_LIVE",
                    "horizon": horizon,
                    "target_threshold": target_threshold,
                    "confidence_threshold": confidence_threshold,
                    "feature_set": "full",
                    "balance_method": balance_method,
                    "calibration_method": calibration_method,
                    "model_version": "dual_setup_v0.2",
                }
            )
    if live_rows:
        predictions = pd.concat([predictions, pd.DataFrame(live_rows)], ignore_index=True)
    joblib.dump(LongSetupModel(long_result), MODELS_DIR / "long_model.pkl")
    joblib.dump(ShortSetupModel(short_result), MODELS_DIR / "short_model.pkl")
    joblib.dump(
        {
            "long_model": long_result,
            "short_model": short_result,
            "metrics": metrics,
            "model_version": "dual_setup_v0.2",
        },
        MODELS_DIR / "dual_setup_v0.2.joblib",
    )
    predictions.to_parquet(PROCESSED_DIR / "predictions.parquet", index=False)
    pd.DataFrame([{**metrics, "horizon": horizon, "target_threshold": target_threshold, "balance_method": balance_method, "calibration_method": calibration_method}]).to_csv(
        REPORTS_DIR / "walk_forward_results.csv",
        index=False,
    )
    save_feature_importance(long_result, "long_feature_importance.csv")
    save_feature_importance(short_result, "short_feature_importance.csv")
    save_feature_importance(long_result, "feature_importance.csv")
    save_calibration_report(predictions)
    LOGGER.info("Saved dual setup models and predictions: %s", metrics)
    return predictions


if __name__ == "__main__":
    from scripts.common import setup_logging

    setup_logging()
    train()
