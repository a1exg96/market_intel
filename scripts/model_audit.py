from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from scripts.common import LAB_CONFIG, PROCESSED_DIR, REPORTS_DIR, ensure_dirs, read_parquet
from scripts.feature_engineering import feature_columns
from scripts.regime_engine import build_regime_labels
from scripts.baseline_ml import threshold_suffix

LOGGER = logging.getLogger(__name__)


@dataclass
class AuditResult:
    has_leakage_risk: bool
    has_overfit_risk: bool
    has_calibration_risk: bool
    has_drift_risk: bool
    has_class_imbalance: bool
    tradable: bool


def _psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = expected.replace([np.inf, -np.inf], np.nan).dropna()
    actual = actual.replace([np.inf, -np.inf], np.nan).dropna()
    if expected.empty or actual.empty or expected.nunique() <= 1:
        return 0.0
    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3:
        return 0.0
    expected_counts = pd.cut(expected, cuts, include_lowest=True).value_counts(normalize=True, sort=False)
    actual_counts = pd.cut(actual, cuts, include_lowest=True).value_counts(normalize=True, sort=False)
    expected_pct = expected_counts.replace(0, 1e-6)
    actual_pct = actual_counts.replace(0, 1e-6)
    return float(((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)).sum())


def _expected_calibration_error(confidence: pd.Series, correct: pd.Series, bins: int = 10) -> float:
    data = pd.DataFrame({"confidence": confidence, "correct": correct}).dropna()
    if data.empty:
        return 0.0
    data["bucket"] = pd.cut(data["confidence"], np.linspace(0, 1, bins + 1), include_lowest=True)
    ece = 0.0
    for _, bucket in data.groupby("bucket", observed=False):
        if bucket.empty:
            continue
        ece += len(bucket) / len(data) * abs(bucket["confidence"].mean() - bucket["correct"].mean())
    return float(ece)


def _low_information_features(df: pd.DataFrame, features: list[str], target: pd.Series) -> pd.DataFrame:
    if not features:
        return pd.DataFrame(columns=["feature", "mutual_info", "variance"])
    x = df[features].fillna(0)
    y = LabelEncoder().fit_transform(target.astype(str))
    try:
        mi = mutual_info_classif(x, y, random_state=42, discrete_features="auto")
    except Exception as exc:
        LOGGER.warning("Mutual information failed: %s", exc)
        mi = np.zeros(len(features))
    out = pd.DataFrame({"feature": features, "mutual_info": mi, "variance": x.var(numeric_only=True).reindex(features).fillna(0).to_numpy()})
    return out.sort_values(["mutual_info", "variance"], ascending=True)


def _correlation_filter(df: pd.DataFrame, features: list[str], threshold: float = 0.95) -> list[str]:
    if len(features) < 2:
        return []
    corr = df[features].corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return [column for column in upper.columns if any(upper[column] > threshold)]


def _in_sample_vs_holdout(df: pd.DataFrame, features: list[str], target: pd.Series) -> tuple[float, float, float]:
    if len(df) < 150 or target.nunique() < 2:
        return 0.0, 0.0, 0.0
    x_train, x_test, y_train, y_test = train_test_split(df[features].fillna(0), target.astype(str), test_size=0.35, shuffle=False)
    model = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.05, random_state=42)
    model.fit(x_train, y_train)
    train_acc = float(accuracy_score(y_train, model.predict(x_train)))
    test_pred = model.predict(x_test)
    test_acc = float(accuracy_score(y_test, test_pred))
    try:
        test_log_loss = float(log_loss(y_test, model.predict_proba(x_test), labels=model.classes_))
    except Exception:
        test_log_loss = float("nan")
    return train_acc, test_acc, test_log_loss


def run_model_audit(symbol: str = LAB_CONFIG.raw_symbol, interval: str = LAB_CONFIG.timeframe, horizon: str = "4h") -> AuditResult:
    ensure_dirs()
    features_path = PROCESSED_DIR / f"{symbol}_{interval}_features.parquet"
    df = read_parquet(features_path)
    suffix = threshold_suffix(0.010)
    long_col = f"long_target_{horizon}_{suffix}"
    short_col = f"short_target_{horizon}_{suffix}"
    if long_col not in df.columns or short_col not in df.columns:
        raise ValueError(f"Missing setup target columns {long_col}/{short_col}. Run scripts/main.py features first.")

    features = feature_columns(df, feature_set="full")
    target = pd.Series(
        np.select(
            [df[long_col].astype(int) == 1, df[short_col].astype(int) == 1],
            ["LONG_SETUP", "SHORT_SETUP"],
            default="NO_SETUP",
        ),
        index=df.index,
    )
    class_counts = target.value_counts()
    class_share = (class_counts / len(target)).sort_values(ascending=False)
    has_class_imbalance = bool(class_share.iloc[0] > 0.70) if not class_share.empty else True

    leakage_name_hits = [col for col in df.columns if any(token in col.lower() for token in ["future", "target", "direction", "max_up", "max_down"]) and col in features]
    suspicious_corr: list[str] = []
    y_binary = target.map({"LONG_SETUP": 1, "SHORT_SETUP": -1, "NO_SETUP": 0}).astype(float)
    for col in features:
        if df[col].nunique(dropna=True) <= 1:
            continue
        corr = abs(float(pd.Series(df[col]).corr(y_binary)))
        if not np.isnan(corr) and corr > 0.60:
            suspicious_corr.append(f"{col} corr={corr:.3f}")

    info = _low_information_features(df, features, target)
    low_info = info[(info["mutual_info"] <= 0.001) | (info["variance"] <= 1e-12)].head(20)
    correlated_drop = _correlation_filter(df, features)
    train_acc, holdout_acc, holdout_log_loss = _in_sample_vs_holdout(df, features, target)
    has_overfit_risk = bool(train_acc - holdout_acc > 0.20 and train_acc > 0.65)

    drift_rows: list[dict[str, float | str]] = []
    split = max(1, len(df) // 2)
    first = df.iloc[:split]
    second = df.iloc[split:]
    for col in features:
        value = _psi(first[col], second[col])
        if value > 0.20:
            drift_rows.append({"feature": col, "psi": value})
    drift_df = pd.DataFrame(drift_rows).sort_values("psi", ascending=False) if drift_rows else pd.DataFrame(columns=["feature", "psi"])
    has_drift_risk = not drift_df.empty

    try:
        regimes = build_regime_labels(symbol=symbol, interval=interval)
        regime_distribution = regimes["regime"].value_counts(normalize=True)
        regime_drift_note = regime_distribution.to_string()
    except Exception as exc:
        regime_drift_note = f"Regime engine unavailable: {exc}"

    predictions_path = PROCESSED_DIR / "predictions.parquet"
    calibration_text = "No predictions.parquet found. Run train before probability calibration audit."
    has_calibration_risk = True
    ece = None
    if predictions_path.exists():
        predictions = read_parquet(predictions_path)
        if {"confidence", "predicted_label", "actual_label"}.issubset(predictions.columns):
            valid = predictions[predictions["actual_label"].astype(str) != ""].copy()
            if not valid.empty:
                correct = valid["predicted_label"].astype(str) == valid["actual_label"].astype(str)
                ece = _expected_calibration_error(valid["confidence"].astype(float), correct.astype(float))
                high_conf = valid[valid["confidence"].astype(float) >= 0.70]
                high_conf_acc = float((high_conf["predicted_label"].astype(str) == high_conf["actual_label"].astype(str)).mean()) if not high_conf.empty else 0.0
                calibration_text = f"ECE={ece:.3f}; high-confidence accuracy={high_conf_acc:.3f}; samples={len(valid)}"
                has_calibration_risk = bool(ece > 0.10 or high_conf_acc < 0.50)
        else:
            calibration_text = "Existing predictions are from the old schema; retrain to audit calibrated probabilities."

    leakage_risk = bool(leakage_name_hits or suspicious_corr)
    tradable = not (leakage_risk or has_overfit_risk or has_calibration_risk or has_drift_risk or has_class_imbalance)
    result = AuditResult(
        has_leakage_risk=leakage_risk,
        has_overfit_risk=has_overfit_risk,
        has_calibration_risk=has_calibration_risk,
        has_drift_risk=has_drift_risk,
        has_class_imbalance=has_class_imbalance,
        tradable=tradable,
    )

    report = f"""# Model Audit

Generated for `{symbol}` `{interval}` horizon `{horizon}`.

## Verdict

{"TRADEABLE" if result.tradable else "NO EDGE CONFIRMED. DO NOT TRADE."}

This audit is diagnostic. It does not tune historical results upward.

## Leakage And Look-Ahead

- Feature columns selected: {len(features)}
- Forbidden/future columns inside selected features: {leakage_name_hits or "none"}
- Suspicious feature-target correlations: {suspicious_corr[:10] or "none above threshold"}
- Target columns are present only as labels and blocked by `feature_columns`.

## Class Imbalance

{class_counts.to_string()}

Dominant class share:

{class_share.to_string()}

Class imbalance risk: {has_class_imbalance}

## Low Information Features

{low_info.to_string(index=False) if not low_info.empty else "No near-zero information features detected by this simple pass."}

## Correlation Filter Candidates

{", ".join(correlated_drop[:30]) if correlated_drop else "No feature pairs above correlation threshold."}

## Overfitting Check

- Train accuracy: {train_acc:.3f}
- Forward holdout accuracy: {holdout_acc:.3f}
- Forward holdout log loss: {holdout_log_loss:.3f}
- Overfit risk: {has_overfit_risk}

## Probability Calibration

{calibration_text}

Calibration risk: {has_calibration_risk}

## Distribution Drift

Features with PSI > 0.20:

{drift_df.head(20).to_string(index=False) if not drift_df.empty else "None detected."}

Distribution drift risk: {has_drift_risk}

## Regime Drift

{regime_drift_note}

## Survivorship Bias

The current v0.1 universe contains only BTCUSDT. Survivorship bias is limited by the single-instrument scope, but conclusions do not generalize to a multi-asset crypto universe.

## Success Criteria

- Profit Factor > 1.2: checked in paper/research reports
- Average Return > 0: checked in paper/research reports
- Max Drawdown < 15%: checked in paper/research reports
- Stable between folds: partially checked by walk-forward; needs expanded fold report
- Enough trades: checked in paper/research reports
- No leakage: {not leakage_risk}

Final decision: {"model may proceed to further paper research" if result.tradable else "advantage is not proven; default action is no trade"}
"""
    (REPORTS_DIR / "model_audit.md").write_text(report, encoding="utf-8")
    info.to_csv(REPORTS_DIR / "feature_information.csv", index=False)
    drift_df.to_csv(REPORTS_DIR / "distribution_drift.csv", index=False)
    LOGGER.info("Saved model audit. tradable=%s", result.tradable)
    return result


if __name__ == "__main__":
    from scripts.common import setup_logging

    setup_logging()
    run_model_audit()
