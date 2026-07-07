from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from scripts.common import LAB_CONFIG, REPORTS_DIR, ensure_dirs

DEFAULT_MAX_ECE = 0.12
MIN_EDGE_SAMPLES = 30
MIN_EDGE_PRECISION = 0.52
MIN_REGIME_SAMPLES = 20
MIN_REGIME_PRECISION = 0.52


@dataclass(frozen=True)
class QualityPolicy:
    max_ece: float = DEFAULT_MAX_ECE
    min_edge_samples: int = MIN_EDGE_SAMPLES
    min_edge_precision: float = MIN_EDGE_PRECISION
    min_regime_samples: int = MIN_REGIME_SAMPLES
    min_regime_precision: float = MIN_REGIME_PRECISION


def expected_calibration_error(probabilities: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    data = pd.DataFrame({"prob": probabilities, "actual": actual}).dropna()
    if data.empty:
        return 0.0
    data["prob"] = pd.to_numeric(data["prob"], errors="coerce").clip(0, 1)
    data["actual"] = pd.to_numeric(data["actual"], errors="coerce").fillna(0).clip(0, 1)
    data = data.dropna()
    if data.empty:
        return 0.0
    data["bucket"] = pd.cut(data["prob"], np.linspace(0, 1, bins + 1), include_lowest=True)
    error = 0.0
    for _, part in data.groupby("bucket", observed=False):
        if not part.empty:
            error += len(part) / len(data) * abs(part["prob"].mean() - part["actual"].mean())
    return float(error)


def _validation_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    actual = predictions.get("actual_label", pd.Series([""] * len(predictions), index=predictions.index))
    return predictions[actual.astype(str) != "UNKNOWN_LIVE"].copy()


def _side_expectancy(selected: pd.DataFrame, side: str) -> float:
    if selected.empty or "future_return" not in selected:
        return 0.0
    raw_return = pd.to_numeric(selected["future_return"], errors="coerce")
    signed = raw_return if side == "LONG" else -raw_return
    signed = signed - LAB_CONFIG.fee_pct * 2 - LAB_CONFIG.slippage_pct
    clean = signed.dropna()
    return float(clean.mean()) if not clean.empty else 0.0


def _candidate_side(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=str)
    side = frame.get("side", pd.Series(["NO_TRADE"] * len(frame), index=frame.index)).astype(str).str.upper()
    predicted = frame.get("predicted_class", pd.Series([""] * len(frame), index=frame.index)).astype(str).str.upper()
    candidate = side.where(side.isin(["LONG", "SHORT"]), "")
    candidate = candidate.mask((candidate == "") & predicted.isin(["LONG_SETUP", "LONG"]), "LONG")
    candidate = candidate.mask((candidate == "") & predicted.isin(["SHORT_SETUP", "SHORT"]), "SHORT")
    if {"long_probability", "short_probability", "confidence_threshold"}.issubset(frame.columns):
        long_prob = pd.to_numeric(frame["long_probability"], errors="coerce").fillna(0.0)
        short_prob = pd.to_numeric(frame["short_probability"], errors="coerce").fillna(0.0)
        threshold = pd.to_numeric(frame["confidence_threshold"], errors="coerce").fillna(1.0)
        candidate = candidate.mask((candidate == "") & long_prob.ge(threshold) & long_prob.ge(short_prob), "LONG")
        candidate = candidate.mask((candidate == "") & short_prob.ge(threshold) & short_prob.gt(long_prob), "SHORT")
    return candidate.replace("", "NO_TRADE")


def _side_precision(selected: pd.DataFrame, side: str) -> float:
    actual_col = "actual_long" if side == "LONG" else "actual_short"
    if selected.empty or actual_col not in selected:
        return 0.0
    actual = pd.to_numeric(selected[actual_col], errors="coerce").fillna(0)
    return float(actual.eq(1).mean())


def _append_reason(existing: Any, reason: str) -> str:
    text = str(existing or "").strip()
    if not text:
        return reason
    if reason in text:
        return text
    return f"{text}; {reason}"


def build_model_quality_tables(predictions: pd.DataFrame, policy: QualityPolicy = QualityPolicy()) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation = _validation_rows(predictions)
    side_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    side_columns = ["side", "samples", "precision", "expectancy", "ece", "status", "reasons"]
    regime_columns = ["side", "regime", "samples", "precision", "expectancy", "status", "reasons"]
    for side, prob_col, actual_col in (
        ("LONG", "long_probability", "actual_long"),
        ("SHORT", "short_probability", "actual_short"),
    ):
        candidate_side = _candidate_side(validation)
        selected = validation[candidate_side == side].copy()
        ece = expected_calibration_error(validation.get(prob_col, pd.Series(dtype=float)), validation.get(actual_col, pd.Series(dtype=float)))
        samples = int(len(selected))
        precision = _side_precision(selected, side)
        expectancy = _side_expectancy(selected, side)
        side_status = "allowed"
        reasons: list[str] = []
        if ece > policy.max_ece:
            reasons.append("CALIBRATION_DRIFT")
        if samples < policy.min_edge_samples:
            reasons.append("INSUFFICIENT_EDGE_SAMPLES")
        if precision < policy.min_edge_precision:
            reasons.append("LOW_EDGE_PRECISION")
        if expectancy <= 0:
            reasons.append("NON_POSITIVE_EDGE_EXPECTANCY")
        if reasons:
            side_status = "blocked"
        side_rows.append(
            {
                "side": side,
                "samples": samples,
                "precision": precision,
                "expectancy": expectancy,
                "ece": ece,
                "status": side_status,
                "reasons": ";".join(reasons),
            }
        )
        if "regime" in validation:
            for regime, part in selected.groupby(selected["regime"].fillna("UNKNOWN").astype(str), dropna=False):
                regime_samples = int(len(part))
                regime_precision = _side_precision(part, side)
                regime_expectancy = _side_expectancy(part, side)
                regime_reasons: list[str] = []
                if regime_samples < policy.min_regime_samples:
                    regime_reasons.append("INSUFFICIENT_REGIME_SAMPLES")
                if regime_precision < policy.min_regime_precision:
                    regime_reasons.append("LOW_REGIME_PRECISION")
                if regime_expectancy <= 0:
                    regime_reasons.append("NON_POSITIVE_REGIME_EXPECTANCY")
                regime_rows.append(
                    {
                        "side": side,
                        "regime": regime,
                        "samples": regime_samples,
                        "precision": regime_precision,
                        "expectancy": regime_expectancy,
                        "status": "blocked" if regime_reasons else "allowed",
                        "reasons": ";".join(regime_reasons),
                    }
                )
    return pd.DataFrame(side_rows, columns=side_columns), pd.DataFrame(regime_rows, columns=regime_columns)


def apply_model_quality_gate(predictions: pd.DataFrame, policy: QualityPolicy = QualityPolicy()) -> tuple[pd.DataFrame, dict[str, float | str]]:
    if predictions.empty or "side" not in predictions:
        return predictions, {}
    out = predictions.copy()
    side_quality, regime_quality = build_model_quality_tables(out, policy)
    blocked_sides = set(side_quality.loc[side_quality["status"] == "blocked", "side"].astype(str)) if not side_quality.empty else set()
    blocked_regimes = {
        (str(row["side"]), str(row["regime"])): str(row["reasons"])
        for _, row in regime_quality.iterrows()
        if str(row.get("status")) == "blocked"
    }

    for idx, row in out.iterrows():
        side = str(row.get("side", ""))
        if side not in {"LONG", "SHORT"}:
            continue
        reasons: list[str] = []
        if side in blocked_sides:
            side_reason = side_quality.loc[side_quality["side"].astype(str) == side, "reasons"]
            reasons.extend(str(side_reason.iloc[0]).split(";") if not side_reason.empty else ["MODEL_QUALITY_BLOCKED"])
        regime = str(row.get("regime", "UNKNOWN"))
        regime_reason = blocked_regimes.get((side, regime))
        if regime_reason:
            reasons.extend(regime_reason.split(";"))
        reasons = [reason for reason in reasons if reason]
        if reasons:
            out.at[idx, "reason"] = _append_reason(row.get("reason", ""), "; ".join(dict.fromkeys(reasons)))
            out.at[idx, "side"] = "NO_TRADE"
            out.at[idx, "predicted_direction"] = "flat"
            out.at[idx, "position_size"] = 0.0
            out.at[idx, "executable"] = False

    metrics: dict[str, float | str] = {}
    for _, row in side_quality.iterrows():
        prefix = str(row["side"]).lower()
        metrics[f"{prefix}_samples"] = float(row["samples"])
        metrics[f"{prefix}_precision"] = float(row["precision"])
        metrics[f"{prefix}_expectancy"] = float(row["expectancy"])
        metrics[f"{prefix}_ece"] = float(row["ece"])
        metrics[f"{prefix}_quality"] = str(row["status"])
        metrics[f"{prefix}_quality_reasons"] = str(row["reasons"])
    save_model_quality_report(side_quality, regime_quality)
    return out, metrics


def save_model_quality_report(side_quality: pd.DataFrame, regime_quality: pd.DataFrame) -> None:
    ensure_dirs()
    side_quality.to_csv(REPORTS_DIR / "model_quality.csv", index=False)
    regime_quality.to_csv(REPORTS_DIR / "regime_quality.csv", index=False)
    lines = ["# Model Quality Gate", ""]
    lines.append("## Side Quality")
    lines.append(side_quality.to_string(index=False) if not side_quality.empty else "No side quality rows.")
    lines.append("")
    lines.append("## Regime Quality")
    lines.append(regime_quality.to_string(index=False) if not regime_quality.empty else "No regime quality rows.")
    lines.append("")
    (REPORTS_DIR / "model_quality.md").write_text("\n".join(lines), encoding="utf-8")
