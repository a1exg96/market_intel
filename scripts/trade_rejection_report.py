from __future__ import annotations

import logging

import pandas as pd

from scripts.common import LAB_CONFIG, PROCESSED_DIR, REPORTS_DIR, ensure_dirs, read_parquet

LOGGER = logging.getLogger(__name__)

REJECTION_COLUMNS = [
    "timestamp",
    "symbol",
    "predicted_class",
    "confidence",
    "regime",
    "reason_rejected",
    "would_trade_without_filter",
    "future_return_if_available",
]


def _empty_report(reason: str) -> pd.DataFrame:
    ensure_dirs()
    df = pd.DataFrame(columns=REJECTION_COLUMNS)
    df.to_csv(REPORTS_DIR / "trade_rejections.csv", index=False)
    (REPORTS_DIR / "trade_rejection_report.md").write_text(
        f"# Trade Rejection Report\n\nInsufficient data: {reason}\n",
        encoding="utf-8",
    )
    return df


def _load_regimes() -> pd.DataFrame:
    path = PROCESSED_DIR / "regime_labels.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "symbol", "regime"])
    regimes = read_parquet(path)
    regimes["timestamp"] = pd.to_datetime(regimes["timestamp"], utc=True)
    return regimes[["timestamp", "symbol", "regime"]]


def rejection_reason(row: pd.Series, threshold: float = LAB_CONFIG.confidence_threshold) -> str:
    predicted_class = str(row.get("predicted_class", row.get("predicted_label", "")))
    confidence = float(row.get("confidence", 0.0))
    if pd.isna(row.get("future_return_if_available", row.get("future_return", 0.0))):
        return "DATA_MISSING"
    if predicted_class in {"NO_TRADE", "no_trade", "flat", ""}:
        return "PREDICTED_NO_TRADE"
    if confidence < threshold:
        return "LOW_CONFIDENCE"
    if str(row.get("regime", "")) in {"PANIC_BLOCKED"}:
        return "REGIME_BLOCKED"
    return "OTHER"


def build_trade_rejection_report(threshold: float = LAB_CONFIG.confidence_threshold) -> pd.DataFrame:
    ensure_dirs()
    predictions_path = PROCESSED_DIR / "predictions.parquet"
    if not predictions_path.exists():
        return _empty_report("predictions.parquet not found. Run train first.")

    predictions = read_parquet(predictions_path)
    if predictions.empty:
        return _empty_report("predictions.parquet is empty.")

    predictions = predictions.copy()
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], utc=True)
    predictions["predicted_class"] = predictions.get("predicted_label", predictions.get("predicted_direction", ""))
    regimes = _load_regimes()
    if not regimes.empty:
        predictions = pd.merge_asof(
            predictions.sort_values("timestamp"),
            regimes.sort_values("timestamp"),
            on="timestamp",
            by="symbol",
            direction="backward",
        )
    else:
        predictions["regime"] = "UNKNOWN"

    rows: list[dict[str, object]] = []
    for _, row in predictions.iterrows():
        predicted_class = str(row.get("predicted_class", ""))
        would_trade_without_filter = predicted_class in {"LONG_SETUP", "SHORT_SETUP", "big_move_up", "big_move_down", "up", "down"}
        reason = rejection_reason(row, threshold=threshold)
        rows.append(
            {
                "timestamp": row["timestamp"],
                "symbol": row.get("symbol", LAB_CONFIG.raw_symbol),
                "predicted_class": predicted_class,
                "confidence": float(row.get("confidence", 0.0)),
                "regime": row.get("regime", "UNKNOWN"),
                "reason_rejected": reason,
                "would_trade_without_filter": bool(would_trade_without_filter),
                "future_return_if_available": row.get("future_return", float("nan")),
            }
        )

    output = pd.DataFrame(rows, columns=REJECTION_COLUMNS)
    output.to_csv(REPORTS_DIR / "trade_rejections.csv", index=False)

    reason_counts = output["reason_rejected"].value_counts()
    executable = int(
        (
            (output["reason_rejected"] == "OTHER")
            & (output["would_trade_without_filter"].astype(bool))
            & (output["confidence"].astype(float) >= threshold)
        ).sum()
    )
    report = f"""# Trade Rejection Report

Total predictions reviewed: {len(output)}

Final executable trades: {executable}

## Rejection Reasons

{reason_counts.to_string()}

## Interpretation

Rows marked `PREDICTED_NO_TRADE` were not rejected by risk management; the model selected the no-trade class. Rows marked `LOW_CONFIDENCE` had a trade-like class but failed the configured confidence threshold of {threshold:.2f}. Rows marked `OTHER` with `would_trade_without_filter=true` are executable under the current diagnostic rules. No live trading is enabled.
"""
    (REPORTS_DIR / "trade_rejection_report.md").write_text(report, encoding="utf-8")
    LOGGER.info("Saved trade rejection report rows=%s executable=%s", len(output), executable)
    return output


if __name__ == "__main__":
    from scripts.common import setup_logging

    setup_logging()
    build_trade_rejection_report()
