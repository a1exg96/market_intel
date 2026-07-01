from __future__ import annotations

import csv
import logging
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.common import LAB_CONFIG, REPORTS_DIR, ensure_dirs, setup_logging
from scripts.paper_execution import read_active_positions, read_audit
from scripts.privacy_audit import checked_request

LOGGER = logging.getLogger(__name__)
RUNNING = True

NOTIFICATION_COLUMNS = ["timestamp", "position_id", "symbol", "signal", "confidence", "chat_id", "status", "error"]
NOTIFICATIONS_PATH = REPORTS_DIR / "telegram_notifications.csv"


def _stop(_: int, __: object) -> None:
    global RUNNING
    RUNNING = False


def _enabled() -> bool:
    return os.getenv("TELEGRAM_NOTIFICATIONS_ENABLED", "false").lower() in {"1", "true", "yes", "on"}


def _min_confidence() -> float:
    try:
        return float(os.getenv("TELEGRAM_MIN_CONFIDENCE", str(LAB_CONFIG.confidence_threshold)))
    except ValueError:
        return float(LAB_CONFIG.confidence_threshold)


def _ensure_notifications_file() -> None:
    ensure_dirs()
    if not NOTIFICATIONS_PATH.exists() or NOTIFICATIONS_PATH.stat().st_size == 0:
        pd.DataFrame(columns=NOTIFICATION_COLUMNS).to_csv(NOTIFICATIONS_PATH, index=False)


def read_notifications() -> pd.DataFrame:
    _ensure_notifications_file()
    frame = pd.read_csv(NOTIFICATIONS_PATH)
    for column in NOTIFICATION_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame[NOTIFICATION_COLUMNS]


def _append_notification(row: dict[str, Any]) -> None:
    _ensure_notifications_file()
    new_file = not NOTIFICATIONS_PATH.exists() or NOTIFICATIONS_PATH.stat().st_size == 0
    with NOTIFICATIONS_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=NOTIFICATION_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in NOTIFICATION_COLUMNS})


def _sent_position_ids() -> set[str]:
    frame = read_notifications()
    if frame.empty:
        return set()
    sent = frame[frame["status"].astype(str).str.upper() == "SENT"]
    return set(sent["position_id"].astype(str))


def _position_by_id() -> dict[str, dict[str, Any]]:
    positions = read_active_positions(open_only=False)
    if positions.empty:
        return {}
    clean = positions.where(pd.notna(positions), None)
    return {str(row["position_id"]): row.to_dict() for _, row in clean.iterrows()}


def _format_price(value: Any) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "-"


def _format_confidence(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "-"


def _message(audit_row: dict[str, Any], position: dict[str, Any] | None) -> str:
    side = str(audit_row.get("signal", "")).upper()
    symbol = str(audit_row.get("symbol", ""))
    confidence = _format_confidence(audit_row.get("confidence"))
    price = _format_price(position.get("entry_price") if position else audit_row.get("price"))
    position_size = _format_price(position.get("position_size") if position else "")
    regime = str((position or {}).get("regime") or "UNKNOWN")
    model_version = str((position or {}).get("model_version") or "")
    position_id = str(audit_row.get("position_id", ""))

    return (
        "Сильний paper-сигнал\n"
        f"Пара: {symbol}\n"
        f"Напрям: {side}\n"
        f"Впевненість: {confidence}\n"
        f"Ціна входу: {price}\n"
        f"Розмір позиції: {position_size} USD\n"
        f"Режим: {regime}\n"
        f"Модель: {model_version}\n"
        f"Position ID: {position_id}\n"
        "Live trading не виконується."
    )


def send_telegram_message(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set")
    response = checked_request(
        "POST",
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
        timeout=10,
    )
    response.raise_for_status()


def notify_new_strong_signals() -> int:
    if not _enabled():
        LOGGER.info("Telegram notifications are disabled.")
        return 0

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        LOGGER.warning("Telegram notifications enabled but TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing.")
        return 0

    audit = read_audit()
    if audit.empty:
        return 0

    sent_ids = _sent_position_ids()
    positions = _position_by_id()
    min_confidence = _min_confidence()
    sent_count = 0

    for _, raw_row in audit.iterrows():
        row = raw_row.to_dict()
        position_id = str(row.get("position_id") or "")
        if not position_id or position_id in sent_ids:
            continue
        if str(row.get("executed", "")).lower() != "true":
            continue
        if str(row.get("reason", "")).upper() != "OPENED":
            continue
        try:
            confidence = float(row.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < min_confidence:
            continue

        error = ""
        status = "SENT"
        try:
            send_telegram_message(_message(row, positions.get(position_id)))
            sent_count += 1
            sent_ids.add(position_id)
            LOGGER.info("Telegram notification sent for position_id=%s", position_id)
        except Exception as exc:
            status = "ERROR"
            error = str(exc)
            LOGGER.exception("Telegram notification failed for position_id=%s: %s", position_id, exc)

        _append_notification(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "position_id": position_id,
                "symbol": row.get("symbol", ""),
                "signal": row.get("signal", ""),
                "confidence": row.get("confidence", ""),
                "chat_id": chat_id,
                "status": status,
                "error": error,
            }
        )

    return sent_count


def run_telegram_notifier_service() -> None:
    setup_logging()
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    interval = int(os.getenv("TELEGRAM_NOTIFIER_INTERVAL_SECONDS", "5"))
    while RUNNING:
        try:
            notify_new_strong_signals()
        except Exception as exc:
            LOGGER.exception("telegram notifier loop failed: %s", exc)
        time.sleep(interval)


if __name__ == "__main__":
    run_telegram_notifier_service()
