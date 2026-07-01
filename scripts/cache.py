from __future__ import annotations

import json
import logging
import os
from typing import Any

import redis

LOGGER = logging.getLogger(__name__)


def redis_url() -> str:
    return os.getenv("REDIS_URL", "redis://redis:6379/0")


def get_redis() -> redis.Redis:
    return redis.Redis.from_url(redis_url(), decode_responses=True)


def publish_event(channel: str, payload: dict[str, Any]) -> None:
    try:
        client = get_redis()
        encoded = json.dumps(payload, default=str)
        client.set(f"market_intel:last:{channel}", encoded, ex=3600)
        client.publish(channel, encoded)
    except Exception as exc:
        LOGGER.debug("Redis publish skipped: %s", exc)


def redis_status() -> str:
    try:
        return "ok" if get_redis().ping() else "unavailable"
    except Exception as exc:
        return f"error: {exc}"
