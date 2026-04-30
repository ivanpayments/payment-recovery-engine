"""Redis-backed idempotency cache — 24 hour TTL.

Callers pass an `Idempotency-Key` header. First call stores the response;
later calls within 24h return the cached body with header X-Idempotency-Replay.
"""
from __future__ import annotations

import json

import redis as _redis


_TTL_SECONDS = 24 * 60 * 60


def _redis_key(api_key_id: str, idem_key: str) -> str:
    return f"idem:p3:{api_key_id}:{idem_key}"


def get_cached(client: _redis.Redis, api_key_id: str, idem_key: str) -> dict | None:
    raw = client.get(_redis_key(api_key_id, idem_key))
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def store(client: _redis.Redis, api_key_id: str, idem_key: str, body: dict) -> None:
    client.setex(_redis_key(api_key_id, idem_key), _TTL_SECONDS, json.dumps(body, default=str))
