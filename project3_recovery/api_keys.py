"""API key generation and validation.

Keys are stored as SHA-256 hashes — plaintext is never persisted.
A hardcoded test keypair is seeded on startup if no keys exist and ENV=local.
"""
from __future__ import annotations

import hashlib
import secrets

from sqlalchemy.orm import Session


# Obviously-fake placeholders — only seeded in ENV=local.
# Prod mints real keys via scripts/create_api_key.py.
_TEST_PK = "pk_test_P3_SEED_LOCAL_DEV_ONLY"
_TEST_SK = "sk_test_P3_SEED_LOCAL_DEV_ONLY"


def generate_key_pair() -> tuple[str, str]:
    pk = "pk_test_" + secrets.token_urlsafe(24)
    sk = "sk_test_" + secrets.token_urlsafe(24)
    return pk, sk


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def seed_test_key(db: Session) -> None:
    from project3_recovery.db import ApiKey

    if db.query(ApiKey).count() == 0:
        record = ApiKey(
            name="test-default",
            publishable_key=_TEST_PK,
            secret_hash=hash_key(_TEST_SK),
        )
        db.add(record)
        db.commit()


def validate_secret_key(db: Session, provided_key: str):
    from project3_recovery.db import ApiKey

    if not provided_key.startswith("sk_test_"):
        return None
    key_hash = hash_key(provided_key)
    return db.query(ApiKey).filter(ApiKey.secret_hash == key_hash).first()
