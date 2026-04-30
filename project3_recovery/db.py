"""Minimal SQLAlchemy setup — only needed for the api_keys table."""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session


DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://project3:project3@localhost:5432/project3",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)


class Base(DeclarativeBase):
    pass


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: str = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    name: str = Column(String, nullable=False)
    publishable_key: str = Column(String, nullable=False, unique=True)
    secret_hash: str = Column(String, nullable=False, unique=True)
    created_at: datetime = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    with Session(engine) as session:
        yield session
