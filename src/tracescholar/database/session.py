"""Database engine, session factory, and transaction management."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from tracescholar.config import Settings, get_settings


class DatabaseConfigurationError(RuntimeError):
    """Raised when database access is requested without a configured URL."""


def get_database_url(settings: Settings | None = None) -> str:
    """Return the configured database URL without exposing it in logs."""
    active_settings = settings or get_settings()
    if active_settings.database_url is None:
        raise DatabaseConfigurationError(
            "TRACESCHOLAR_DATABASE_URL is required for database operations."
        )
    return active_settings.database_url.get_secret_value()


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create and cache the process-wide SQLAlchemy engine."""
    return create_engine(get_database_url(), pool_pre_ping=True)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create a session factory bound to an explicit engine."""
    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False)


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    """Return the process-wide session factory."""
    return create_session_factory(get_engine())


@contextmanager
def session_scope(
    factory: sessionmaker[Session] | None = None,
) -> Iterator[Session]:
    """Provide a transactional session with commit/rollback semantics."""
    active_factory = factory or get_session_factory()
    session = active_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
