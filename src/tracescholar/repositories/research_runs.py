"""Persistence operations for ResearchRun records."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from tracescholar.database.session import session_scope
from tracescholar.models import ResearchRun


def create_research_run(
    question: str,
    *,
    scope: dict[str, Any] | None = None,
    config_snapshot: dict[str, Any] | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> ResearchRun:
    """Create and persist a pending research run."""
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("Research question must not be blank.")

    research_run = ResearchRun(
        question=normalized_question,
        scope=scope or {},
        config_snapshot=config_snapshot or {},
    )
    with session_scope(session_factory) as session:
        session.add(research_run)
        session.flush()
        session.refresh(research_run)

    return research_run
