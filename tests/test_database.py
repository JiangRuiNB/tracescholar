"""Database-layer tests for ResearchRun persistence."""

from __future__ import annotations

import unittest

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from tracescholar.database import Base, create_session_factory, session_scope
from tracescholar.models import ResearchRun, ResearchRunStatus
from tracescholar.repositories import create_research_run


class ResearchRunDatabaseTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.session_factory = create_session_factory(self.engine)

    def tearDown(self) -> None:
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    def test_create_research_run_persists_required_fields(self) -> None:
        research_run = create_research_run(
            "  What evidence supports agentic retrieval?  ",
            scope={"year_from": 2024},
            config_snapshot={"source": "test"},
            session_factory=self.session_factory,
        )

        with self.session_factory() as session:
            persisted = session.get(ResearchRun, research_run.id)

        self.assertIsNotNone(persisted)
        self.assertEqual(persisted.question, "What evidence supports agentic retrieval?")
        self.assertEqual(persisted.status, ResearchRunStatus.PENDING)
        self.assertEqual(persisted.scope, {"year_from": 2024})
        self.assertEqual(persisted.config_snapshot, {"source": "test"})
        self.assertIsNotNone(persisted.created_at)
        self.assertIsNotNone(persisted.updated_at)

    def test_session_scope_rolls_back_failed_transaction(self) -> None:
        with self.assertRaises(RuntimeError):
            with session_scope(self.session_factory) as session:
                session.add(ResearchRun(question="This must be rolled back"))
                raise RuntimeError("simulated failure")

        with self.session_factory() as session:
            count = session.scalar(select(func.count()).select_from(ResearchRun))

        self.assertEqual(count, 0)

    def test_blank_question_is_rejected_before_database_write(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be blank"):
            create_research_run("   ", session_factory=self.session_factory)

        with Session(self.engine) as session:
            count = session.scalar(select(func.count()).select_from(ResearchRun))

        self.assertEqual(count, 0)
