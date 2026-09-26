"""Persisted workflow sequencing, retries, and no-op handling."""

from __future__ import annotations

import unittest
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine, select
from sqlalchemy.pool import StaticPool

from tracescholar.database import Base, create_session_factory, session_scope
from tracescholar.models import (
    ClaimGeneration, FullTextAcquisition, Paper, PaperVersion, ResearchPlanRecord,
    ResearchRun, ScreeningResult, SearchQuery,
    SearchResult, WorkflowStageExecution,
)
from tracescholar.workflow import (
    STAGE_ORDER, WorkflowOrderError, WorkflowStage, inspect_workflow,
    run_next_stage, run_workflow,
)


class WorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.factory = create_session_factory(self.engine)
        self.run_id = uuid.uuid4()
        self.paper_id = uuid.uuid4()
        with session_scope(self.factory) as session:
            run = ResearchRun(
                id=self.run_id, question="Does query rewriting improve multi-hop QA?",
                scope={"year_from": 2022}, config_snapshot={},
            )
            session.add(run)
            session.flush()
            plan = ResearchPlanRecord(
                run_id=run.id, plan_json={"normalized_question": run.question},
                input_question=run.question, input_scope=run.scope,
                schema_version=1, prompt_version="planner-v1", llm_model="test",
            )
            session.add(plan)
            paper = Paper(
                id=self.paper_id, title="Query rewriting for multi-hop question answering",
                normalized_title="query rewriting for multi hop question answering",
                year=2024, authors=[],
            )
            session.add(paper)
            session.flush()
            query = SearchQuery(
                run_id=run.id, query="query rewriting multi-hop QA", source="openalex",
                filters={}, returned_count=1,
            )
            session.add(query)
            session.flush()
            session.add(SearchResult(
                search_query_id=query.id, paper_id=paper.id,
                source_record_id="W123", source_url="https://openalex.org/W123",
            ))
            session.add(ScreeningResult(
                run_id=run.id, paper_id=paper.id, plan_id=plan.id,
                label="include", relevance_score=0.9, rationale="Relevant topic.",
                matched_inclusion_criteria=["multi-hop QA"], matched_exclusion_criteria=[],
                needs_full_text=True, sub_question_index=0, evidence_role="primary_empirical",
                input_hash="a" * 64, input_snapshot={"fixture": True},
                prompt_version="screen-v1", llm_model="test",
            ))

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_detects_completed_planning_discovery_screening_and_next_acquisition(self) -> None:
        status = inspect_workflow(self.run_id, session_factory=self.factory)
        self.assertEqual(status.completed_stages, (
            WorkflowStage.PLANNED, WorkflowStage.DISCOVERED, WorkflowStage.SCREENED,
        ))
        self.assertEqual(status.next_stage, WorkflowStage.ACQUIRED)

    def test_failed_stage_is_persisted_and_retry_keeps_prior_outputs(self) -> None:
        def fail_after_partial_save(stage, run_id, *, session_factory):
            with session_scope(session_factory) as session:
                session.add(FullTextAcquisition(
                    run_id=run_id, paper_id=self.paper_id, status="unavailable",
                    failure_code="temporary_locator_error", failure_detail="Retryable test failure.",
                ))
            raise RuntimeError("temporary source failure")

        with patch("tracescholar.workflow.service._call_stage", side_effect=fail_after_partial_save) as call:
            failed = run_next_stage(self.run_id, session_factory=self.factory)
        self.assertEqual(failed.stage, WorkflowStage.ACQUIRED)
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.snapshot.next_stage, WorkflowStage.ACQUIRED)
        self.assertEqual(failed.snapshot.last_failure_stage, WorkflowStage.ACQUIRED)
        call.assert_called_once()

        with self.factory() as session:
            self.assertIsNotNone(session.get(ResearchRun, self.run_id))
            self.assertIsNotNone(session.get(ResearchPlanRecord, session.scalar(
                select(ResearchPlanRecord.id).where(ResearchPlanRecord.run_id == self.run_id)
            )))
            attempts = list(session.scalars(select(WorkflowStageExecution).where(
                WorkflowStageExecution.run_id == self.run_id,
            ).order_by(WorkflowStageExecution.attempt)))
            self.assertEqual(len(attempts), 1)
            self.assertEqual(attempts[0].status, "failed")
            self.assertEqual(attempts[0].failure_reason, "temporary source failure")
            self.assertIsNotNone(attempts[0].finished_at)
            self.assertIsNotNone(attempts[0].duration_seconds)
            self.assertEqual(session.scalar(select(FullTextAcquisition.status).where(
                FullTextAcquisition.run_id == self.run_id,
                FullTextAcquisition.paper_id == self.paper_id,
            )), "unavailable")

        def succeed(stage, run_id, *, session_factory):
            with session_scope(session_factory) as session:
                version = PaperVersion(
                    paper_id=self.paper_id, content_hash="d" * 64,
                    storage_path="fulltext/test.pdf", content_bytes=32,
                    source_url="https://papers.example/paper.pdf",
                    source_name="Test OA Repository", license="CC-BY-4.0",
                )
                session.add(version)
                session.flush()
                acquisition = session.scalar(select(FullTextAcquisition).where(
                    FullTextAcquisition.run_id == run_id,
                    FullTextAcquisition.paper_id == self.paper_id,
                ))
                assert acquisition is not None
                acquisition.paper_version_id = version.id
                acquisition.status = "downloaded"
                acquisition.source_url = version.source_url
                acquisition.source_name = version.source_name
                acquisition.license = version.license
            return SimpleNamespace(failed=0, pending=0, failures=())

        with patch("tracescholar.workflow.service._call_stage", side_effect=succeed) as retry:
            resumed = run_next_stage(self.run_id, session_factory=self.factory)
        self.assertEqual(resumed.stage, WorkflowStage.ACQUIRED)
        self.assertEqual(resumed.status, "completed")
        self.assertEqual(resumed.snapshot.next_stage, WorkflowStage.PARSED)
        retry.assert_called_once()

        with self.factory() as session:
            attempts = list(session.scalars(select(WorkflowStageExecution).where(
                WorkflowStageExecution.run_id == self.run_id,
                WorkflowStageExecution.stage == WorkflowStage.ACQUIRED.value,
            ).order_by(WorkflowStageExecution.attempt)))
            self.assertEqual([item.status for item in attempts], ["failed", "completed"])
            self.assertEqual([item.attempt for item in attempts], [1, 2])

    def test_completed_stage_is_not_called_again_and_wrong_order_is_rejected(self) -> None:
        with patch("tracescholar.workflow.service._call_stage") as call:
            result = run_next_stage(
                self.run_id, stage=WorkflowStage.PLANNED, session_factory=self.factory,
            )
        self.assertTrue(result.skipped)
        call.assert_not_called()
        with self.assertRaises(WorkflowOrderError):
            run_next_stage(
                self.run_id, stage=WorkflowStage.PARSED, session_factory=self.factory,
            )
        call.assert_not_called()

    def test_discovery_is_blocked_without_a_frozen_plan(self) -> None:
        unplanned_id = uuid.uuid4()
        now = datetime.now(UTC)
        with session_scope(self.factory) as session:
            session.add(ResearchRun(
                id=unplanned_id, question="A research question", scope={}, config_snapshot={},
            ))
            session.add(WorkflowStageExecution(
                run_id=unplanned_id, stage=WorkflowStage.PLANNED.value, attempt=1,
                status="completed", started_at=now, finished_at=now, duration_seconds=0,
            ))
        with patch("tracescholar.workflow.service._call_stage") as call:
            result = run_next_stage(
                unplanned_id, stage=WorkflowStage.DISCOVERED, session_factory=self.factory,
            )
        self.assertEqual(result.status, "blocked")
        self.assertIn("frozen ResearchPlan", result.detail)
        self.assertEqual(result.snapshot.next_stage, WorkflowStage.DISCOVERED)
        call.assert_not_called()

    def test_pdf_parse_is_a_noop_when_no_pdf_was_acquired(self) -> None:
        with session_scope(self.factory) as session:
            session.add(FullTextAcquisition(
                run_id=self.run_id, paper_id=self.paper_id, status="unavailable",
                failure_code="not_open_access", failure_detail="No OA PDF.",
            ))
        snapshot = inspect_workflow(self.run_id, session_factory=self.factory)
        self.assertEqual(snapshot.next_stage, WorkflowStage.EVIDENCE_EXTRACTED)
        parsed = next(item for item in snapshot.stages if item.stage == WorkflowStage.PARSED)
        self.assertEqual(parsed.status, "completed")
        self.assertIn("not applicable", parsed.detail)
        with patch("tracescholar.workflow.service._call_stage") as call:
            result = run_next_stage(
                self.run_id, stage=WorkflowStage.PARSED, session_factory=self.factory,
            )
        self.assertTrue(result.skipped)
        call.assert_not_called()

    def test_synthesis_is_blocked_without_saved_evidence_spans(self) -> None:
        now = datetime.now(UTC)
        with session_scope(self.factory) as session:
            plan = session.scalar(select(ResearchPlanRecord).where(
                ResearchPlanRecord.run_id == self.run_id,
            ))
            assert plan is not None
            session.add(ClaimGeneration(
                run_id=self.run_id, plan_id=plan.id, input_hash="e" * 64,
                input_snapshot={}, prompt_version="claims-v1", llm_model="test",
                status="success", attempt_count=1, no_claim_reason="No grounded claim.",
            ))
            for stage in STAGE_ORDER[:STAGE_ORDER.index(WorkflowStage.SYNTHESIZED)]:
                session.add(WorkflowStageExecution(
                    run_id=self.run_id, stage=stage.value, attempt=1,
                    status="completed", started_at=now, finished_at=now,
                    duration_seconds=0,
                ))
        with patch("tracescholar.workflow.service._call_stage") as call:
            result = run_next_stage(
                self.run_id, stage=WorkflowStage.SYNTHESIZED, session_factory=self.factory,
            )
        self.assertEqual(result.status, "blocked")
        self.assertIn("saved EvidenceSpan", result.detail)
        self.assertEqual(result.snapshot.next_stage, WorkflowStage.SYNTHESIZED)
        call.assert_not_called()

    def test_continuous_run_advances_serially_and_stops_at_failure(self) -> None:
        calls = []

        def stage_service(stage, run_id, *, session_factory):
            calls.append(stage)
            if stage == WorkflowStage.ACQUIRED:
                with session_scope(session_factory) as session:
                    session.add(FullTextAcquisition(
                        run_id=run_id, paper_id=self.paper_id, status="unavailable",
                        failure_code="not_open_access", failure_detail="No OA PDF.",
                    ))
                return SimpleNamespace(failed=0, pending=0, failures=())
            raise RuntimeError("Evidence service temporarily unavailable")

        with patch("tracescholar.workflow.service._call_stage", side_effect=stage_service):
            result = run_workflow(self.run_id, session_factory=self.factory)

        self.assertEqual(result.status, "failed")
        self.assertEqual(calls, [WorkflowStage.ACQUIRED, WorkflowStage.EVIDENCE_EXTRACTED])
        self.assertEqual([step.status for step in result.steps], ["completed", "failed"])
        self.assertEqual(result.snapshot.next_stage, WorkflowStage.EVIDENCE_EXTRACTED)
        self.assertEqual(result.snapshot.last_failure_stage, WorkflowStage.EVIDENCE_EXTRACTED)
        with self.factory() as session:
            self.assertEqual(session.scalar(select(FullTextAcquisition.status).where(
                FullTextAcquisition.run_id == self.run_id,
                FullTextAcquisition.paper_id == self.paper_id,
            )), "unavailable")

    def test_continuous_run_returns_immediately_when_already_complete(self) -> None:
        now = datetime.now(UTC)
        with session_scope(self.factory) as session:
            for stage in STAGE_ORDER:
                session.add(WorkflowStageExecution(
                    run_id=self.run_id, stage=stage.value, attempt=1,
                    status="completed", started_at=now, finished_at=now,
                    duration_seconds=0,
                ))
        with patch("tracescholar.workflow.service._call_stage") as call:
            result = run_workflow(self.run_id, session_factory=self.factory)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.steps, ())
        self.assertIsNone(result.snapshot.next_stage)
        call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
