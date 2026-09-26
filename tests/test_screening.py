"""Title/abstract screening schema, persistence, idempotency and retry tests."""

from __future__ import annotations

import json
import unittest
import uuid

from pydantic import ValidationError
from sqlalchemy import create_engine, func, select
from sqlalchemy.pool import StaticPool

from tracescholar.database import Base, create_session_factory, session_scope
from tracescholar.llm import LLMError
from tracescholar.models import Paper, ResearchRun, ScreeningResult
from tracescholar.planning import ResearchPlan
from tracescholar.repositories import (
    add_search_result, create_research_run, create_search_query, save_research_plan,
    upsert_paper,
)
from tracescholar.screening import ScreeningDecision, screen_research_run


def _plan() -> ResearchPlan:
    return ResearchPlan.model_validate({
        "normalized_question": "Does query rewriting help multi-hop RAG?",
        "sub_questions": ["Does it improve retrieval?", "What are failure cases?"],
        "concepts": [{"term": "query rewriting", "synonyms": ["query reformulation"],
                      "abbreviations": ["QR"]}],
        "exclusion_terms": [],
        "inclusion_criteria": ["RAG on multi-hop QA", "Comparative evaluation"],
        "exclusion_criteria": ["Single-hop-only work", "Unrelated modality"],
        "constraints": [], "ambiguity_items": [],
        "search_tracks": [{"label": "counter", "intent": "counter_evidence",
                           "query": "query rewriting failures", "rationale": "negative results"}],
        "stop_conditions": ["Saturation"], "scope_snapshot": {},
    })


def _decision(label: str, **changes):
    payload = {
        "label": label, "relevance_score": 0.75, "rationale": "Matches the RAG task and evaluation.",
        "matched_inclusion_indices": [0, 1], "matched_exclusion_indices": [],
        "needs_full_text": False, "sub_question_index": 0, "evidence_role": "outcome",
    }
    payload.update(changes)
    return payload


class FakeLLM:
    model_name = "test-screening-model"

    def __init__(self, *, fail_title: str | None = None, invalid_title: str | None = None) -> None:
        self.fail_title = fail_title
        self.invalid_title = invalid_title
        self.calls: list[dict] = []

    def generate(self, schema, *, system_prompt: str, user_prompt: str):
        assert schema is ScreeningDecision
        assert "Prioritize recall" in system_prompt
        prompt = json.loads(user_prompt)
        self.calls.append(prompt)
        title = prompt["paper"]["title"]
        if title == self.fail_title:
            raise LLMError("Simulated provider failure.")
        if title == self.invalid_title:
            return _decision("exclude", matched_inclusion_indices=[999])
        if title == "Clearly unrelated image paper":
            return _decision(
                "exclude", relevance_score=0.05, rationale="Studies images, not RAG QA.",
                matched_inclusion_indices=[], matched_exclusion_indices=[1],
                sub_question_index=None, evidence_role="other",
            )
        if title == "Title-only candidate":
            return _decision("exclude", rationale="Insufficient abstract to judge.")
        return _decision("include")


class ScreeningDecisionSchemaTestCase(unittest.TestCase):
    def test_rejects_invalid_label_score_fields_and_criterion_indices(self) -> None:
        for invalid in (
            _decision("interesting"), _decision("include", relevance_score=1.2),
            _decision("include", rationale=" "),
            _decision("include", matched_inclusion_indices=[0, 0]),
            _decision("include", made_up_field="x"),
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                ScreeningDecision.model_validate(invalid)
        decision = ScreeningDecision.model_validate(_decision("include", sub_question_index=9))
        with self.assertRaises(ValueError):
            decision.validate_against_plan(_plan())


class ScreeningServiceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:",
                                    connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(self.engine)
        self.factory = create_session_factory(self.engine)
        self.run = create_research_run("Does query rewriting help multi-hop RAG?",
                                       session_factory=self.factory)
        with session_scope(self.factory) as session:
            save_research_plan(session, run=session.get(ResearchRun, self.run.id),
                               plan=_plan(), model_name="planner", prompt_version="planner-v1")
            query = create_search_query(session, run_id=self.run.id, query="query rewriting RAG",
                                        source="openalex")
            for i, (title, abstract) in enumerate((
                ("Relevant RAG paper", "Compares query rewriting for multi-hop QA."),
                ("Title-only candidate", None),
                ("Clearly unrelated image paper", "An image recognition study."),
            )):
                paper = upsert_paper(session, title=title, doi=f"10.5555/screen-{i}",
                                     year=2025, abstract=abstract, language="en")
                add_search_result(session, search_query=query, paper=paper,
                                  source_record_id=f"W{i + 1}")

    def tearDown(self) -> None:
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    def test_batch_persists_all_decisions_and_rerun_skips_llm(self) -> None:
        llm = FakeLLM()
        first = screen_research_run(self.run.id, llm=llm, session_factory=self.factory)
        self.assertEqual((first.total_papers, first.include, first.maybe, first.exclude), (3, 1, 1, 1))
        self.assertEqual(first.newly_screened, 3)
        self.assertEqual(first.pending, 0)
        self.assertEqual(len(llm.calls), 3)
        with self.factory() as session:
            results = list(session.scalars(select(ScreeningResult)))
            self.assertEqual(len(results), 3)
            by_title = {session.get(Paper, row.paper_id).title: row for row in results}
            self.assertEqual(by_title["Relevant RAG paper"].matched_inclusion_criteria,
                             ["RAG on multi-hop QA", "Comparative evaluation"])
            self.assertEqual(by_title["Clearly unrelated image paper"].matched_exclusion_criteria,
                             ["Unrelated modality"])
            self.assertEqual(by_title["Clearly unrelated image paper"].evidence_role, "other")
            self.assertEqual(by_title["Title-only candidate"].label, "maybe")
            self.assertTrue(by_title["Title-only candidate"].needs_full_text)
            self.assertIn("Abstract unavailable", by_title["Title-only candidate"].rationale)
            self.assertEqual(by_title["Relevant RAG paper"].sub_question_index, 0)
            self.assertTrue(all(row.input_hash and row.input_snapshot for row in results))
        again = screen_research_run(self.run.id, llm=llm, session_factory=self.factory)
        self.assertEqual(again.newly_screened, 0)
        self.assertEqual(again.skipped_unchanged, 3)
        self.assertEqual(len(llm.calls), 3)

    def test_limit_and_failure_retry_only_pending_papers(self) -> None:
        llm = FakeLLM(fail_title="Clearly unrelated image paper")
        first = screen_research_run(self.run.id, llm=llm, limit=2,
                                    session_factory=self.factory)
        self.assertEqual(first.newly_screened, 1)
        self.assertEqual(len(first.failures), 1)
        self.assertEqual(first.pending, 2)
        with self.factory() as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(ScreeningResult)), 1)
        llm.fail_title = None
        second = screen_research_run(self.run.id, llm=llm, session_factory=self.factory)
        self.assertEqual(second.newly_screened, 2)
        self.assertEqual(second.skipped_unchanged, 1)
        self.assertEqual(second.pending, 0)
        self.assertEqual(len(llm.calls), 4)

    def test_invalid_output_is_not_saved_and_can_retry(self) -> None:
        llm = FakeLLM(invalid_title="Relevant RAG paper")
        first = screen_research_run(self.run.id, llm=llm, session_factory=self.factory)
        self.assertEqual(len(first.failures), 1)
        self.assertEqual(first.pending, 1)
        self.assertEqual(first.newly_screened, 2)
        llm.invalid_title = None
        second = screen_research_run(self.run.id, llm=llm, session_factory=self.factory)
        self.assertEqual(second.newly_screened, 1)
        self.assertEqual(second.skipped_unchanged, 2)
        self.assertEqual(second.pending, 0)

    def test_changed_paper_metadata_rescreens_only_that_paper(self) -> None:
        llm = FakeLLM()
        screen_research_run(self.run.id, llm=llm, session_factory=self.factory)
        with session_scope(self.factory) as session:
            paper = session.scalar(select(Paper).where(Paper.title == "Relevant RAG paper"))
            paper.abstract = "Updated comparative multi-hop experiment."
        second = screen_research_run(self.run.id, llm=llm, session_factory=self.factory)
        self.assertEqual(second.newly_screened, 1)
        self.assertEqual(second.skipped_unchanged, 2)
        self.assertEqual(len(llm.calls), 4)
        with self.factory() as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(ScreeningResult)), 3)

    def test_model_change_invalidates_saved_input_hashes(self) -> None:
        llm = FakeLLM()
        screen_research_run(self.run.id, llm=llm, session_factory=self.factory)
        llm.model_name = "new-screening-model"
        second = screen_research_run(self.run.id, llm=llm, session_factory=self.factory)
        self.assertEqual(second.newly_screened, 3)
        self.assertEqual(second.skipped_unchanged, 0)
        self.assertEqual(len(llm.calls), 6)
        with self.factory() as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(ScreeningResult)), 3)
            self.assertEqual({row.llm_model for row in session.scalars(select(ScreeningResult))},
                             {"new-screening-model"})

    def test_requires_frozen_plan_and_existing_run(self) -> None:
        llm = FakeLLM()
        with self.assertRaises(LookupError):
            screen_research_run(uuid.uuid4(), llm=llm, session_factory=self.factory)
        unplanned = create_research_run("Unplanned", session_factory=self.factory)
        with self.assertRaisesRegex(ValueError, "frozen ResearchPlan"):
            screen_research_run(unplanned.id, llm=llm, session_factory=self.factory)
        self.assertEqual(llm.calls, [])

    def test_changed_run_scope_is_rejected_before_model_calls(self) -> None:
        llm = FakeLLM()
        with session_scope(self.factory) as session:
            session.get(ResearchRun, self.run.id).scope = {"year_from": 2024}
        with self.assertRaisesRegex(ValueError, "changed after planning"):
            screen_research_run(self.run.id, llm=llm, session_factory=self.factory)
        self.assertEqual(llm.calls, [])


if __name__ == "__main__":
    unittest.main()
