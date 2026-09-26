"""Scope Planner schema, freeze semantics, and persistence tests."""

from __future__ import annotations

import json
import unittest
import uuid

from pydantic import ValidationError
from sqlalchemy import create_engine, func, select
from sqlalchemy.pool import StaticPool

from tracescholar.database import Base, create_session_factory, session_scope
from tracescholar.models import ResearchPlanRecord, ResearchRun, SearchQuery
from tracescholar.planning import (
    PLANNER_PROMPT_VERSION,
    PlanningError,
    ResearchPlan,
    get_research_plan,
    plan_research_run,
)
from tracescholar.planning.schemas import ResearchPlanDraft
from tracescholar.repositories import PlanFrozenError, create_research_run


def _draft() -> dict:
    return {
        "normalized_question": "RAG 中 query rewriting 对多跳问答的效果如何？",
        "sub_questions": [
            "不同 query rewriting 方法如何定义？",
            "多跳问答中的检索和答案质量如何比较？",
            "哪些实验报告无收益或负面结果？",
        ],
        "concepts": [
            {
                "term": "query rewriting",
                "synonyms": ["query reformulation", "query expansion"],
                "abbreviations": ["QR"],
            },
            {
                "term": "retrieval augmented generation",
                "synonyms": ["retrieval-augmented generation"],
                "abbreviations": ["RAG"],
            },
        ],
        "exclusion_terms": ["image retrieval"],
        "inclusion_criteria": ["实证研究：RAG 多跳问答中的 query rewriting。"],
        "exclusion_criteria": ["仅研究单跳问答且无法分离多跳结果。"],
        "constraints": ["关注多跳问答和可比较的评测。"],
        "ambiguity_items": [
            {
                "item": "是否只纳入英文论文？",
                "impact_on_scope": "会影响可纳入论文集合。",
                "conservative_assumption": "先以英语论文为主，等待用户确认。",
                "requires_clarification": True,
            }
        ],
        "search_tracks": [
            {
                "label": "核心研究",
                "intent": "core",
                "query": "retrieval augmented generation query rewriting multi hop question answering",
                "rationale": "找直接评估研究。",
            },
            {
                "label": "负面或空结果",
                "intent": "counter_evidence",
                "query": "RAG query rewriting multi hop failure no improvement negative results",
                "rationale": "主动寻找反证和边界条件。",
            },
        ],
        "stop_conditions": ["主要方法和负面结果均有覆盖，新增检索不再带来新证据。"],
    }


class FakeLLM:
    model_name = "test-compatible-model"

    def __init__(self, output: object) -> None:
        self.output = output
        self.calls: list[tuple[type, str, str]] = []

    def generate(self, schema, *, system_prompt: str, user_prompt: str):
        self.calls.append((schema, system_prompt, user_prompt))
        return self.output


class ResearchPlanSchemaTestCase(unittest.TestCase):
    def test_draft_validates_counter_evidence_and_rejects_extra_fields(self) -> None:
        valid = ResearchPlanDraft.model_validate(_draft())
        self.assertEqual(valid.search_tracks[1].intent, "counter_evidence")
        invalid = _draft()
        invalid["search_tracks"] = invalid["search_tracks"][:1]
        with self.assertRaisesRegex(ValidationError, "counter-evidence"):
            ResearchPlanDraft.model_validate(invalid)
        invalid = _draft()
        invalid["answer"] = "Query rewriting improves accuracy."
        with self.assertRaises(ValidationError):
            ResearchPlanDraft.model_validate(invalid)

    def test_blank_question_and_missing_subquestions_are_rejected(self) -> None:
        invalid = _draft()
        invalid["normalized_question"] = "  "
        with self.assertRaises(ValidationError):
            ResearchPlanDraft.model_validate(invalid)
        invalid = _draft()
        invalid["sub_questions"] = []
        with self.assertRaises(ValidationError):
            ResearchPlanDraft.model_validate(invalid)


class ScopePlannerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.factory = create_session_factory(self.engine)
        self.run = create_research_run(
            "RAG 中 query rewriting 对多跳问答的效果如何？",
            scope={"year_from": 2023, "languages": ["en"]},
            session_factory=self.factory,
        )

    def tearDown(self) -> None:
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    def test_plan_is_structured_frozen_and_restored_from_database(self) -> None:
        llm = FakeLLM(_draft())
        plan = plan_research_run(self.run.id, llm=llm, session_factory=self.factory)
        self.assertIsInstance(plan, ResearchPlan)
        self.assertEqual(plan.scope_snapshot, {"year_from": 2023, "languages": ["en"]})
        self.assertTrue(any("year_from=2023" in item for item in plan.constraints))
        self.assertTrue(any('languages=["en"]' in item for item in plan.inclusion_criteria))
        self.assertEqual(llm.calls[0][0], ResearchPlanDraft)
        self.assertIn("Do NOT answer", llm.calls[0][1])
        prompt_input = json.loads(llm.calls[0][2])
        self.assertEqual(prompt_input["user_scope"], self.run.scope)
        self.assertEqual(prompt_input["research_question"], self.run.question)

        with self.factory() as session:
            stored_run = session.get(ResearchRun, self.run.id)
            record = stored_run.research_plan
            self.assertIsNotNone(record)
            self.assertEqual(record.input_scope, self.run.scope)
            self.assertEqual(record.input_question, self.run.question)
            self.assertEqual(record.schema_version, 1)
            self.assertEqual(record.prompt_version, PLANNER_PROMPT_VERSION)
            self.assertEqual(record.llm_model, "test-compatible-model")
            self.assertEqual(record.plan_json, plan.model_dump(mode="json"))
            self.assertEqual(session.scalar(select(func.count()).select_from(SearchQuery)), 0)

        reloaded = get_research_plan(self.run.id, session_factory=self.factory)
        self.assertEqual(reloaded, plan)
        again = plan_research_run(self.run.id, llm=llm, session_factory=self.factory)
        self.assertEqual(again, plan)
        self.assertEqual(len(llm.calls), 1)

    def test_invalid_llm_output_is_not_persisted(self) -> None:
        invalid = _draft()
        invalid["search_tracks"] = invalid["search_tracks"][:1]
        with self.assertRaises(PlanningError):
            plan_research_run(self.run.id, llm=FakeLLM(invalid), session_factory=self.factory)
        with self.factory() as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(ResearchPlanRecord)), 0)

    def test_changed_scope_after_freeze_is_detected(self) -> None:
        plan_research_run(self.run.id, llm=FakeLLM(_draft()), session_factory=self.factory)
        with session_scope(self.factory) as session:
            run = session.get(ResearchRun, self.run.id)
            run.scope = {"year_from": 2024}
        with self.assertRaises(PlanFrozenError):
            get_research_plan(self.run.id, session_factory=self.factory)

    def test_blank_run_and_unknown_run_never_call_llm(self) -> None:
        llm = FakeLLM(_draft())
        with self.assertRaises(LookupError):
            plan_research_run(uuid.uuid4(), llm=llm, session_factory=self.factory)
        with self.assertRaises(ValueError):
            create_research_run("  ", session_factory=self.factory)
        self.assertEqual(llm.calls, [])
