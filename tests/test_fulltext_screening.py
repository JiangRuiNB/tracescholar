"""Evidence scoping, decision validation, persistence and resumability."""

from __future__ import annotations

import json
import unittest
import uuid

from pydantic import ValidationError
from sqlalchemy import create_engine, select
from sqlalchemy.pool import StaticPool

from tracescholar.database import Base, create_session_factory, session_scope
from tracescholar.fulltext_screening import (
    FullTextDecision, get_fulltext_decision, screen_fulltext_research_run,
)
from tracescholar.llm import LLMError
from tracescholar.models import (
    Chunk, ChunkEmbedding, FullTextAcquisition, FullTextScreeningResult,
    Paper, PaperVersion, PdfParseRecord, ResearchPlanRecord, ResearchRun, ScreeningResult,
)
from tracescholar.planning import ResearchPlan
from tracescholar.repositories import save_research_plan
from tracescholar.retrieval import search_paper_chunks, search_plan_evidence


def _plan() -> ResearchPlan:
    return ResearchPlan.model_validate({
        "normalized_question": "Does rewriting help multi-hop RAG?",
        "sub_questions": ["Does query rewriting improve multi-hop QA?", "What harms performance?"],
        "concepts": [{"term": "query rewriting", "synonyms": [], "abbreviations": []}],
        "exclusion_terms": [],
        "inclusion_criteria": ["Multi-hop RAG study", "Comparative evaluation"],
        "exclusion_criteria": ["Single-hop-only paper"],
        "constraints": [], "ambiguity_items": [],
        "search_tracks": [{"label": "counter", "intent": "counter_evidence",
                           "query": "query rewriting negative", "rationale": "Find failures"}],
        "stop_conditions": ["Saturated"], "scope_snapshot": {},
    })


class FakeEncoder:
    provider = "test"
    model_name = "tiny"
    model_revision = "a" * 64
    source_revision = "v1"
    endpoint_url = "https://example.org/v1"
    encoder_version = "1"
    dimensions = 3

    def __init__(self):
        self.queries = []

    def embed_query(self, query):
        self.queries.append(query)
        return [1.0, 0.0, 0.0]

    def embed_passages(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeLLM:
    model_name = "fulltext-test-model"

    def __init__(self):
        self.calls = []
        self.fail_title = None
        self.bad_citation_title = None

    def generate(self, schema, *, system_prompt, user_prompt):
        assert schema is FullTextDecision
        assert "low_page_coverage" in system_prompt
        data = json.loads(user_prompt)
        self.calls.append(data)
        title = data["paper"]["title"]
        if title == self.fail_title:
            raise LLMError("simulated paper-specific failure")
        cid = data["candidate_evidence"][0]["chunk_id"]
        if title == self.bad_citation_title:
            cid = str(uuid.uuid4())
        base = {
            "rationale": "Cited passage states the paper task and evaluation scope.",
            "matched_inclusion_indices": [0], "matched_exclusion_indices": [],
            "supported_sub_question_indices": [0],
            "evidence_role": "primary_empirical_evidence",
            "evidence_chunk_ids": [cid],
        }
        if title == "Direct empirical paper":
            return {"label": "include", **base}
        if title == "Clearly single-hop paper":
            return {"label": "exclude", **base,
                    "matched_exclusion_indices": [0],
                    "supported_sub_question_indices": [], "evidence_role": "methods"}
        return {"label": "uncertain", **base,
                "rationale": "Relevant passage exists but incomplete PDF coverage prevents a firm decision.",
                "supported_sub_question_indices": [], "evidence_role": "review_background"}


class FullTextSchemaTests(unittest.TestCase):
    def test_invalid_output_and_hallucinated_chunk_are_rejected(self):
        payload = {
            "label": "include", "rationale": "Evidence is relevant.",
            "matched_inclusion_indices": [0], "matched_exclusion_indices": [],
            "supported_sub_question_indices": [0],
            "evidence_role": "primary_empirical_evidence",
            "evidence_chunk_ids": [str(uuid.uuid4())],
        }
        decision = FullTextDecision.model_validate(payload)
        with self.assertRaisesRegex(ValueError, "outside retrieved"):
            decision.validate_against(_plan(), set())
        for update in ({"label": "maybe"}, {"evidence_chunk_ids": []},
                       {"supported_sub_question_indices": [99]},
                       {"matched_inclusion_indices": [0, 0]}):
            with self.subTest(update=update):
                changed = {**payload, **update}
                if update.get("supported_sub_question_indices") == [99]:
                    with self.assertRaises(ValueError):
                        FullTextDecision.model_validate(changed).validate_against(
                            _plan(), {uuid.UUID(payload["evidence_chunk_ids"][0])})
                else:
                    with self.assertRaises(ValidationError):
                        FullTextDecision.model_validate(changed)


class FullTextServiceTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite+pysqlite:///:memory:",
                                    connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(self.engine)
        self.factory = create_session_factory(self.engine)
        self.run_id = uuid.uuid4()
        self.other_run_id = uuid.uuid4()
        self.encoder = FakeEncoder()
        self.llm = FakeLLM()
        self.paper_ids = {}
        with session_scope(self.factory) as session:
            run = ResearchRun(id=self.run_id, question="Does rewriting help multi-hop RAG?")
            session.add_all([run, ResearchRun(id=self.other_run_id, question="Unrelated")])
            session.flush()
            plan_record = save_research_plan(session, run=run, plan=_plan(),
                                             model_name="planner", prompt_version="v1")
            for index, (title, initial_label, flags) in enumerate((
                ("Direct empirical paper", "maybe", []),
                ("Clearly single-hop paper", "include", []),
                ("Noisy candidate", "maybe", ["low_page_coverage"]),
                ("First-stage excluded", "exclude", []),
            )):
                paper = Paper(title=title, normalized_title=title.lower(), authors=[],
                              year=2025, abstract="Paper abstract", language="en")
                session.add(paper)
                session.flush()
                self.paper_ids[title] = paper.id
                text = f"{title}: query rewriting evaluated in this PDF."
                version = PaperVersion(
                    paper_id=paper.id, content_hash=f"{index + 1:064x}",
                    storage_path=f"paper-{index}.pdf", content_bytes=100,
                    source_url="https://example.org/paper.pdf", source_name="test",
                )
                session.add(version)
                session.flush()
                session.add(FullTextAcquisition(
                    run_id=self.run_id, paper_id=paper.id, paper_version_id=version.id,
                    status="downloaded", attempt_count=1,
                ))
                session.add(PdfParseRecord(
                    paper_version_id=version.id, status="success", parser_version="test",
                    input_hash=version.content_hash, page_count=1, text_page_count=1,
                    chunk_count=1, total_char_count=len(text), quality_flags=flags,
                ))
                chunk = Chunk(
                    paper_version_id=version.id, ordinal=0, text=text,
                    page_start=1, page_end=1, section="Results",
                    document_char_start=0, document_char_end=len(text),
                    locator={"page": 1, "char_start": 0, "char_end": len(text)},
                    char_count=len(text), token_count=len(text.split()), parser_version="test",
                )
                session.add(chunk)
                session.flush()
                session.add(ChunkEmbedding(
                    chunk_id=chunk.id, provider=self.encoder.provider,
                    model_name=self.encoder.model_name,
                    model_revision=self.encoder.model_revision,
                    source_revision=self.encoder.source_revision,
                    endpoint_url=self.encoder.endpoint_url,
                    encoder_version=self.encoder.encoder_version,
                    dimensions=3, input_hash="a" * 64, status="success",
                    vector=[1.0, 0.0, 0.0], attempt_count=1,
                ))
                session.add(ScreeningResult(
                    run_id=self.run_id, paper_id=paper.id, plan_id=plan_record.id,
                    label=initial_label, relevance_score=0.7, rationale="Preliminary review",
                    matched_inclusion_criteria=[], matched_exclusion_criteria=[],
                    needs_full_text=True, sub_question_index=0, evidence_role="other",
                    input_hash="b" * 64, input_snapshot={},
                    prompt_version="title-v1", llm_model="title-model",
                ))

    def tearDown(self):
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    def _screen(self, limit=None):
        return screen_fulltext_research_run(
            self.run_id, llm=self.llm, encoder=self.encoder,
            limit=limit, session_factory=self.factory,
        )

    def test_paper_scoped_retrieval_and_subquestion(self):
        paper_id = self.paper_ids["Direct empirical paper"]
        hits = search_paper_chunks(self.run_id, paper_id, "query rewriting", top_k=10,
                                   encoder=self.encoder, session_factory=self.factory)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].paper_id, paper_id)
        self.assertEqual((hits[0].page_start, hits[0].section), (1, "Results"))
        self.assertEqual(search_paper_chunks(self.other_run_id, paper_id, "query rewriting",
                                             encoder=self.encoder, session_factory=self.factory), [])
        planned = search_plan_evidence(self.run_id, paper_id, 0, encoder=self.encoder,
                                       session_factory=self.factory)
        self.assertEqual([item.chunk_id for item in planned], [hits[0].chunk_id])
        with self.assertRaises(ValueError):
            search_plan_evidence(self.run_id, paper_id, 99, encoder=self.encoder,
                                 session_factory=self.factory)

    def test_fulltext_decisions_audit_and_second_run_skips_all_calls(self):
        first = self._screen()
        self.assertEqual((first.candidates, first.newly_screened, first.include,
                          first.exclude, first.uncertain, first.failed), (3, 3, 1, 1, 1, 0))
        self.assertEqual(len(self.llm.calls), 3)
        self.assertEqual(len(self.encoder.queries), 2)
        decision = get_fulltext_decision(self.run_id, self.paper_ids["Direct empirical paper"],
                                         session_factory=self.factory)
        self.assertEqual(decision["label"], "include")
        self.assertEqual(decision["supported_sub_question_indices"], [0])
        self.assertEqual(len(decision["evidence"]), 1)
        self.assertEqual((decision["evidence"][0]["page_start"],
                          decision["evidence"][0]["section"]), (1, "Results"))
        noisy = get_fulltext_decision(self.run_id, self.paper_ids["Noisy candidate"],
                                      session_factory=self.factory)
        self.assertIn("low_page_coverage", noisy["quality_warnings"])
        self.assertIn("low_page_coverage", self.llm.calls[-1]["quality_warnings"])
        with self.factory() as session:
            self.assertEqual(len(list(session.scalars(select(FullTextScreeningResult)))), 3)
        again = self._screen()
        self.assertEqual((again.newly_screened, again.skipped_unchanged, again.pending), (0, 3, 0))
        self.assertEqual(len(self.llm.calls), 3)
        self.assertEqual(len(self.encoder.queries), 2)

    def test_failure_isolation_invalid_citation_and_retry(self):
        self.llm.fail_title = "Clearly single-hop paper"
        self.llm.bad_citation_title = "Noisy candidate"
        first = self._screen()
        self.assertEqual((first.newly_screened, first.failed, first.pending), (1, 2, 2))
        with self.factory() as session:
            rows = list(session.scalars(select(FullTextScreeningResult)))
            self.assertEqual(len(rows), 3)
            self.assertEqual(sum(row.status == "failed" for row in rows), 2)
        self.llm.fail_title = None
        self.llm.bad_citation_title = None
        again = self._screen()
        self.assertEqual((again.newly_screened, again.skipped_unchanged,
                          again.failed, again.pending), (2, 1, 0, 0))
        self.assertEqual(len(self.llm.calls), 5)

    def test_missing_embedding_is_recorded_and_retried_after_repair(self):
        paper_id = self.paper_ids["Direct empirical paper"]
        with session_scope(self.factory) as session:
            embedding = session.scalar(
                select(ChunkEmbedding)
                .join(Chunk, ChunkEmbedding.chunk_id == Chunk.id)
                .join(PaperVersion, Chunk.paper_version_id == PaperVersion.id)
                .where(PaperVersion.paper_id == paper_id)
            )
            chunk_id = embedding.chunk_id
            session.delete(embedding)
        first = self._screen()
        self.assertEqual((first.newly_screened, first.failed, first.pending), (2, 1, 1))
        with self.factory() as session:
            failed = session.scalar(select(FullTextScreeningResult).where(
                FullTextScreeningResult.paper_id == paper_id,
            ))
            self.assertEqual(failed.status, "failed")
            self.assertIn("No embedded PDF chunks", failed.failure_detail)
        with session_scope(self.factory) as session:
            session.add(ChunkEmbedding(
                chunk_id=chunk_id, provider=self.encoder.provider,
                model_name=self.encoder.model_name,
                model_revision=self.encoder.model_revision,
                source_revision=self.encoder.source_revision,
                endpoint_url=self.encoder.endpoint_url,
                encoder_version=self.encoder.encoder_version,
                dimensions=3, input_hash="a" * 64, status="success",
                vector=[1.0, 0.0, 0.0], attempt_count=1,
            ))
        second = self._screen()
        self.assertEqual((second.newly_screened, second.skipped_unchanged,
                          second.failed, second.pending), (1, 2, 0, 0))

    def test_version_change_invalidates_only_that_paper(self):
        self._screen()
        paper_id = self.paper_ids["Direct empirical paper"]
        with session_scope(self.factory) as session:
            old = session.scalar(select(FullTextAcquisition).where(
                FullTextAcquisition.run_id == self.run_id,
                FullTextAcquisition.paper_id == paper_id,
            ))
            version = PaperVersion(
                paper_id=paper_id, content_hash="f" * 64,
                storage_path="new.pdf", content_bytes=100,
                source_url="https://example.org/new.pdf", source_name="test",
            )
            session.add(version)
            session.flush()
            old.paper_version_id = version.id
            text = "New version: query rewriting evidence."
            session.add(PdfParseRecord(
                paper_version_id=version.id, status="success", parser_version="test",
                input_hash=version.content_hash, page_count=1, text_page_count=1,
                chunk_count=1, total_char_count=len(text), quality_flags=[],
            ))
            chunk = Chunk(
                paper_version_id=version.id, ordinal=0, text=text, page_start=1,
                page_end=1, section="Results", document_char_start=0,
                document_char_end=len(text),
                locator={"page": 1, "char_start": 0, "char_end": len(text)},
                char_count=len(text), token_count=5, parser_version="test",
            )
            session.add(chunk)
            session.flush()
            session.add(ChunkEmbedding(
                chunk_id=chunk.id, provider=self.encoder.provider,
                model_name=self.encoder.model_name,
                model_revision=self.encoder.model_revision,
                source_revision=self.encoder.source_revision,
                endpoint_url=self.encoder.endpoint_url,
                encoder_version=self.encoder.encoder_version,
                dimensions=3, input_hash="a" * 64, status="success",
                vector=[1.0, 0.0, 0.0], attempt_count=1,
            ))
        second = self._screen()
        self.assertEqual((second.newly_screened, second.skipped_unchanged), (1, 2))
        decision = get_fulltext_decision(self.run_id, paper_id, session_factory=self.factory)
        self.assertEqual(decision["paper_version_id"], str(version.id))
        self.assertEqual(decision["evidence"][0]["paper_version_id"], str(version.id))

    def test_plan_content_change_invalidates_prior_decisions(self):
        self._screen()
        with session_scope(self.factory) as session:
            record = session.scalar(select(ResearchPlanRecord).where(
                ResearchPlanRecord.run_id == self.run_id,
            ))
            changed = dict(record.plan_json)
            changed["sub_questions"] = ["Updated multi-hop effectiveness question?",
                                        "What harms performance?"]
            record.plan_json = changed
        second = self._screen()
        self.assertEqual((second.newly_screened, second.skipped_unchanged), (3, 0))


if __name__ == "__main__":
    unittest.main()
