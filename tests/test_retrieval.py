"""Embedding idempotency, retry isolation and run-scoped semantic search."""

from __future__ import annotations

import tempfile
import unittest
import uuid
import json
from pathlib import Path

import httpx
from openai import OpenAI
from sqlalchemy import create_engine, func, select
from sqlalchemy.pool import StaticPool

from tracescholar.config import Settings
from tracescholar.database import Base, create_session_factory, session_scope
from tracescholar.models import Chunk, ChunkEmbedding, FullTextAcquisition, Paper, PaperVersion, ResearchRun
from tracescholar.retrieval import (
    EmbeddingError, OpenAICompatibleEmbeddings, embed_research_run, search_chunks,
)


def basis(index: int) -> list[float]:
    values = [0.0] * 384
    values[index] = 1.0
    return values


class FakeEncoder:
    provider = "test"
    model_name = "deterministic-384"
    model_revision = "a" * 64
    source_revision = "deterministic-v1"
    endpoint_url = "https://example.org/v1"
    encoder_version = "1.0"
    dimensions = 384

    def __init__(self):
        self.passages: list[list[str]] = []
        self.queries: list[str] = []
        self.fail_texts: set[str] = set()
        self.fail_queries = False

    def embed_passages(self, texts):
        self.passages.append(list(texts))
        if any(value in self.fail_texts for value in texts):
            raise EmbeddingError("model_failed", "Simulated encoder failure")
        return [basis(0 if "rewriting" in value else 1) for value in texts]

    def embed_query(self, query):
        self.queries.append(query)
        if self.fail_queries:
            raise EmbeddingError("model_failed", "Simulated query failure")
        return basis(0)


class CloudAdapterTestCase(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(
            _env_file=None, embedding_base_url="https://embed.example/v1",
            embedding_api_key="separate-test-key", embedding_model="provider-model-v1",
            embedding_model_version="2026-09", embedding_dimensions=3,
            llm_api_key="unrelated-chat-key",
        )

    def _adapter(self, handler):
        client = OpenAI(
            api_key="separate-test-key", base_url=self.settings.embedding_base_url,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
            max_retries=0,
        )
        return OpenAICompatibleEmbeddings(settings=self.settings, client=client)

    def test_api_batches_and_reorders_by_response_index(self):
        def handler(request):
            self.assertEqual(request.url.path, "/v1/embeddings")
            self.assertEqual(request.headers["Authorization"], "Bearer separate-test-key")
            body = json.loads(request.content)
            self.assertEqual(body["model"], "provider-model-v1")
            self.assertEqual(body["dimensions"], 3)
            self.assertEqual(body["input"], ["first passage", "second passage"])
            return httpx.Response(200, json={
                "object": "list", "model": "provider-model-v1",
                "data": [
                    {"object": "embedding", "index": 1, "embedding": [0.0, 1.0, 0.0]},
                    {"object": "embedding", "index": 0, "embedding": [1.0, 0.0, 0.0]},
                ], "usage": {"prompt_tokens": 5, "total_tokens": 5},
            })
        adapter = self._adapter(handler)
        self.assertEqual(adapter.embed_passages(["first passage", "second passage"]),
                         [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.assertEqual(adapter.source_revision, "2026-09")
        self.assertEqual(adapter.endpoint_url, "https://embed.example/v1")
        self.assertEqual(len(adapter.model_revision), 64)

    def test_separate_key_missing_and_provider_errors(self):
        settings = Settings(_env_file=None, embedding_api_key=None, llm_api_key="chat-key")
        with self.assertRaises(EmbeddingError) as raised:
            OpenAICompatibleEmbeddings(settings=settings)
        self.assertEqual(raised.exception.code, "not_configured")

        adapter = self._adapter(lambda _: httpx.Response(429, json={
            "error": {"message": "rate limited", "type": "rate_limit_error"},
        }))
        with self.assertRaises(EmbeddingError) as raised:
            adapter.embed_query("question")
        self.assertEqual(raised.exception.code, "rate_limited")

    def test_missing_response_item_and_unsafe_url_are_rejected(self):
        adapter = self._adapter(lambda _: httpx.Response(200, json={
            "object": "list", "model": "provider-model-v1", "data": [],
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }))
        with self.assertRaises(EmbeddingError) as raised:
            adapter.embed_query("question")
        self.assertEqual(raised.exception.code, "invalid_response")
        with self.assertRaises(EmbeddingError) as raised:
            OpenAICompatibleEmbeddings(settings=Settings(
                _env_file=None, embedding_base_url="http://remote.example/v1",
                embedding_api_key="test", embedding_model="model", embedding_dimensions=3,
            ))
        self.assertEqual(raised.exception.code, "invalid_url")


class RetrievalTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings = Settings(_env_file=None, data_dir=Path(self.temp.name),
                                 embedding_batch_size=2)
        self.engine = create_engine("sqlite+pysqlite:///:memory:",
                                    connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(self.engine)
        self.factory = create_session_factory(self.engine)
        self.first_run = uuid.uuid4()
        self.other_run = uuid.uuid4()
        self.empty_run = uuid.uuid4()
        self.chunk_ids: list[uuid.UUID] = []
        with session_scope(self.factory) as session:
            for run_id in (self.first_run, self.other_run, self.empty_run):
                session.add(ResearchRun(id=run_id, question=f"Run {run_id}"))
            session.flush()
            for number, (run_id, content) in enumerate((
                (self.first_run, "query rewriting improves multi-hop question answering"),
                (self.first_run, "retrieval latency measurements on a benchmark"),
                (self.other_run, "query rewriting improves multi-hop question answering"),
            )):
                paper = Paper(title=f"Paper {number}", normalized_title=f"paper {number}", authors=[])
                session.add(paper)
                session.flush()
                version = PaperVersion(
                    paper_id=paper.id, content_hash=f"{number + 1:064x}",
                    storage_path=f"sample-{number}.pdf", content_bytes=100,
                    source_url="https://example.org/paper.pdf", source_name="test",
                )
                session.add(version)
                session.flush()
                session.add(FullTextAcquisition(
                    run_id=run_id, paper_id=paper.id, paper_version_id=version.id,
                    status="downloaded", attempt_count=1,
                ))
                chunk = Chunk(
                    paper_version_id=version.id, ordinal=0, text=content,
                    page_start=number + 1, page_end=number + 1,
                    section="Results", document_char_start=0,
                    document_char_end=len(content),
                    locator={"page": number + 1, "char_start": 0,
                             "char_end": len(content), "bbox": [10, 20, 200, 50]},
                    char_count=len(content), token_count=len(content.split()),
                    parser_version="test",
                )
                session.add(chunk)
                session.flush()
                self.chunk_ids.append(chunk.id)

    def tearDown(self):
        self.engine.dispose()
        self.temp.cleanup()

    def test_embed_skip_unchanged_and_reembed_changed_text(self):
        encoder = FakeEncoder()
        first = embed_research_run(self.first_run, encoder=encoder, settings=self.settings,
                                   session_factory=self.factory)
        self.assertEqual((first.total_chunks, first.newly_embedded, first.skipped_unchanged,
                          first.failed, first.pending), (2, 2, 0, 0, 0))
        self.assertEqual(len(encoder.passages), 1)
        second = embed_research_run(self.first_run, encoder=encoder, settings=self.settings,
                                    session_factory=self.factory)
        self.assertEqual((second.newly_embedded, second.skipped_unchanged), (0, 2))
        self.assertEqual(len(encoder.passages), 1)
        with session_scope(self.factory) as session:
            chunk = session.get(Chunk, self.chunk_ids[0])
            chunk.text = "updated query rewriting evidence"
            chunk.char_count = len(chunk.text)
        third = embed_research_run(self.first_run, encoder=encoder, settings=self.settings,
                                   session_factory=self.factory)
        self.assertEqual((third.newly_embedded, third.skipped_unchanged), (1, 1))
        with self.factory() as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(ChunkEmbedding)), 2)
            changed = session.scalar(select(ChunkEmbedding).where(ChunkEmbedding.chunk_id == self.chunk_ids[0]))
            self.assertEqual(changed.attempt_count, 2)

    def test_batch_failure_isolates_chunk_and_retry_succeeds(self):
        encoder = FakeEncoder()
        bad = "retrieval latency measurements on a benchmark"
        encoder.fail_texts.add(bad)
        first = embed_research_run(self.first_run, encoder=encoder, settings=self.settings,
                                   session_factory=self.factory)
        self.assertEqual((first.newly_embedded, first.failed), (1, 1))
        self.assertEqual(first.failures[0].code, "model_failed")
        with self.factory() as session:
            failed = session.scalar(select(ChunkEmbedding).where(ChunkEmbedding.status == "failed"))
            self.assertEqual(failed.chunk_id, self.chunk_ids[1])
            self.assertIsNone(failed.vector)
        encoder.fail_texts.clear()
        second = embed_research_run(self.first_run, encoder=encoder, settings=self.settings,
                                    session_factory=self.factory)
        self.assertEqual((second.newly_embedded, second.skipped_unchanged,
                          second.failed), (1, 1, 0))

    def test_run_isolation_top_k_and_locator_chain(self):
        encoder = FakeEncoder()
        embed_research_run(self.first_run, encoder=encoder, settings=self.settings,
                           session_factory=self.factory)
        embed_research_run(self.other_run, encoder=encoder, settings=self.settings,
                           session_factory=self.factory)
        first = search_chunks(self.first_run, "multi-hop query rewriting", top_k=1,
                              encoder=encoder, session_factory=self.factory)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].chunk_id, self.chunk_ids[0])
        self.assertEqual(first[0].paper_title, "Paper 0")
        self.assertEqual(first[0].page_start, 1)
        self.assertEqual(first[0].locator["page"], 1)
        self.assertAlmostEqual(first[0].similarity, 1.0)
        others = search_chunks(self.other_run, "multi-hop query rewriting", top_k=10,
                               encoder=encoder, session_factory=self.factory)
        self.assertEqual([item.chunk_id for item in others], [self.chunk_ids[2]])
        all_first = search_chunks(self.first_run, "multi-hop query rewriting", top_k=10,
                                  encoder=encoder, session_factory=self.factory)
        self.assertEqual(len(all_first), 2)
        self.assertGreaterEqual(all_first[0].similarity, all_first[1].similarity)

    def test_no_results_invalid_query_and_query_model_failure(self):
        encoder = FakeEncoder()
        self.assertEqual(search_chunks(self.empty_run, "topic", encoder=encoder,
                                       session_factory=self.factory), [])
        self.assertEqual(encoder.queries, [])
        with self.assertRaises(ValueError):
            search_chunks(self.first_run, "  ", encoder=encoder, session_factory=self.factory)
        with self.assertRaises(ValueError):
            search_chunks(self.first_run, "topic", top_k=0, encoder=encoder,
                          session_factory=self.factory)
        embed_research_run(self.first_run, encoder=encoder, settings=self.settings,
                           session_factory=self.factory)
        encoder.fail_queries = True
        with self.assertRaises(EmbeddingError):
            search_chunks(self.first_run, "topic", encoder=encoder,
                          session_factory=self.factory)

    def test_invalid_vector_only_fails_that_chunk(self):
        class InvalidEncoder(FakeEncoder):
            def embed_passages(self, texts):
                self.passages.append(list(texts))
                return [[float("nan")] * 384 if "latency" in text else basis(0)
                        for text in texts]

        result = embed_research_run(self.first_run, encoder=InvalidEncoder(),
                                    settings=self.settings, session_factory=self.factory)
        self.assertEqual((result.newly_embedded, result.failed), (1, 1))
        self.assertEqual(result.failures[0].code, "invalid_vector")

    def test_rate_limited_batch_is_recorded_without_calling_each_chunk(self):
        class RateLimitedEncoder(FakeEncoder):
            def embed_passages(self, texts):
                self.passages.append(list(texts))
                raise EmbeddingError("rate_limited", "HTTP 429")

        encoder = RateLimitedEncoder()
        result = embed_research_run(self.first_run, encoder=encoder,
                                    settings=self.settings, session_factory=self.factory)
        self.assertEqual((result.failed, result.newly_embedded), (2, 0))
        self.assertEqual(len(encoder.passages), 1)


if __name__ == "__main__":
    unittest.main()
