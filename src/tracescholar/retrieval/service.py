"""Resumable chunk vectorization and exact, run-scoped pgvector search."""

from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import Float, bindparam, func, select
from sqlalchemy.orm import Session, sessionmaker

from tracescholar.config import Settings, get_settings
from tracescholar.database import get_session_factory, session_scope
from tracescholar.models import (
    Chunk, ChunkEmbedding, FullTextAcquisition, Paper, PaperVersion, ResearchRun,
)
from tracescholar.retrieval.encoder import (
    EmbeddingEncoder, EmbeddingError, OpenAICompatibleEmbeddings,
)
from tracescholar.repositories import load_research_plan


@dataclass(frozen=True, slots=True)
class EmbeddingFailure:
    chunk_id: uuid.UUID
    code: str
    detail: str


@dataclass(frozen=True, slots=True)
class EmbeddingSummary:
    run_id: uuid.UUID
    total_chunks: int
    newly_embedded: int
    skipped_unchanged: int
    failed: int
    pending: int
    provider: str
    model_name: str
    model_revision: str
    encoder_version: str
    dimensions: int
    failures: tuple[EmbeddingFailure, ...]


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    run_id: uuid.UUID
    chunk_id: uuid.UUID
    paper_id: uuid.UUID
    paper_title: str
    paper_version_id: uuid.UUID
    storage_path: str
    content_hash: str
    page_start: int
    page_end: int
    section: str
    ordinal: int
    locator: dict[str, Any]
    text: str
    cosine_distance: float
    similarity: float
    model_name: str
    model_revision: str


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_vector(vector: Any, dimensions: int) -> list[float]:
    try:
        values = [float(item) for item in vector]
    except (TypeError, ValueError) as error:
        raise EmbeddingError("invalid_vector", "Embedding contains a non-numeric value.") from error
    if len(values) != dimensions or not all(math.isfinite(item) for item in values):
        raise EmbeddingError("invalid_vector", "Embedding has an invalid dimension or non-finite value.")
    if not any(item != 0 for item in values):
        raise EmbeddingError("invalid_vector", "Zero vectors cannot be used for cosine search.")
    return values


def _run_chunks(session: Session, run_id: uuid.UUID) -> list[Chunk]:
    return list(session.scalars(
        select(Chunk)
        .join(FullTextAcquisition,
              FullTextAcquisition.paper_version_id == Chunk.paper_version_id)
        .where(FullTextAcquisition.run_id == run_id,
               FullTextAcquisition.status.in_(("downloaded", "cached")))
        .distinct()
        .order_by(Chunk.paper_version_id, Chunk.ordinal)
    ))


def _save_batch(
    factory: sessionmaker[Session], encoder: EmbeddingEncoder,
    outcomes: list[tuple[Chunk, list[float] | None, EmbeddingError | None]],
) -> None:
    with session_scope(factory) as session:
        ids = [chunk.id for chunk, _, _ in outcomes]
        existing = {row.chunk_id: row for row in session.scalars(
            select(ChunkEmbedding).where(
                ChunkEmbedding.chunk_id.in_(ids),
                ChunkEmbedding.provider == encoder.provider,
                ChunkEmbedding.model_name == encoder.model_name,
                ChunkEmbedding.model_revision == encoder.model_revision,
                ChunkEmbedding.encoder_version == encoder.encoder_version,
            )
        )}
        for chunk, vector, error in outcomes:
            row = existing.get(chunk.id)
            if row is None:
                row = ChunkEmbedding(
                    chunk_id=chunk.id, provider=encoder.provider,
                    model_name=encoder.model_name,
                    model_revision=encoder.model_revision,
                    source_revision=encoder.source_revision,
                    endpoint_url=encoder.endpoint_url,
                    encoder_version=encoder.encoder_version,
                    attempt_count=0,
                )
                session.add(row)
            row.dimensions = encoder.dimensions
            row.source_revision = encoder.source_revision
            row.endpoint_url = encoder.endpoint_url
            row.input_hash = _hash(chunk.text)
            row.attempt_count += 1
            row.attempted_at = datetime.now(timezone.utc)
            if error is None:
                row.status = "success"
                row.vector = vector
                row.failure_code = row.failure_detail = None
            else:
                row.status = "failed"
                row.vector = None
                row.failure_code = error.code
                row.failure_detail = str(error)[:2000]


def _encode_batch(
    encoder: EmbeddingEncoder, chunks: list[Chunk],
) -> list[tuple[Chunk, list[float] | None, EmbeddingError | None]]:
    """Fall back to individual calls if a batch fails; preserve per-chunk errors."""
    try:
        raw = encoder.embed_passages([chunk.text for chunk in chunks])
        if len(raw) != len(chunks):
            raise EmbeddingError("batch_size_mismatch", "Provider returned the wrong number of vectors.")
    except EmbeddingError as error:
        if error.code in {"rate_limited", "timeout", "connection_failed"}:
            return [(chunk, None, error) for chunk in chunks]
        raw = []
    except Exception:
        raw = []
    if not raw:
        outcomes = []
        for chunk in chunks:
            try:
                single = encoder.embed_passages([chunk.text])
                if len(single) != 1:
                    raise EmbeddingError("batch_size_mismatch", "Provider returned no vector for a chunk.")
                outcomes.append((chunk, _validate_vector(single[0], encoder.dimensions), None))
            except Exception as error:
                issue = error if isinstance(error, EmbeddingError) else EmbeddingError(
                    "encoding_failed", f"Embedding provider failed: {error}"
                )
                outcomes.append((chunk, None, issue))
        return outcomes
    outcomes = []
    for chunk, vector in zip(chunks, raw, strict=True):
        try:
            outcomes.append((chunk, _validate_vector(vector, encoder.dimensions), None))
        except EmbeddingError as error:
            outcomes.append((chunk, None, error))
    return outcomes


def embed_research_run(
    run_id: uuid.UUID, *, encoder: EmbeddingEncoder | None = None,
    settings: Settings | None = None,
    session_factory: sessionmaker[Session] | None = None,
    limit: int | None = None,
) -> EmbeddingSummary:
    """Vectorize run-scoped chunks, skipping successes with unchanged text."""
    if limit is not None and not 1 <= limit <= 100_000:
        raise ValueError("Embedding limit must be between 1 and 100000.")
    active_settings = settings or get_settings()
    factory = session_factory or get_session_factory()
    with factory() as session:
        if session.get(ResearchRun, run_id) is None:
            raise LookupError(f"ResearchRun {run_id} does not exist.")
        chunks = _run_chunks(session, run_id)
    if not chunks:
        return EmbeddingSummary(run_id, 0, 0, 0, 0, 0, "openai-compatible",
                                active_settings.embedding_model or "", "", "",
                                active_settings.embedding_dimensions or 0, ())

    active = encoder or OpenAICompatibleEmbeddings(settings=active_settings)
    if not 1 <= active.dimensions <= 2000:
        raise EmbeddingError("dimension_mismatch", "pgvector supports 1 to 2000 dimensions here.")
    chunk_ids = [chunk.id for chunk in chunks]
    with factory() as session:
        records = {row.chunk_id: row for row in session.scalars(select(ChunkEmbedding).where(
            ChunkEmbedding.chunk_id.in_(chunk_ids),
            ChunkEmbedding.provider == active.provider,
            ChunkEmbedding.model_name == active.model_name,
            ChunkEmbedding.model_revision == active.model_revision,
            ChunkEmbedding.encoder_version == active.encoder_version,
        ))}
    pending = []
    skipped = 0
    for chunk in chunks:
        record = records.get(chunk.id)
        if record is not None and record.status == "success" \
                and record.input_hash == _hash(chunk.text) \
                and record.dimensions == active.dimensions and record.vector is not None:
            skipped += 1
        else:
            pending.append(chunk)

    selected = pending[:limit] if limit is not None else pending
    failures: list[EmbeddingFailure] = []
    new = 0
    for start in range(0, len(selected), active_settings.embedding_batch_size):
        outcomes = _encode_batch(active, selected[start:start + active_settings.embedding_batch_size])
        _save_batch(factory, active, outcomes)
        for chunk, _, error in outcomes:
            if error is None:
                new += 1
            else:
                failures.append(EmbeddingFailure(chunk.id, error.code, str(error)))
    with factory() as session:
        final = {row.chunk_id: (row.status, row.input_hash) for row in session.scalars(
            select(ChunkEmbedding).where(
                ChunkEmbedding.chunk_id.in_(chunk_ids),
                ChunkEmbedding.provider == active.provider,
                ChunkEmbedding.model_name == active.model_name,
                ChunkEmbedding.model_revision == active.model_revision,
                ChunkEmbedding.encoder_version == active.encoder_version,
            )
        )}
    failed = sum(final.get(chunk.id) == ("failed", _hash(chunk.text)) for chunk in chunks)
    return EmbeddingSummary(
        run_id, len(chunks), new, skipped, failed,
        len(chunks) - new - skipped - failed,
        active.provider, active.model_name, active.model_revision,
        active.encoder_version, active.dimensions, tuple(failures),
    )


def _scoped_rows(session: Session, run_id: uuid.UUID, encoder: EmbeddingEncoder,
                 paper_id: uuid.UUID | None = None,
                 paper_version_id: uuid.UUID | None = None):
    statement = (
        select(Chunk, PaperVersion, Paper, ChunkEmbedding)
        .join(PaperVersion, Chunk.paper_version_id == PaperVersion.id)
        .join(Paper, PaperVersion.paper_id == Paper.id)
        .join(FullTextAcquisition,
              FullTextAcquisition.paper_version_id == PaperVersion.id)
        .join(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.id)
        .where(
            FullTextAcquisition.run_id == run_id,
            FullTextAcquisition.status.in_(("downloaded", "cached")),
            ChunkEmbedding.status == "success",
            ChunkEmbedding.provider == encoder.provider,
            ChunkEmbedding.model_name == encoder.model_name,
            ChunkEmbedding.model_revision == encoder.model_revision,
            ChunkEmbedding.encoder_version == encoder.encoder_version,
            ChunkEmbedding.dimensions == encoder.dimensions,
        )
    )
    if paper_id is not None:
        statement = statement.where(Paper.id == paper_id)
    if paper_version_id is not None:
        statement = statement.where(PaperVersion.id == paper_version_id)
    return statement


def _result(run_id: uuid.UUID, chunk: Chunk, version: PaperVersion,
            paper: Paper, distance: float, encoder: EmbeddingEncoder) -> RetrievedChunk:
    return RetrievedChunk(
        run_id, chunk.id, paper.id, paper.title, version.id,
        version.storage_path, version.content_hash,
        chunk.page_start, chunk.page_end, chunk.section, chunk.ordinal,
        chunk.locator, chunk.text, distance, 1.0 - distance,
        encoder.model_name, encoder.model_revision,
    )


def search_chunks(
    run_id: uuid.UUID, query: str, *, top_k: int = 10,
    encoder: EmbeddingEncoder | None = None,
    settings: Settings | None = None,
    session_factory: sessionmaker[Session] | None = None,
    paper_id: uuid.UUID | None = None,
    paper_version_id: uuid.UUID | None = None,
    query_vector: list[float] | None = None,
) -> list[RetrievedChunk]:
    """Cosine-search acquired full text, optionally limited to one paper in the run."""
    if not query.strip():
        raise ValueError("Search query must not be empty.")
    if not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100.")
    factory = session_factory or get_session_factory()
    with factory() as session:
        if session.get(ResearchRun, run_id) is None:
            raise LookupError(f"ResearchRun {run_id} does not exist.")
        available = (
            select(ChunkEmbedding.id)
            .join(Chunk, ChunkEmbedding.chunk_id == Chunk.id)
            .join(PaperVersion, PaperVersion.id == Chunk.paper_version_id)
            .join(FullTextAcquisition,
                  FullTextAcquisition.paper_version_id == Chunk.paper_version_id)
            .where(FullTextAcquisition.run_id == run_id,
                   FullTextAcquisition.status.in_(("downloaded", "cached")),
                   ChunkEmbedding.status == "success")
            .limit(1)
        )
        if paper_id is not None:
            available = available.where(PaperVersion.paper_id == paper_id)
        if paper_version_id is not None:
            available = available.where(PaperVersion.id == paper_version_id)
        has_vectors = session.scalar(available)
    if has_vectors is None:
        return []
    active = encoder or OpenAICompatibleEmbeddings(settings=settings or get_settings())
    vector = _validate_vector(
        active.embed_query(query.strip()) if query_vector is None else query_vector,
        active.dimensions,
    )
    with factory() as session:
        base = _scoped_rows(session, run_id, active, paper_id, paper_version_id)
        if session.bind.dialect.name == "postgresql":
            parameter = bindparam("query_vector", value=vector, type_=VECTOR())
            distance = ChunkEmbedding.vector.op("<=>", return_type=Float)(parameter)
            rows = session.execute(
                base.add_columns(distance.label("distance"))
                .order_by(distance, Chunk.id).limit(top_k)
            ).all()
            return [_result(run_id, chunk, version, paper, float(dist), active)
                    for chunk, version, paper, _, dist in rows]
        # SQLite exists only for lightweight unit tests; production uses pgvector.
        matches = []
        query_norm = math.sqrt(sum(value * value for value in vector))
        for chunk, version, paper, embedding in session.execute(base):
            values = embedding.vector
            if values is None:
                continue
            norm = math.sqrt(sum(value * value for value in values))
            if norm == 0:
                continue
            distance = 1 - sum(a * b for a, b in zip(vector, values, strict=True)) / (query_norm * norm)
            matches.append(_result(run_id, chunk, version, paper, distance, active))
        return sorted(matches, key=lambda item: (item.cosine_distance, str(item.chunk_id)))[:top_k]


def search_paper_chunks(
    run_id: uuid.UUID, paper_id: uuid.UUID, query: str, *, top_k: int = 3,
    encoder: EmbeddingEncoder | None = None,
    session_factory: sessionmaker[Session] | None = None,
    query_vector: list[float] | None = None,
    paper_version_id: uuid.UUID | None = None,
) -> list[RetrievedChunk]:
    """Return page-located candidate evidence from only this run's PDF of a paper."""
    return search_chunks(
        run_id, query, top_k=top_k, encoder=encoder,
        session_factory=session_factory, paper_id=paper_id,
        paper_version_id=paper_version_id, query_vector=query_vector,
    )


def plan_evidence_queries(plan) -> list[str]:
    """Turn frozen sub-questions and concepts into bounded passage queries."""
    terms = " ".join(concept.term for concept in plan.concepts[:6])
    return [f"{question} {terms}".strip() for question in plan.sub_questions]


def search_plan_evidence(
    run_id: uuid.UUID, paper_id: uuid.UUID, sub_question_index: int, *,
    top_k: int = 3, encoder: EmbeddingEncoder | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> list[RetrievedChunk]:
    """Find candidate evidence for one plan sub-question inside one paper only."""
    factory = session_factory or get_session_factory()
    with factory() as session:
        plan = load_research_plan(session, run_id)
        if plan is None:
            raise ValueError("ResearchRun needs a frozen ResearchPlan for evidence retrieval.")
        queries = plan_evidence_queries(plan)
    if not 0 <= sub_question_index < len(queries):
        raise ValueError("Sub-question index is outside the frozen ResearchPlan.")
    return search_paper_chunks(
        run_id, paper_id, queries[sub_question_index], top_k=top_k,
        encoder=encoder, session_factory=factory,
    )
