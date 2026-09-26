"""Isolated, idempotent parsing of acquired PDF versions in a research run."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from tracescholar.config import Settings, get_settings
from tracescholar.database import get_session_factory, session_scope
from tracescholar.models import (
    Chunk, FullTextAcquisition, Paper, PaperVersion, ParsedPage, PdfParseRecord, ResearchRun,
)
from tracescholar.pdf_parsing.parser import PDFParser, ParseError, ParsedDocument


@dataclass(frozen=True, slots=True)
class ParseFailure:
    paper_version_id: uuid.UUID
    code: str
    detail: str


@dataclass(frozen=True, slots=True)
class ParseSummary:
    run_id: uuid.UUID
    paper_versions: int
    parsed_successfully: int
    failed: int
    newly_parsed: int
    skipped_existing: int
    total_pages: int
    total_chunks: int
    failures: tuple[ParseFailure, ...]


@dataclass(frozen=True, slots=True)
class ChunkProvenance:
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
    locator: dict
    text: str


def get_chunk_provenance(
    chunk_id: uuid.UUID, *, session_factory: sessionmaker[Session] | None = None,
) -> ChunkProvenance:
    """Resolve a chunk to its paper, immutable PDF version, and exact locator."""
    factory = session_factory or get_session_factory()
    with factory() as session:
        row = session.execute(
            select(Chunk, PaperVersion, Paper)
            .join(PaperVersion, Chunk.paper_version_id == PaperVersion.id)
            .join(Paper, PaperVersion.paper_id == Paper.id)
            .where(Chunk.id == chunk_id)
        ).one_or_none()
    if row is None:
        raise LookupError(f"Chunk {chunk_id} does not exist.")
    chunk, version, paper = row
    return ChunkProvenance(
        chunk.id, paper.id, paper.title, version.id, version.storage_path,
        version.content_hash, chunk.page_start, chunk.page_end, chunk.section,
        chunk.ordinal, chunk.locator, chunk.text,
    )


def _pdf_path(version: PaperVersion, data_dir: Path) -> Path:
    root = data_dir.resolve()
    path = (root / version.storage_path).resolve()
    if not path.is_relative_to(root):
        raise ParseError("unsafe_path", "PDF storage path escapes the configured data directory.")
    if not path.is_file():
        raise ParseError("missing_file", "Cached PDF file is missing.")
    if path.stat().st_size != version.content_bytes:
        raise ParseError("size_mismatch", "Cached PDF size differs from PaperVersion metadata.")
    with path.open("rb") as handle:
        actual_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual_hash != version.content_hash:
        raise ParseError("hash_mismatch", "Cached PDF content differs from PaperVersion hash.")
    return path


def _record(session: Session, version: PaperVersion) -> PdfParseRecord:
    row = session.scalar(select(PdfParseRecord).where(PdfParseRecord.paper_version_id == version.id))
    if row is None:
        row = PdfParseRecord(paper_version_id=version.id, status="failed",
                             parser_version="", input_hash=version.content_hash,
                             page_count=0, text_page_count=0, chunk_count=0,
                             total_char_count=0, quality_flags=[])
        session.add(row)
    return row


def _save_success(session: Session, version: PaperVersion, parser: PDFParser,
                  document: ParsedDocument) -> None:
    session.execute(delete(Chunk).where(Chunk.paper_version_id == version.id))
    session.execute(delete(ParsedPage).where(ParsedPage.paper_version_id == version.id))
    for page in document.pages:
        session.add(ParsedPage(
            paper_version_id=version.id, page_number=page.page_number,
            text=page.text, char_count=len(page.text), width=page.width,
            height=page.height, column_count=page.column_count,
            quality_flags=list(page.quality_flags),
        ))
    page_texts = {page.page_number: page.text for page in document.pages}
    for chunk in document.chunks:
        page_text = page_texts[chunk.page_start]
        if page_text[chunk.locator["char_start"]:chunk.locator["char_end"]] != chunk.text:
            raise ValueError("Parser emitted a chunk that does not match its page locator.")
        session.add(Chunk(
            paper_version_id=version.id, ordinal=chunk.ordinal, text=chunk.text,
            page_start=chunk.page_start, page_end=chunk.page_end,
            section=chunk.section, document_char_start=chunk.document_char_start,
            document_char_end=chunk.document_char_end, locator=chunk.locator,
            char_count=len(chunk.text), token_count=chunk.token_count,
            parser_version=parser.version,
        ))
    row = _record(session, version)
    row.status = "success"
    row.parser_version = parser.version
    row.input_hash = version.content_hash
    row.page_count = len(document.pages)
    row.text_page_count = sum(bool(page.text) for page in document.pages)
    row.chunk_count = len(document.chunks)
    row.total_char_count = sum(len(page.text) for page in document.pages)
    row.quality_flags = list(document.quality_flags)
    row.failure_code = None
    row.failure_detail = None
    row.attempted_at = datetime.now(timezone.utc)


def _save_failure(session: Session, version: PaperVersion, parser: PDFParser,
                  error: ParseError) -> None:
    session.execute(delete(Chunk).where(Chunk.paper_version_id == version.id))
    session.execute(delete(ParsedPage).where(ParsedPage.paper_version_id == version.id))
    row = _record(session, version)
    row.status = "failed"
    row.parser_version = parser.version
    row.input_hash = version.content_hash
    row.page_count = row.text_page_count = row.chunk_count = row.total_char_count = 0
    row.quality_flags = []
    row.failure_code = error.code
    row.failure_detail = str(error)[:2000]
    row.attempted_at = datetime.now(timezone.utc)


def parse_research_run(
    run_id: uuid.UUID, *, parser: PDFParser | None = None,
    settings: Settings | None = None,
    session_factory: sessionmaker[Session] | None = None,
    limit: int | None = None,
) -> ParseSummary:
    """Parse each distinct acquired version; failure of one does not stop the rest."""
    if limit is not None and not 1 <= limit <= 1000:
        raise ValueError("Parse limit must be between 1 and 1000.")
    active_settings = settings or get_settings()
    active_factory = session_factory or get_session_factory()
    active_parser = parser or PDFParser()
    with active_factory() as session:
        if session.get(ResearchRun, run_id) is None:
            raise LookupError(f"ResearchRun {run_id} does not exist.")
        version_ids = list(session.scalars(
            select(FullTextAcquisition.paper_version_id)
            .where(FullTextAcquisition.run_id == run_id,
                   FullTextAcquisition.status.in_(("downloaded", "cached")),
                   FullTextAcquisition.paper_version_id.is_not(None))
            .distinct().order_by(FullTextAcquisition.paper_version_id)
        ))

    new = skipped = attempted = 0
    failures: list[ParseFailure] = []
    for version_id in version_ids:
        with active_factory() as session:
            version = session.get(PaperVersion, version_id)
            if version is None:
                failures.append(ParseFailure(version_id, "missing_version", "PaperVersion is missing."))
                continue
            record = session.scalar(select(PdfParseRecord).where(
                PdfParseRecord.paper_version_id == version_id
            ))
            if limit is not None and attempted >= limit \
                    and (record is None or record.status != "success"):
                continue
            try:
                path = _pdf_path(version, active_settings.data_dir)
            except (ParseError, OSError) as error:
                if limit is not None and attempted >= limit:
                    continue
                attempted += 1
                issue = error if isinstance(error, ParseError) else ParseError("file_error", str(error))
                with session_scope(active_factory) as write_session:
                    _save_failure(write_session, write_session.get(PaperVersion, version_id), active_parser, issue)
                failures.append(ParseFailure(version_id, issue.code, str(issue)))
                continue
            if record is not None and record.status == "success" \
                    and record.parser_version == active_parser.version \
                    and record.input_hash == version.content_hash \
                    and session.scalar(select(func.count()).select_from(Chunk).where(
                        Chunk.paper_version_id == version_id)) == record.chunk_count \
                    and session.scalar(select(func.count()).select_from(ParsedPage).where(
                        ParsedPage.paper_version_id == version_id)) == record.page_count:
                skipped += 1
                continue
        if limit is not None and attempted >= limit:
            continue
        attempted += 1
        try:
            document = active_parser.parse(path)
        except Exception as error:
            issue = error if isinstance(error, ParseError) else ParseError(
                "unexpected_parse_error", f"Unexpected PDF parse failure: {error}"
            )
            with session_scope(active_factory) as session:
                _save_failure(session, session.get(PaperVersion, version_id), active_parser, issue)
            failures.append(ParseFailure(version_id, issue.code, str(issue)))
            continue
        try:
            with session_scope(active_factory) as session:
                _save_success(session, session.get(PaperVersion, version_id), active_parser, document)
            new += 1
        except ValueError as error:
            issue = ParseError("invalid_output", str(error))
            with session_scope(active_factory) as session:
                _save_failure(session, session.get(PaperVersion, version_id), active_parser, issue)
            failures.append(ParseFailure(version_id, issue.code, str(issue)))

    with active_factory() as session:
        records = list(session.scalars(select(PdfParseRecord).where(
            PdfParseRecord.paper_version_id.in_(version_ids)
        ))) if version_ids else []
    successful = [row for row in records if row.status == "success"]
    failed_ids = {row.paper_version_id for row in records if row.status == "failed"}
    failed_ids.update(failure.paper_version_id for failure in failures)
    return ParseSummary(
        run_id, len(version_ids), len(successful), len(failed_ids), new, skipped,
        sum(row.page_count for row in successful),
        sum(row.chunk_count for row in successful), tuple(failures),
    )
