"""Acquire only screened OA PDFs and persist auditable status and versions."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tracescholar.config import Settings, get_settings
from tracescholar.database import get_session_factory, session_scope
from tracescholar.fulltext.fetcher import DocumentFetcher, FetchError, FetchedPDF
from tracescholar.fulltext.locator import (
    FullTextLocation, FullTextLocator, FullTextPaper, LocateError, OpenAlexOALocator,
)
from tracescholar.models import (
    FullTextAcquisition, Paper, PaperVersion, ResearchRun, ScreeningResult,
    SearchQuery, SearchResult,
)
from tracescholar.repositories import load_research_plan
from tracescholar.screening.service import _paper_snapshot


@dataclass(frozen=True, slots=True)
class AcquisitionSummary:
    run_id: uuid.UUID
    candidate_papers: int
    full_text_available: int
    downloaded: int
    already_cached: int
    unavailable: int
    failed: int
    pending: int
    stale_screening: int


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _stored_path(settings: Settings, content_hash: str) -> tuple[Path, str]:
    relative = Path("fulltext") / "sha256" / content_hash[:2] / f"{content_hash}.pdf"
    return settings.data_dir.resolve() / relative, relative.as_posix()


def _valid_cache(version: PaperVersion, settings: Settings) -> bool:
    """A cached success is usable only if its local bytes still match the hash."""
    root = settings.data_dir.resolve()
    try:
        path = (root / version.storage_path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            return False
        if path.stat().st_size != version.content_bytes or path.stat().st_size > settings.fulltext_max_bytes:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest() == version.content_hash
    except OSError:
        return False


def _store_pdf(settings: Settings, document: FetchedPDF) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", document.content_hash) \
            or hashlib.sha256(document.content).hexdigest() != document.content_hash:
        raise FetchError("hash_mismatch", "Fetched PDF hash does not match its bytes.")
    if len(document.content) > settings.fulltext_max_bytes:
        raise FetchError("too_large", "Fetched PDF exceeds the configured maximum size.")
    path, relative = _stored_path(settings, document.content_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.is_file():
        if path.stat().st_size == len(document.content):
            with path.open("rb") as handle:
                if hashlib.sha256(handle.read()).hexdigest() == document.content_hash:
                    return relative
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".download-", delete=False) as handle:
            temporary = handle.name
            handle.write(document.content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return relative


def _state(
    session: Session, *, run_id: uuid.UUID, paper_id: uuid.UUID, status: str,
    location: FullTextLocation | None = None, version: PaperVersion | None = None,
    failure_code: str | None = None, failure_detail: str | None = None,
    retry_after: datetime | None = None, increment_attempt: bool = False,
) -> FullTextAcquisition:
    row = session.scalar(select(FullTextAcquisition).where(
        FullTextAcquisition.run_id == run_id, FullTextAcquisition.paper_id == paper_id
    ))
    if row is None:
        row = FullTextAcquisition(run_id=run_id, paper_id=paper_id, attempt_count=0)
        session.add(row)
    row.status = status
    row.paper_version_id = version.id if version is not None else None
    row.source_url = location.url if location is not None else None
    row.source_name = location.source_name if location is not None else None
    row.license = location.license if location is not None else None
    row.failure_code = failure_code
    row.failure_detail = failure_detail
    row.retry_after = retry_after
    row.attempted_at = _now()
    if increment_attempt:
        row.attempt_count += 1
    session.flush()
    return row


def _ensure_version(
    session: Session, *, paper_id: uuid.UUID, content_hash: str,
    storage_path: str, content_bytes: int, location: FullTextLocation,
    retrieved_at: datetime | None = None,
) -> PaperVersion:
    version = session.scalar(select(PaperVersion).where(
        PaperVersion.paper_id == paper_id, PaperVersion.content_hash == content_hash
    ))
    if version is None:
        version = PaperVersion(
            paper_id=paper_id, content_hash=content_hash, storage_path=storage_path,
            content_bytes=content_bytes, source_url=location.url,
            source_name=location.source_name, license=location.license,
            version_label=location.version_label, retrieved_at=retrieved_at or _now(),
        )
        session.add(version)
        session.flush()
    return version


def _existing_version(session: Session, paper_id: uuid.UUID, settings: Settings) -> PaperVersion | None:
    versions = session.scalars(select(PaperVersion).where(
        PaperVersion.paper_id == paper_id
    ).order_by(PaperVersion.retrieved_at.desc()))
    return next((version for version in versions if _valid_cache(version, settings)), None)


def _shared_url_version(
    session: Session, location: FullTextLocation, settings: Settings
) -> PaperVersion | None:
    versions = session.scalars(select(PaperVersion).where(
        PaperVersion.source_url == location.url
    ).order_by(PaperVersion.retrieved_at.desc()))
    return next((version for version in versions if _valid_cache(version, settings)), None)


def _openalex_ids(session: Session, run_id: uuid.UUID, paper_id: uuid.UUID) -> tuple[str, ...]:
    rows = session.scalars(
        select(SearchResult.source_record_id)
        .join(SearchQuery, SearchQuery.id == SearchResult.search_query_id)
        .where(SearchQuery.run_id == run_id, SearchQuery.source == "openalex",
               SearchResult.paper_id == paper_id)
        .distinct()
    )
    return tuple(value for value in rows if value)


def _candidate_papers(session: Session, run_id: uuid.UUID) -> tuple[list[FullTextPaper], int]:
    plan = load_research_plan(session, run_id)
    if plan is None:
        raise ValueError("ResearchRun needs a frozen ResearchPlan before full-text acquisition.")
    records = list(session.scalars(select(ScreeningResult).where(
        ScreeningResult.run_id == run_id,
        ScreeningResult.label.in_(("include", "maybe")),
    ).order_by(ScreeningResult.paper_id)))
    candidates: list[FullTextPaper] = []
    stale = 0
    plan_snapshot = plan.model_dump(mode="json")
    discovered_ids = set(session.scalars(
        select(SearchResult.paper_id)
        .join(SearchQuery, SearchQuery.id == SearchResult.search_query_id)
        .where(SearchQuery.run_id == run_id).distinct()
    ))
    for result in records:
        paper = session.get(Paper, result.paper_id)
        if paper is None or paper.id not in discovered_ids \
                or not isinstance(result.input_snapshot, dict) \
                or result.input_snapshot.get("paper") != _paper_snapshot(paper) \
                or result.input_snapshot.get("plan") != plan_snapshot:
            stale += 1
            continue
        candidates.append(FullTextPaper(
            id=paper.id, title=paper.title, doi=paper.doi, arxiv_id=paper.arxiv_id,
            openalex_ids=_openalex_ids(session, run_id, paper.id),
        ))
    return candidates, stale


def acquire_fulltext(
    run_id: uuid.UUID, *, locator: FullTextLocator | None = None,
    fetcher: DocumentFetcher | None = None, settings: Settings | None = None,
    session_factory: sessionmaker[Session] | None = None,
    limit: int | None = None, retry_unavailable: bool = False,
) -> AcquisitionSummary:
    """Locate and fetch screened OA PDFs; one paper's failure does not stop the run."""
    if limit is not None and not 1 <= limit <= 1000:
        raise ValueError("Acquisition limit must be between 1 and 1000.")
    active_settings = settings or get_settings()
    active_factory = session_factory or get_session_factory()
    with active_factory() as session:
        if session.get(ResearchRun, run_id) is None:
            raise LookupError(f"ResearchRun {run_id} does not exist.")
        candidates, stale = _candidate_papers(session, run_id)

    active_locator = locator or OpenAlexOALocator(settings=active_settings)
    active_fetcher = fetcher or DocumentFetcher(settings=active_settings)
    downloaded = cached = processed = 0
    candidate_ids = [paper.id for paper in candidates]
    for paper in candidates:
        with active_factory() as session:
            row = session.scalar(select(FullTextAcquisition).where(
                FullTextAcquisition.run_id == run_id,
                FullTextAcquisition.paper_id == paper.id,
            ))
            if row is not None and row.status in {"downloaded", "cached"} \
                    and row.paper_version is not None and _valid_cache(row.paper_version, active_settings):
                cached += 1
                continue
            if row is not None and row.status == "unavailable" and not retry_unavailable \
                    and row.retry_after is not None and _aware(row.retry_after) > _now():
                continue
            version = _existing_version(session, paper.id, active_settings)
            if version is not None:
                location = FullTextLocation(version.source_url, version.source_name,
                                            version.license, version.version_label)
                with session_scope(active_factory) as write_session:
                    saved_version = write_session.get(PaperVersion, version.id)
                    _state(write_session, run_id=run_id, paper_id=paper.id, status="cached",
                           location=location, version=saved_version)
                cached += 1
                continue
        if limit is not None and processed >= limit:
            continue
        processed += 1
        try:
            locations = active_locator.locate(paper)
        except LocateError as error:
            with session_scope(active_factory) as session:
                _state(session, run_id=run_id, paper_id=paper.id, status="failed",
                       failure_code="locator_error", failure_detail=str(error), increment_attempt=True)
            continue
        if not locations:
            with session_scope(active_factory) as session:
                _state(
                    session, run_id=run_id, paper_id=paper.id, status="unavailable",
                    failure_detail="No verified open-access PDF location was found.",
                    retry_after=_now() + timedelta(
                        hours=active_settings.fulltext_unavailable_retry_hours
                    ), increment_attempt=True,
                )
            continue
        with session_scope(active_factory) as session:
            _state(session, run_id=run_id, paper_id=paper.id, status="located",
                   location=locations[0], increment_attempt=True)
        errors: list[str] = []
        last_error: FetchError | None = None
        last_location: FullTextLocation | None = None
        succeeded = False
        for location in locations:
            last_location = location
            with active_factory() as session:
                shared = _shared_url_version(session, location, active_settings)
                if shared is not None:
                    shared_hash = shared.content_hash
                    shared_path = shared.storage_path
                    shared_bytes = shared.content_bytes
                    shared_retrieved = shared.retrieved_at
                else:
                    shared_hash = None
            if shared_hash is not None:
                with session_scope(active_factory) as session:
                    version = _ensure_version(
                        session, paper_id=paper.id, content_hash=shared_hash,
                        storage_path=shared_path, content_bytes=shared_bytes,
                        location=location, retrieved_at=shared_retrieved,
                    )
                    _state(session, run_id=run_id, paper_id=paper.id, status="cached",
                           location=location, version=version)
                cached += 1
                succeeded = True
                break
            try:
                document = active_fetcher.fetch(location)
                relative_path = _store_pdf(active_settings, document)
            except FetchError as error:
                last_error = error
                errors.append(f"{location.source_name}: {error.code}")
                continue
            except OSError as error:
                last_error = FetchError("storage_error", "Could not save the PDF locally.")
                errors.append(f"{location.source_name}: storage_error")
                continue
            with session_scope(active_factory) as session:
                version = _ensure_version(
                    session, paper_id=paper.id, content_hash=document.content_hash,
                    storage_path=relative_path, content_bytes=len(document.content),
                    location=location,
                )
                _state(session, run_id=run_id, paper_id=paper.id, status="downloaded",
                       location=location, version=version)
            downloaded += 1
            succeeded = True
            break
        if not succeeded:
            with session_scope(active_factory) as session:
                _state(
                    session, run_id=run_id, paper_id=paper.id, status="failed",
                    location=last_location,
                    failure_code=last_error.code if last_error else "fetch_failed",
                    failure_detail="; ".join(errors)[:1000] or "No candidate PDF could be fetched.",
                )

    with active_factory() as session:
        rows = list(session.scalars(select(FullTextAcquisition).where(
            FullTextAcquisition.run_id == run_id,
            FullTextAcquisition.paper_id.in_(candidate_ids),
        ))) if candidate_ids else []
    unavailable = sum(row.status == "unavailable" for row in rows)
    failed = sum(row.status == "failed" for row in rows)
    full_text_available = sum(
        row.status in {"downloaded", "cached", "located"}
        or (row.status == "failed" and row.source_url is not None)
        for row in rows
    )
    pending = len(candidates) - sum(row.status in {
        "downloaded", "cached", "unavailable", "failed"
    } for row in rows)
    return AcquisitionSummary(
        run_id=run_id, candidate_papers=len(candidates),
        full_text_available=full_text_available, downloaded=downloaded,
        already_cached=cached, unavailable=unavailable, failed=failed,
        pending=pending, stale_screening=stale,
    )
