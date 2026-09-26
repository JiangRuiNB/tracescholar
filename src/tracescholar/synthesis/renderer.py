"""Deterministically render a saved structured synthesis as Markdown."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import quote, urlsplit

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tracescholar.database import get_session_factory
from tracescholar.models import EvidenceSpan, PaperVersion, StudyPaper, SynthesisDraft
from tracescholar.synthesis.schemas import SynthesisDocument, SYNTHESIS_SCHEMA_VERSION
from tracescholar.synthesis.validation import SynthesisValidationError, validate_synthesis_references


@dataclass(frozen=True, slots=True)
class CitationSource:
    """Citation metadata loaded only from persisted evidence and paper records."""

    evidence_id: uuid.UUID
    claim_id: uuid.UUID
    claim_statement: str
    study_id: uuid.UUID
    paper_version_id: uuid.UUID
    paper_title: str
    authors: tuple[str, ...]
    year: int | None
    venue: str | None
    doi: str | None
    arxiv_id: str | None
    version_label: str | None
    source_url: str
    page: int
    section: str
    quote: str
    stance: str


def _escape_markdown(value: str) -> str:
    """Keep model text and paper metadata from creating Markdown structure."""
    compact = " ".join(value.split())
    compact = compact.replace("&", "&amp;").replace("<", "&lt;")
    escaped = re.sub(r"([\\`*_{}\[\]()#+!|>~])", r"\\\1", compact)
    if re.match(r"^(?:[-+]\s|\d+[.)]\s|---)", escaped):
        escaped = "\\" + escaped
    return escaped


def _safe_url(value: str) -> str | None:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or \
            parsed.username or parsed.password:
        return None
    return quote(value, safe=":/?&=%#@+-._~")


def _reference_note(source: CitationSource) -> str:
    authors = ", ".join(source.authors[:3])
    if len(source.authors) > 3:
        authors += " et al."
    prefix = f"{_escape_markdown(authors)}. " if authors else ""
    year = str(source.year) if source.year is not None else "n.d."
    title = _escape_markdown(source.paper_title)
    parts = [f"{prefix}({year}). *{title}*."]
    if source.venue:
        parts.append(f"{_escape_markdown(source.venue)}.")
    if source.version_label:
        parts.append(f"Version: {_escape_markdown(source.version_label)}.")
    parts.append(f"PaperVersion `{source.paper_version_id}`.")
    parts.append(f"PDF p. {source.page}, section {_escape_markdown(source.section)}.")
    parts.append(f"Stance: {source.stance}.")
    if source.doi:
        doi_url = _safe_url(f"https://doi.org/{source.doi}")
        if doi_url:
            parts.append(f"[DOI]({doi_url}).")
    elif source.arxiv_id:
        arxiv_url = _safe_url(f"https://arxiv.org/abs/{source.arxiv_id}")
        if arxiv_url:
            parts.append(f"[arXiv]({arxiv_url}).")
    pdf_url = _safe_url(source.source_url)
    if pdf_url:
        parts.append(f"[Source PDF]({pdf_url}).")
    parts.append(f"Claim `{source.claim_id}`: {_escape_markdown(source.claim_statement)}.")
    parts.append(f"EvidenceSpan `{source.evidence_id}`; Study `{source.study_id}`.")
    parts.append(f"Quoted passage: “{_escape_markdown(source.quote)}”")
    return " ".join(parts)


def render_markdown(
    document: SynthesisDocument, sources: Mapping[uuid.UUID, CitationSource],
) -> str:
    """Render text and numbered footnotes from DB-backed citation records only."""
    numbers: dict[uuid.UUID, int] = {}
    lines = [f"# {_escape_markdown(document.title)}", "",
             f"Research question: {_escape_markdown(document.research_question)}", ""]
    for section in document.sections:
        lines.extend((f"## {_escape_markdown(section.heading)}", ""))
        for paragraph in section.paragraphs:
            sentences = []
            for sentence in paragraph.sentences:
                markers = []
                for evidence_id in sentence.evidence_ids:
                    if evidence_id not in sources:
                        raise SynthesisValidationError(
                            "Synthesis cited an EvidenceSpan missing from the database")
                    numbers.setdefault(evidence_id, len(numbers) + 1)
                    markers.append(f"[^e{numbers[evidence_id]}]")
                sentences.append(_escape_markdown(sentence.text) + "".join(markers))
            lines.extend((" ".join(sentences), ""))
    if numbers:
        lines.extend(("## References", ""))
        for evidence_id, number in numbers.items():
            lines.append(f"[^e{number}]: {_reference_note(sources[evidence_id])}")
    return "\n".join(lines).rstrip() + "\n"


def render_synthesis(
    run_id: uuid.UUID, *, draft_id: uuid.UUID | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> str:
    """Load a saved draft and its cited EvidenceSpans, then render in memory."""
    factory = session_factory or get_session_factory()
    with factory() as session:
        statement = select(SynthesisDraft).where(
            SynthesisDraft.run_id == run_id, SynthesisDraft.status == "success")
        if draft_id is not None:
            statement = statement.where(SynthesisDraft.id == draft_id)
        draft = session.scalar(statement.order_by(
            SynthesisDraft.created_at.desc(), SynthesisDraft.id.desc()))
        if draft is None or draft.document_json is None:
            raise LookupError("No successful structured synthesis exists for this ResearchRun")
        if draft.schema_version != SYNTHESIS_SCHEMA_VERSION:
            raise ValueError("Unsupported SynthesisDocument schema version")
        document = SynthesisDocument.model_validate_json(json.dumps(draft.document_json))
        snapshot = draft.input_snapshot
        validate_synthesis_references(document, snapshot["ledger"],
                                      snapshot["research_question"])
        cited_ids = {evidence_id for section in document.sections
                     for paragraph in section.paragraphs
                     for sentence in paragraph.sentences
                     for evidence_id in sentence.evidence_ids}
        sources: dict[uuid.UUID, CitationSource] = {}
        if cited_ids:
            for span in session.scalars(select(EvidenceSpan).where(
                EvidenceSpan.id.in_(cited_ids))):
                if span.extraction.claim.generation_id != draft.claim_generation_id or \
                        span.study_id != span.extraction.study_id or \
                        span.paper_version_id != span.extraction.paper_version_id:
                    raise SynthesisValidationError("Stored citation has inconsistent provenance")
                record = session.get(PaperVersion, span.paper_version_id)
                if record is None or not session.scalar(select(StudyPaper.id).where(
                    StudyPaper.study_id == span.study_id,
                    StudyPaper.paper_id == record.paper_id)):
                    raise SynthesisValidationError("Citation version is outside its Study")
                paper = record.paper
                sources[span.id] = CitationSource(
                    evidence_id=span.id, claim_id=span.extraction.claim.id,
                    claim_statement=span.extraction.claim.statement,
                    study_id=span.study_id,
                    paper_version_id=record.id, paper_title=paper.title,
                    authors=tuple(paper.authors or []), year=paper.year,
                    venue=paper.venue, doi=paper.doi, arxiv_id=paper.arxiv_id,
                    version_label=record.version_label, source_url=record.source_url,
                    page=span.page_number, section=span.section,
                    quote=span.quote, stance=span.stance,
                )
        if set(sources) != cited_ids:
            raise SynthesisValidationError("A cited EvidenceSpan is missing from the database")
        return render_markdown(document, sources)
