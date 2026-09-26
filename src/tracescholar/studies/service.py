"""Conservative, auditable paper-version linkage for an evidence corpus.

Bibliographic similarity can establish a *candidate* relationship, but a differing
PDF hash says nothing about whether experiments changed. That judgment is stored
separately and never inferred from a filename or hash.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import combinations

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, selectinload, sessionmaker

from tracescholar.database import session_scope
from tracescholar.models import (
    CanonicalStudy, Claim, EvidenceExtraction, EvidenceSpan, FullTextScreeningResult,
    Paper, PaperVersion, ResearchRun,
    StudyLinkCandidate, StudyPaper, StudyRunSelection, StudyVersionComparison,
)

RULE_VERSION = "study-link-v1"
SELECTION_POLICY = "published-complete-v1"


@dataclass(frozen=True)
class StudySummary:
    run_id: uuid.UUID
    paper_records: int
    paper_versions: int
    potential_version_groups: int
    canonical_studies: int
    linked_records: int
    collapsed_surplus: int
    unresolved: int
    result_pairs_not_assessed: int
    newly_linked: int
    skipped_unchanged: int


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     default=str).encode()).hexdigest()


def _norm(value: str | None) -> str:
    return " ".join(re.findall(r"\w+", (value or "").casefold()))


def _authors(paper: Paper) -> set[str]:
    return {_norm(str(name)) for name in (paper.authors or []) if _norm(str(name))}


def _overlap(a: set[str], b: set[str]) -> float:
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


def _arxiv(paper: Paper) -> str | None:
    if paper.arxiv_id:
        return paper.arxiv_id.casefold().removeprefix("arxiv:")
    doi = (paper.doi or "").casefold()
    return doi.removeprefix("10.48550/arxiv.") if doi.startswith("10.48550/arxiv.") else None


def _role(paper: Paper) -> str:
    venue = _norm(paper.venue)
    doi = (paper.doi or "").casefold()
    if _arxiv(paper) and (not doi or doi.startswith("10.48550/arxiv.")):
        return "preprint"
    if any(term in venue for term in ("journal", "transactions", "review", "letters")):
        return "journal"
    if paper.venue or doi.startswith("10.18653/"):
        return "conference"
    return "other"


def _pair_decision(a: Paper, b: Paper) -> tuple[str, float, str, dict]:
    title_a, title_b = _norm(a.title), _norm(b.title)
    title_score = SequenceMatcher(None, title_a, title_b).ratio()
    author_score = _overlap(_authors(a), _authors(b))
    abstract_score = SequenceMatcher(None, _norm(a.abstract), _norm(b.abstract)).ratio() if a.abstract and b.abstract else 0.0
    year_distance = abs(a.year - b.year) if a.year and b.year else None
    shared_hash = bool({v.content_hash for v in a.versions} & {v.content_hash for v in b.versions})
    shared_arxiv = bool(_arxiv(a) and _arxiv(a) == _arxiv(b))
    signals = {"title_similarity": round(title_score, 4), "author_overlap": round(author_score, 4),
               "abstract_similarity": round(abstract_score, 4), "year_distance": year_distance,
               "shared_pdf_hash": shared_hash, "shared_arxiv_id": shared_arxiv}
    if shared_hash or shared_arxiv:
        return "confirmed", 1.0, "Shared PDF content or arXiv identifier", signals
    close_year = year_distance is None or year_distance <= 2
    if close_year and title_a == title_b and author_score >= 0.5 and len(_authors(a)) >= 1:
        return "confirmed", 0.98, "Identical normalized title and overlapping authors", signals
    if close_year and title_score >= 0.82 and author_score >= 0.75 and abstract_score >= 0.75:
        return "confirmed", 0.94, "Near-identical title, authors, and abstract", signals
    if close_year and title_score >= 0.78 and (author_score >= 0.3 or not _authors(a) or not _authors(b)):
        return "unresolved", 0.65, "Similar bibliographic records need manual verification", signals
    return "rejected", 0.0, "Insufficient same-study evidence", signals


def _paper_rank(paper: Paper) -> tuple:
    role = _role(paper)
    return ({"journal": 3, "conference": 2, "other": 1, "preprint": 0}[role],
            bool(paper.doi and not paper.doi.casefold().startswith("10.48550/arxiv.")),
            len(paper.abstract or ""), len(paper.versions), str(paper.id))


def _version_rank(version: PaperVersion, role: str) -> tuple:
    parse = version.parse_record
    good_parse = bool(parse and parse.status == "success")
    coverage_good = good_parse and "low_page_coverage" not in (parse.quality_flags or [])
    return (int(coverage_good), int(good_parse),
            {"journal": 3, "conference": 2, "other": 1, "preprint": 0}[role],
            (parse.total_char_count or 0) if parse else 0, version.content_bytes,
            str(version.id))


def _run_papers(session: Session, run_id: uuid.UUID) -> list[Paper]:
    paper_ids = session.scalars(select(FullTextScreeningResult.paper_id).where(
        FullTextScreeningResult.run_id == run_id,
        FullTextScreeningResult.status == "success",
        FullTextScreeningResult.label.in_(("include", "uncertain")),
    )).all()
    if not paper_ids:
        return []
    return list(session.scalars(select(Paper).where(Paper.id.in_(paper_ids)).options(
        selectinload(Paper.versions).selectinload(PaperVersion.parse_record),
        selectinload(Paper.study_membership),
    )).all())


def normalize_studies(
    run_id: uuid.UUID, *, session_factory: sessionmaker[Session] | None = None,
) -> StudySummary:
    """Link high-confidence versions and retain ambiguous pairs for review.

    The operation is transactional and repeatable; it never deletes Paper or PDF rows.
    Existing cross-study memberships are not silently merged.
    """
    with session_scope(session_factory) as session:
        if session.get(ResearchRun, run_id) is None:
            raise LookupError(f"ResearchRun {run_id} does not exist")
        papers = _run_papers(session, run_id)
        by_id = {paper.id: paper for paper in papers}
        existing = {(row.paper_a_id, row.paper_b_id): row for row in session.scalars(
            select(StudyLinkCandidate).where(StudyLinkCandidate.paper_a_id.in_(by_id),
                                              StudyLinkCandidate.paper_b_id.in_(by_id))
        )}
        confirmed: list[tuple[uuid.UUID, uuid.UUID]] = []
        skipped = 0
        unresolved = 0
        for a, b in combinations(sorted(papers, key=lambda p: str(p.id)), 2):
            status, confidence, reason, signals = _pair_decision(a, b)
            key = (a.id, b.id)
            row = existing.get(key)
            if status == "rejected" and row is None:
                continue
            fingerprint = _digest({"a": [a.title, a.authors, a.abstract, a.doi, a.arxiv_id, a.year,
                                          sorted(v.content_hash for v in a.versions)],
                                   "b": [b.title, b.authors, b.abstract, b.doi, b.arxiv_id, b.year,
                                          sorted(v.content_hash for v in b.versions)],
                                   "rule": RULE_VERSION})
            if row is None:
                row = StudyLinkCandidate(paper_a_id=a.id, paper_b_id=b.id, status=status,
                                         confidence=confidence, match_reason=reason, signals=signals,
                                         input_hash=fingerprint, rule_version=RULE_VERSION,
                                         decision_source="rule")
                session.add(row)
            elif row.decision_source == "rule" and row.input_hash != fingerprint:
                row.status, row.confidence, row.match_reason = status, confidence, reason
                row.signals, row.input_hash, row.rule_version = signals, fingerprint, RULE_VERSION
            else:
                skipped += 1
            if row.status == "confirmed":
                confirmed.append(key)
            elif row.status == "unresolved":
                unresolved += 1

        parent = {pid: pid for pid in by_id}

        def find(pid: uuid.UUID) -> uuid.UUID:
            while parent[pid] != pid:
                parent[pid] = parent[parent[pid]]
                pid = parent[pid]
            return pid

        for a_id, b_id in confirmed:
            root_a, root_b = find(a_id), find(b_id)
            if root_a != root_b:
                parent[root_b] = root_a
        groups: dict[uuid.UUID, list[Paper]] = defaultdict(list)
        for paper in papers:
            groups[find(paper.id)].append(paper)

        newly_linked = 0
        for members in groups.values():
            existing_studies = {p.study_membership.study_id for p in members if p.study_membership}
            if len(existing_studies) > 1:
                # Earlier runs may have formed separate singleton studies before a
                # later source revealed a strong version link. Re-parent only the
                # grouping rows; bibliographic records and PDFs remain untouched.
                existing_rows = [session.get(CanonicalStudy, sid) for sid in existing_studies]
                target = max(existing_rows, key=lambda row: _paper_rank(row.canonical_paper))
                for source in existing_rows:
                    if source.id == target.id:
                        continue
                    source_selections = list(session.scalars(select(StudyRunSelection).where(
                        StudyRunSelection.study_id == source.id)))
                    for item in source_selections:
                        target_selection = session.scalar(select(StudyRunSelection).where(
                            StudyRunSelection.run_id == item.run_id,
                            StudyRunSelection.study_id == target.id))
                        if target_selection:
                            session.delete(item)
                        else:
                            item.study_id = target.id
                    session.flush()
                    session.execute(update(StudyVersionComparison).where(
                        StudyVersionComparison.study_id == source.id).values(study_id=target.id))
                    session.execute(update(Claim).where(
                        Claim.basis_study_id == source.id).values(basis_study_id=target.id))
                    session.execute(update(EvidenceExtraction).where(
                        EvidenceExtraction.study_id == source.id).values(study_id=target.id))
                    session.execute(update(EvidenceSpan).where(
                        EvidenceSpan.study_id == source.id).values(study_id=target.id))
                    session.execute(update(StudyPaper).where(
                        StudyPaper.study_id == source.id).values(study_id=target.id))
                    session.flush()
                    session.delete(source)
                    session.flush()
                session.expire_all()
                existing_studies = {target.id}
            if existing_studies:
                study = session.get(CanonicalStudy, next(iter(existing_studies)))
                assert study is not None
                other_members = list(session.scalars(select(Paper).join(
                    StudyPaper, StudyPaper.paper_id == Paper.id,
                ).where(StudyPaper.study_id == study.id).options(selectinload(Paper.versions))))
            else:
                other_members = []
                canonical = max(members, key=_paper_rank)
                study = CanonicalStudy(canonical_paper_id=canonical.id,
                                       canonical_reason="Published/metadata-complete record preferred; PDFs remain separate")
                session.add(study)
                session.flush()
            all_members = list({p.id: p for p in [*other_members, *members]}.values())
            canonical = max(all_members, key=_paper_rank)
            if study.canonical_paper_id != canonical.id:
                study.canonical_paper_id = canonical.id
                study.canonical_reason = "Published/metadata-complete record preferred; PDFs remain separate"
            for paper in members:
                if paper.study_membership is None:
                    session.add(StudyPaper(study_id=study.id, paper_id=paper.id,
                                           publication_role=_role(paper),
                                           relationship_reason="Confirmed version link" if len(members) > 1
                                           else "Only bibliographic record for this study"))
                    newly_linked += 1
            all_versions = sorted((v for p in all_members for v in p.versions),
                                  key=lambda v: str(v.id))
            versions = sorted((v for p in members for v in p.versions), key=lambda v: str(v.id))
            comparisons = {(r.version_a_id, r.version_b_id): r for r in session.scalars(
                select(StudyVersionComparison).where(StudyVersionComparison.study_id == study.id)
            )}
            for version_a, version_b in combinations(all_versions, 2):
                if (version_a.id, version_b.id) not in comparisons:
                    same = version_a.content_hash == version_b.content_hash
                    session.add(StudyVersionComparison(
                        study_id=study.id, version_a_id=version_a.id, version_b_id=version_b.id,
                        result_relation="identical_content" if same else "not_assessed",
                        assessment_source="hash" if same else "system",
                        assessment_note="Byte-identical PDFs" if same else
                        "Different PDF bytes do not establish whether experimental results changed",
                    ))
            preferred = max(versions, key=lambda v: _version_rank(v, _role(by_id[v.paper_id]))) if versions else None
            input_hash = _digest({"policy": SELECTION_POLICY, "members": sorted(str(p.id) for p in members),
                                  "versions": [(str(v.id), v.content_hash, _version_rank(v, _role(by_id[v.paper_id])))
                                               for v in versions]})
            selection = session.scalar(select(StudyRunSelection).where(
                StudyRunSelection.run_id == run_id, StudyRunSelection.study_id == study.id))
            if selection is None:
                session.add(StudyRunSelection(
                    run_id=run_id, study_id=study.id,
                    preferred_paper_version_id=preferred.id if preferred else None,
                    selection_reason="Best parsed PDF, then published version, then completeness"
                    if preferred else "No acquired PDF version",
                    input_hash=input_hash, policy_version=SELECTION_POLICY,
                ))
            elif selection.input_hash != input_hash:
                selection.preferred_paper_version_id = preferred.id if preferred else None
                selection.selection_reason = "Best parsed PDF, then published version, then completeness" \
                    if preferred else "No acquired PDF version"
                selection.input_hash = input_hash
                selection.policy_version = SELECTION_POLICY

        session.flush()
        memberships = list(session.scalars(select(StudyPaper).where(StudyPaper.paper_id.in_(by_id))))
        study_ids = {m.study_id for m in memberships}
        counts = defaultdict(int)
        for membership in memberships:
            counts[membership.study_id] += 1
        multi = [n for n in counts.values() if n > 1]
        unassessed = session.scalar(select(func.count()).select_from(StudyVersionComparison).where(
            StudyVersionComparison.study_id.in_(study_ids),
            StudyVersionComparison.result_relation == "not_assessed",
        )) if study_ids else 0
        return StudySummary(run_id, len(papers), sum(len(p.versions) for p in papers),
                            len(multi) + unresolved, len(study_ids), sum(multi),
                            sum(n - 1 for n in multi), unresolved, unassessed or 0,
                            newly_linked, skipped)


def get_study_details(run_id: uuid.UUID, study_id: uuid.UUID, *,
                      session_factory: sessionmaker[Session] | None = None) -> dict:
    """Return one run-scoped study with all bibliographic and PDF provenance."""
    with session_scope(session_factory) as session:
        selection = session.scalar(select(StudyRunSelection).where(
            StudyRunSelection.run_id == run_id, StudyRunSelection.study_id == study_id))
        if selection is None:
            raise LookupError("Study is not selected for this ResearchRun")
        study = session.get(CanonicalStudy, study_id)
        assert study is not None
        members = list(session.scalars(select(StudyPaper).where(StudyPaper.study_id == study_id)
                                       .options(selectinload(StudyPaper.paper).selectinload(Paper.versions))))
        comparisons = list(session.scalars(select(StudyVersionComparison).where(
            StudyVersionComparison.study_id == study_id)))
        return {
            "study_id": str(study.id), "canonical_paper_id": str(study.canonical_paper_id),
            "canonical_reason": study.canonical_reason,
            "independent_study_count": 1,
            "preferred_paper_version_id": str(selection.preferred_paper_version_id)
            if selection.preferred_paper_version_id else None,
            "selection_reason": selection.selection_reason,
            "papers": [{"paper_id": str(m.paper_id), "title": m.paper.title,
                        "doi": m.paper.doi, "arxiv_id": m.paper.arxiv_id,
                        "publication_role": m.publication_role,
                        "versions": [{"version_id": str(v.id), "source_url": v.source_url,
                                      "source_name": v.source_name, "storage_path": v.storage_path,
                                      "content_hash": v.content_hash, "license": v.license,
                                      "retrieved_at": v.retrieved_at.isoformat()}
                                     for v in m.paper.versions]} for m in members],
            "result_comparisons": [{"version_a_id": str(c.version_a_id),
                                    "version_b_id": str(c.version_b_id),
                                    "result_relation": c.result_relation,
                                    "assessment_note": c.assessment_note} for c in comparisons],
            "extraction_version_ids": [str(v) for v in _extraction_versions(selection, comparisons)],
        }


def _extraction_versions(selection: StudyRunSelection,
                         comparisons: list[StudyVersionComparison]) -> list[uuid.UUID]:
    """Use the preferred PDF, plus changed-results versions for separate extraction."""
    if selection.preferred_paper_version_id is None:
        return []
    versions = {selection.preferred_paper_version_id}
    for comparison in comparisons:
        if comparison.result_relation == "changed":
            versions.update((comparison.version_a_id, comparison.version_b_id))
    return sorted(versions, key=str)


def set_version_result_relation(
    run_id: uuid.UUID, study_id: uuid.UUID, version_a_id: uuid.UUID,
    version_b_id: uuid.UUID, relation: str, note: str, *,
    session_factory: sessionmaker[Session] | None = None,
) -> None:
    """Record a human-reviewed result comparison; never infer it from PDF hashes."""
    if relation not in {"equivalent", "changed"} or not note.strip():
        raise ValueError("A supported relation and non-empty human assessment note are required")
    a, b = sorted((version_a_id, version_b_id), key=str)
    if a == b:
        raise ValueError("Two distinct PDF versions are required")
    with session_scope(session_factory) as session:
        if session.scalar(select(StudyRunSelection).where(
            StudyRunSelection.run_id == run_id, StudyRunSelection.study_id == study_id)) is None:
            raise LookupError("Study is not selected for this ResearchRun")
        comparison = session.scalar(select(StudyVersionComparison).where(
            StudyVersionComparison.study_id == study_id,
            StudyVersionComparison.version_a_id == a,
            StudyVersionComparison.version_b_id == b))
        if comparison is None:
            raise LookupError("These PDF versions are not a recorded pair in this study")
        if relation == "changed" and session.get(PaperVersion, a).content_hash == \
                session.get(PaperVersion, b).content_hash:
            raise ValueError("Byte-identical PDFs cannot have different experimental results")
        comparison.result_relation = relation
        comparison.assessment_note = note.strip()
        comparison.assessment_source = "human"


def decide_study_link(
    run_id: uuid.UUID, paper_a_id: uuid.UUID, paper_b_id: uuid.UUID,
    decision: str, reason: str, *, session_factory: sessionmaker[Session] | None = None,
) -> None:
    """Resolve an ambiguous bibliographic pair without modifying Paper records."""
    if decision not in {"confirmed", "rejected"} or not reason.strip():
        raise ValueError("Decision must be confirmed/rejected with a non-empty review reason")
    a, b = sorted((paper_a_id, paper_b_id), key=str)
    with session_scope(session_factory) as session:
        run_papers = {p.id for p in _run_papers(session, run_id)}
        if a == b or not {a, b} <= run_papers:
            raise ValueError("Both distinct papers must belong to this run's screened corpus")
        candidate = session.scalar(select(StudyLinkCandidate).where(
            StudyLinkCandidate.paper_a_id == a, StudyLinkCandidate.paper_b_id == b))
        if candidate is None:
            raise LookupError("No version-link candidate exists for this pair")
        if decision == "rejected":
            memberships = list(session.scalars(select(StudyPaper).where(
                StudyPaper.paper_id.in_((a, b)))))
            if len(memberships) == 2 and memberships[0].study_id == memberships[1].study_id:
                raise ValueError("Already-linked studies require an explicit split review; no automatic split")
        candidate.status = decision
        candidate.decision_source = "human"
        candidate.match_reason = reason.strip()
        candidate.confidence = 1.0 if decision == "confirmed" else 0.0
