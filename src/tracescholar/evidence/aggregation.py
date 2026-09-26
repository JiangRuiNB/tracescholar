"""Read-only, deterministic aggregation of stored evidence by canonical Study."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from typing import Any

from tracescholar.models import Claim, EvidenceExtraction, EvidenceSpan


STANCES = ("supports", "contradicts", "qualifies", "unrelated")


def aggregate_claim_evidence(
    claim: Claim,
    study_versions: Mapping[uuid.UUID, set[uuid.UUID]],
    latest_attempts: Mapping[tuple[uuid.UUID, uuid.UUID], EvidenceExtraction],
    evidence_role: Callable[[EvidenceSpan], str],
) -> dict[str, Any]:
    """Aggregate already-persisted outcomes; never invoke retrieval or an LLM.

    A Study may have several PDFs. Its stance membership is set-based, and it
    is `no_evidence` only when every applicable version succeeded with that
    disposition. Distinct stances can overlap for a Study whose versions or
    passages differ; callers must not add stance-study counts together.
    """
    stance_studies: dict[str, set[uuid.UUID]] = {stance: set() for stance in STANCES}
    stance_spans = {stance: 0 for stance in STANCES}
    evidence_studies: set[uuid.UUID] = set()
    primary_studies: set[uuid.UUID] = set()
    review_studies: set[uuid.UUID] = set()
    no_evidence_studies: set[uuid.UUID] = set()
    evaluated_studies: set[uuid.UUID] = set()
    pending_studies: set[uuid.UUID] = set()
    spans_out: list[dict[str, Any]] = []
    studies_out: list[dict[str, Any]] = []
    failed_tasks = 0
    no_evidence_tasks = 0

    for study_id in sorted(study_versions, key=str):
        version_ids = sorted(study_versions[study_id], key=str)
        attempts = [latest_attempts.get((study_id, version_id)) for version_id in version_ids]
        successes = [item for item in attempts if item is not None and item.status == "success"]
        failures = [item for item in attempts if item is not None and item.status == "failed"]
        missing = len(version_ids) - len(successes) - len(failures)
        failed_tasks += len(failures)
        no_evidence_tasks += sum(item.disposition == "no_evidence" for item in successes)
        complete = bool(version_ids) and len(successes) == len(version_ids)
        if complete:
            evaluated_studies.add(study_id)
        else:
            pending_studies.add(study_id)

        study_spans: list[EvidenceSpan] = []
        for item in successes:
            for span in item.spans:
                if span.study_id != study_id or span.paper_version_id != item.paper_version_id:
                    raise ValueError("EvidenceSpan study/version does not match its extraction")
                study_spans.append(span)
        study_spans.sort(key=lambda span: (str(span.paper_version_id), span.page_number,
                                            span.page_char_start, str(span.id)))
        study_stances = {span.stance for span in study_spans}
        for stance in study_stances:
            stance_studies[stance].add(study_id)
        for span in study_spans:
            stance_spans[span.stance] += 1
            role = evidence_role(span)
            if span.stance != "unrelated":
                evidence_studies.add(study_id)
                if role == "primary_empirical_evidence":
                    primary_studies.add(study_id)
                elif role == "review_background":
                    review_studies.add(study_id)
            spans_out.append({
                "evidence_span_id": str(span.id), "study_id": str(study_id),
                "paper_version_id": str(span.paper_version_id),
                "chunk_id": str(span.chunk_id), "stance": span.stance,
                "evidence_role": role, "page": span.page_number,
                "quote": span.quote, "confidence": span.confidence,
            })

        all_no_evidence = complete and all(item.disposition == "no_evidence"
                                           for item in successes)
        if all_no_evidence:
            no_evidence_studies.add(study_id)
        reasons = [{"paper_version_id": str(item.paper_version_id),
                    "reason": item.no_evidence_reason}
                   for item in successes if item.disposition == "no_evidence"]
        studies_out.append({
            "study_id": str(study_id), "paper_version_ids": [str(value) for value in version_ids],
            "status": "no_evidence" if all_no_evidence else
                      "evidence" if complete and study_spans else
                      "failed" if failures and not successes else "pending",
            "stances": [stance for stance in STANCES if stance in study_stances],
            "evidence_span_ids": [str(span.id) for span in study_spans],
            "no_evidence_reasons": reasons,
            "failed_versions": [str(item.paper_version_id) for item in failures],
            "missing_versions": [str(version_ids[index]) for index, item in enumerate(attempts)
                                 if item is None],
        })

    outcomes = {
        stance: {"span_count": stance_spans[stance],
                 "study_count": len(stance_studies[stance]),
                 "study_ids": sorted(map(str, stance_studies[stance]))}
        for stance in STANCES
    }
    outcomes["no_evidence"] = {
        "task_count": no_evidence_tasks,
        "study_count": len(no_evidence_studies),
        "study_ids": sorted(map(str, no_evidence_studies)),
    }
    return {
        "claim_id": str(claim.id), "sub_question_index": claim.sub_question_index,
        "statement": claim.statement, "scope_kind": claim.scope_kind,
        "basis_study_id": str(claim.basis_study_id) if claim.basis_study_id else None,
        "basis_chunk_id": str(claim.basis_chunk_id), "basis_quote": claim.basis_quote,
        "evidence_spans": len(spans_out), "applicable_studies": len(study_versions),
        "evaluated_studies": len(evaluated_studies),
        "pending_studies": len(pending_studies),
        "independent_evidence_studies": len(evidence_studies),
        "independent_primary_studies": len(primary_studies),
        "review_background_studies": len(review_studies),
        "stance_spans": stance_spans,
        "stance_studies": {stance: len(stance_studies[stance]) for stance in STANCES},
        "no_evidence_studies": len(no_evidence_studies),
        "failed_tasks": failed_tasks,
        "outcomes": outcomes, "study_results": studies_out, "spans": spans_out,
    }
