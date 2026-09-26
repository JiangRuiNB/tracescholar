"""Local reference checks shared by Writer and renderer."""

from __future__ import annotations

import uuid
from typing import Any

from tracescholar.synthesis.schemas import SynthesisDocument


class SynthesisValidationError(ValueError):
    """A structured draft refers to material outside its input Evidence Ledger."""


def validate_synthesis_references(
    document: SynthesisDocument, ledger: dict[str, Any], research_question: str,
) -> None:
    """Check identifier provenance without judging citation semantics."""
    if document.research_question != research_question:
        raise SynthesisValidationError("Synthesis changed the frozen research question")
    claim_evidence = {
        uuid.UUID(item["claim_id"]): {uuid.UUID(span["evidence_span_id"])
                                       for span in item["spans"]}
        for item in ledger["claims"]
    }
    cited_claims: set[uuid.UUID] = set()
    cited_evidence: set[uuid.UUID] = set()
    for section in document.sections:
        for paragraph in section.paragraphs:
            for sentence in paragraph.sentences:
                if not sentence.claim_ids:
                    raise SynthesisValidationError("Every synthesis sentence needs a Claim ID")
                if any(claim_id not in claim_evidence for claim_id in sentence.claim_ids):
                    raise SynthesisValidationError("Synthesis cited a Claim outside this ledger")
                if any(not claim_evidence[claim_id] for claim_id in sentence.claim_ids):
                    raise SynthesisValidationError("Synthesis cited a Claim without EvidenceSpans")
                if not sentence.evidence_ids:
                    raise SynthesisValidationError("Every synthesis sentence needs an EvidenceSpan ID")
                allowed = set().union(*(claim_evidence[claim_id]
                                        for claim_id in sentence.claim_ids))
                if any(evidence_id not in allowed for evidence_id in sentence.evidence_ids):
                    raise SynthesisValidationError(
                        "Synthesis cited an EvidenceSpan outside its sentence's Claims")
                cited_claims.update(sentence.claim_ids)
                cited_evidence.update(sentence.evidence_ids)
    claims_with_evidence = {claim_id for claim_id, evidence_ids in claim_evidence.items()
                            if evidence_ids}
    if cited_claims != claims_with_evidence:
        raise SynthesisValidationError("Synthesis omitted a Claim with direct EvidenceSpans")
    if not cited_evidence:
        raise SynthesisValidationError("Synthesis did not cite any EvidenceSpan")
