"""Grounded candidate claims and auditable page-level evidence."""

from tracescholar.evidence.schemas import ClaimDraftBatch, EvidenceDecision
from tracescholar.evidence.service import (
    EvidenceSummary, extract_evidence, generate_claims, get_evidence_ledger,
    get_evidence_span,
)

__all__ = ["ClaimDraftBatch", "EvidenceDecision", "EvidenceSummary", "extract_evidence",
           "generate_claims", "get_evidence_ledger", "get_evidence_span"]
