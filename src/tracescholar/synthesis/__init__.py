"""Fixed schemas for a future research synthesis stage."""

from tracescholar.synthesis.schemas import (
    SynthesisDocument, SynthesisParagraph, SynthesisSection, SynthesisSentence,
)
from tracescholar.synthesis.renderer import render_markdown, render_synthesis
from tracescholar.synthesis.auditor import (
    CitationAuditResult, CitationSentenceResult, audit_synthesis_citations,
)
from tracescholar.synthesis.semantic_auditor import (
    SemanticAuditResult, SemanticSentenceResult, audit_synthesis_semantics,
)
from tracescholar.synthesis.semantic_schemas import SemanticCitationJudgment
from tracescholar.synthesis.omission_auditor import (
    EvidenceImpactResult, OmissionAuditResult, OmissionSentenceResult,
    audit_omitted_counterevidence,
)
from tracescholar.synthesis.omission_schemas import OmissionJudgment
from tracescholar.synthesis.validation import SynthesisValidationError
from tracescholar.synthesis.writer import SynthesisWriteResult, write_synthesis

__all__ = [
    "SynthesisDocument", "SynthesisParagraph", "SynthesisSection", "SynthesisSentence",
    "SynthesisValidationError", "SynthesisWriteResult", "render_markdown",
    "render_synthesis", "write_synthesis", "CitationAuditResult",
    "CitationSentenceResult",
    "audit_synthesis_citations",
    "SemanticAuditResult", "SemanticSentenceResult", "SemanticCitationJudgment",
    "audit_synthesis_semantics",
    "EvidenceImpactResult", "OmissionAuditResult", "OmissionSentenceResult", "OmissionJudgment",
    "audit_omitted_counterevidence",
]
