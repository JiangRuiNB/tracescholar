"""Versioned chunk embeddings and run-scoped semantic retrieval."""

from tracescholar.retrieval.encoder import EmbeddingError, OpenAICompatibleEmbeddings
from tracescholar.retrieval.service import (
    EmbeddingSummary, RetrievedChunk, embed_research_run, search_chunks,
    plan_evidence_queries, search_paper_chunks, search_plan_evidence,
)

__all__ = ["EmbeddingError", "OpenAICompatibleEmbeddings", "EmbeddingSummary",
           "RetrievedChunk", "embed_research_run", "search_chunks", "search_paper_chunks",
           "plan_evidence_queries", "search_plan_evidence"]
