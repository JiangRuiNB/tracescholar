"""Persistence operations for TraceScholar domain models."""

from tracescholar.repositories.discovery import (
    PaperIdentityConflict,
    add_search_result,
    create_search_query,
    list_run_papers,
    normalize_arxiv_id,
    normalize_doi,
    normalize_title,
    upsert_paper,
)
from tracescholar.repositories.research_runs import create_research_run
from tracescholar.repositories.plans import PlanFrozenError, load_research_plan, save_research_plan

__all__ = [
    "PaperIdentityConflict",
    "PlanFrozenError",
    "add_search_result",
    "create_research_run",
    "create_search_query",
    "list_run_papers",
    "load_research_plan",
    "normalize_arxiv_id",
    "normalize_doi",
    "normalize_title",
    "upsert_paper",
    "save_research_plan",
]
