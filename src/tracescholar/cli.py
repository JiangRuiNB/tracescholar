"""Command-line interface for TraceScholar."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tracescholar import __version__
from tracescholar.database import DatabaseConfigurationError
from tracescholar.discovery import discover_papers, run_planned_discovery
from tracescholar.evidence import (
    extract_evidence, generate_claims, get_evidence_ledger, get_evidence_span,
)
from tracescholar.exports import build_grounded_review_export, write_grounded_review_export
from tracescholar.fulltext import acquire_fulltext
from tracescholar.fulltext_screening import get_fulltext_decision, screen_fulltext_research_run
from tracescholar.llm import LLMError
from tracescholar.manifests import create_run_manifest, get_run_manifest
from tracescholar.planning import plan_research_run
from tracescholar.pdf_parsing import get_chunk_provenance, parse_research_run
from tracescholar.repositories import create_research_run
from tracescholar.retrieval import (
    EmbeddingError, embed_research_run, search_chunks, search_plan_evidence,
)
from tracescholar.screening import screen_research_run
from tracescholar.studies import (
    decide_study_link, get_study_details, normalize_studies, set_version_result_relation,
)
from tracescholar.synthesis import (
    audit_omitted_counterevidence, audit_synthesis_citations,
    audit_synthesis_semantics, render_synthesis, write_synthesis,
)
from tracescholar.workflow import (
    STAGE_ORDER, WorkflowOrderError, inspect_workflow, run_next_stage, run_workflow,
)


def _json_object(value: str) -> dict[str, Any]:
    """Parse a CLI JSON object with an argparse-compatible error."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"invalid JSON: {error.msg}") from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build and return the TraceScholar argument parser."""
    parser = argparse.ArgumentParser(
        prog="tracescholar",
        description="Evidence-first scientific literature research agent.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")
    research_parser = subparsers.add_parser(
        "research",
        help="create and persist a research run",
    )
    research_parser.add_argument("question", help="the research question to investigate")
    research_parser.add_argument(
        "--scope-json",
        type=_json_object,
        default={},
        metavar="OBJECT",
        help="optional JSON object describing the research scope",
    )
    research_parser.set_defaults(handler=_handle_research)

    plan_parser = subparsers.add_parser(
        "plan", help="generate or reload a frozen ResearchPlan for an existing run"
    )
    plan_parser.add_argument("run_id", type=uuid.UUID, help="existing ResearchRun UUID")
    plan_parser.set_defaults(handler=_handle_plan)

    search_parser = subparsers.add_parser(
        "search",
        help="search OpenAlex and Crossref for an existing research run",
    )
    search_parser.add_argument("run_id", type=uuid.UUID, help="existing ResearchRun UUID")
    search_parser.add_argument("query", help="keyword query sent to both paper sources")
    search_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        metavar="N",
        help="number of results to request per source (1-100; default: 20)",
    )
    search_parser.set_defaults(handler=_handle_search)

    discover_parser = subparsers.add_parser(
        "discover", help="generate and execute queries from a frozen ResearchPlan"
    )
    discover_parser.add_argument("run_id", type=uuid.UUID, help="planned ResearchRun UUID")
    discover_parser.add_argument(
        "--limit", type=int, default=10, metavar="N",
        help="results requested per query and source (1-100; default: 10)",
    )
    discover_parser.set_defaults(handler=_handle_discover)

    screen_parser = subparsers.add_parser(
        "screen", help="run title/abstract screening for papers in a planned research run"
    )
    screen_parser.add_argument("run_id", type=uuid.UUID, help="planned ResearchRun UUID")
    screen_parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="screen at most N pending papers in this invocation",
    )
    screen_parser.set_defaults(handler=_handle_screen)

    acquire_parser = subparsers.add_parser(
        "acquire", help="locate and cache legal OA PDFs for include/maybe papers"
    )
    acquire_parser.add_argument("run_id", type=uuid.UUID, help="screened ResearchRun UUID")
    acquire_parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="attempt at most N uncached papers in this invocation",
    )
    acquire_parser.add_argument(
        "--retry-unavailable", action="store_true",
        help="recheck papers still within the unavailable retry window",
    )
    acquire_parser.set_defaults(handler=_handle_acquire)
    parse_parser = subparsers.add_parser(
        "parse", help="parse acquired PDFs into page-aware, section-aware chunks"
    )
    parse_parser.add_argument("run_id", type=uuid.UUID, help="ResearchRun UUID with acquired PDFs")
    parse_parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="parse at most N not-yet-successful PDF versions in this invocation",
    )
    parse_parser.set_defaults(handler=_handle_parse)
    chunk_parser = subparsers.add_parser(
        "chunk", help="show a chunk's paper, PDF version, page, section, and locator"
    )
    chunk_parser.add_argument("chunk_id", type=uuid.UUID, help="stored Chunk UUID")
    chunk_parser.set_defaults(handler=_handle_chunk)
    embed_parser = subparsers.add_parser(
        "embed", help="embed parsed chunks for one research run into pgvector"
    )
    embed_parser.add_argument("run_id", type=uuid.UUID, help="ResearchRun UUID with parsed PDFs")
    embed_parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="attempt at most N pending or failed chunks in this invocation",
    )
    embed_parser.set_defaults(handler=_handle_embed)

    retrieve_parser = subparsers.add_parser(
        "retrieve", help="semantic-search only a run's acquired PDF chunks"
    )
    retrieve_parser.add_argument("run_id", type=uuid.UUID, help="ResearchRun UUID")
    retrieve_parser.add_argument("query", help="natural-language passage query")
    retrieve_parser.add_argument("--top-k", type=int, default=10, metavar="N")
    retrieve_parser.set_defaults(handler=_handle_retrieve)

    evidence_parser = subparsers.add_parser(
        "evidence", help="retrieve page-level evidence for one paper and plan sub-question"
    )
    evidence_parser.add_argument("run_id", type=uuid.UUID)
    evidence_parser.add_argument("paper_id", type=uuid.UUID)
    evidence_parser.add_argument("--sub-question-index", type=int, required=True, metavar="N")
    evidence_parser.add_argument("--top-k", type=int, default=3, metavar="N")
    evidence_parser.set_defaults(handler=_handle_evidence)

    fulltext_parser = subparsers.add_parser(
        "screen-fulltext", help="review include/maybe papers against retrieved PDF passages"
    )
    fulltext_parser.add_argument("run_id", type=uuid.UUID)
    fulltext_parser.add_argument("--limit", type=int, default=None, metavar="N")
    fulltext_parser.set_defaults(handler=_handle_screen_fulltext)

    decision_parser = subparsers.add_parser(
        "fulltext-decision", help="show one auditable full-text decision and its cited chunks"
    )
    decision_parser.add_argument("run_id", type=uuid.UUID)
    decision_parser.add_argument("paper_id", type=uuid.UUID)
    decision_parser.set_defaults(handler=_handle_fulltext_decision)
    studies_parser = subparsers.add_parser(
        "studies", help="link paper records into independent studies for a run"
    )
    studies_parser.add_argument("run_id", type=uuid.UUID)
    studies_parser.set_defaults(handler=_handle_studies)
    study_parser = subparsers.add_parser("study", help="inspect linked records and PDF versions")
    study_parser.add_argument("run_id", type=uuid.UUID)
    study_parser.add_argument("study_id", type=uuid.UUID)
    study_parser.set_defaults(handler=_handle_study)
    compare_parser = subparsers.add_parser(
        "study-compare", help="record a reviewed PDF-version result comparison"
    )
    compare_parser.add_argument("run_id", type=uuid.UUID)
    compare_parser.add_argument("study_id", type=uuid.UUID)
    compare_parser.add_argument("version_a_id", type=uuid.UUID)
    compare_parser.add_argument("version_b_id", type=uuid.UUID)
    compare_parser.add_argument("relation", choices=("equivalent", "changed"))
    compare_parser.add_argument("--note", required=True, help="human review evidence or explanation")
    compare_parser.set_defaults(handler=_handle_study_compare)
    link_parser = subparsers.add_parser(
        "study-link", help="resolve an ambiguous same-study paper pair after review"
    )
    link_parser.add_argument("run_id", type=uuid.UUID)
    link_parser.add_argument("paper_a_id", type=uuid.UUID)
    link_parser.add_argument("paper_b_id", type=uuid.UUID)
    link_parser.add_argument("decision", choices=("confirmed", "rejected"))
    link_parser.add_argument("--reason", required=True, help="human review explanation")
    link_parser.set_defaults(handler=_handle_study_link)
    claims_parser = subparsers.add_parser(
        "claims", help="generate quote-grounded candidate claims from included studies"
    )
    claims_parser.add_argument("run_id", type=uuid.UUID)
    claims_parser.set_defaults(handler=_handle_claims)
    extraction_parser = subparsers.add_parser(
        "extract-evidence", help="build or resume the study-aware evidence ledger"
    )
    extraction_parser.add_argument("run_id", type=uuid.UUID)
    extraction_parser.add_argument("--limit", type=int, default=None, metavar="N",
                                   help="process at most N pending claim-study PDF tasks")
    extraction_parser.set_defaults(handler=_handle_extract_evidence)
    ledger_parser = subparsers.add_parser("ledger", help="show claims and independent-study evidence counts")
    ledger_parser.add_argument("run_id", type=uuid.UUID)
    ledger_parser.set_defaults(handler=_handle_ledger)
    span_parser = subparsers.add_parser("evidence-span", help="verify a quote back to a PDF page")
    span_parser.add_argument("run_id", type=uuid.UUID)
    span_parser.add_argument("span_id", type=uuid.UUID)
    span_parser.set_defaults(handler=_handle_evidence_span)
    synthesis_parser = subparsers.add_parser(
        "synthesize", help="write and save a fixed-schema evidence synthesis"
    )
    synthesis_parser.add_argument("run_id", type=uuid.UUID)
    synthesis_parser.set_defaults(handler=_handle_synthesize)
    render_parser = subparsers.add_parser(
        "render-synthesis", help="render a saved structured synthesis as Markdown"
    )
    render_parser.add_argument("run_id", type=uuid.UUID)
    render_parser.add_argument("--draft-id", type=uuid.UUID,
                               help="render a specific saved synthesis draft")
    render_parser.set_defaults(handler=_handle_render_synthesis)
    audit_parser = subparsers.add_parser(
        "audit-citations", help="verify and save a draft's citation provenance chains")
    audit_parser.add_argument("run_id", type=uuid.UUID)
    audit_parser.add_argument("--draft-id", type=uuid.UUID,
                              help="audit a specific saved synthesis draft")
    audit_parser.set_defaults(handler=_handle_audit_citations)
    semantic_parser = subparsers.add_parser(
        "audit-semantics", help="audit each draft sentence against its cited evidence")
    semantic_parser.add_argument("run_id", type=uuid.UUID)
    semantic_parser.add_argument("--draft-id", type=uuid.UUID,
                                 help="audit a specific saved synthesis draft")
    semantic_parser.set_defaults(handler=_handle_audit_semantics)
    omission_parser = subparsers.add_parser(
        "audit-omissions", help="check each sentence for material uncited evidence")
    omission_parser.add_argument("run_id", type=uuid.UUID)
    omission_parser.add_argument("--draft-id", type=uuid.UUID,
                                 help="audit a specific saved synthesis draft")
    omission_parser.set_defaults(handler=_handle_audit_omissions)
    manifest_parser = subparsers.add_parser(
        "manifest", help="create or reuse a persisted reproducibility manifest"
    )
    manifest_parser.add_argument("run_id", type=uuid.UUID, help="ResearchRun UUID")
    manifest_parser.set_defaults(handler=_handle_manifest)
    show_manifest_parser = subparsers.add_parser(
        "show-manifest", help="read a previously persisted reproducibility manifest"
    )
    show_manifest_parser.add_argument("manifest_id", type=uuid.UUID, help="RunManifest UUID")
    show_manifest_parser.set_defaults(handler=_handle_show_manifest)
    export_parser = subparsers.add_parser(
        "export", help="write deterministic Grounded Review Markdown and JSON files"
    )
    export_parser.add_argument("run_id", type=uuid.UUID, help="ResearchRun UUID")
    export_parser.add_argument(
        "--manifest-id", type=uuid.UUID,
        help="use a specific immutable RunManifest; defaults to the newest saved one",
    )
    export_parser.add_argument(
        "--output-dir", type=Path,
        help="destination directory (default: data/exports/<run-id>)",
    )
    export_parser.set_defaults(handler=_handle_export)
    run_parser = subparsers.add_parser(
        "run", help="synchronously continue a ResearchRun through remaining stages"
    )
    run_parser.add_argument("run_id", type=uuid.UUID, help="ResearchRun UUID")
    run_parser.set_defaults(handler=_handle_run)
    workflow_status_parser = subparsers.add_parser(
        "workflow-status", help="inspect completed and next workflow stages for one run"
    )
    workflow_status_parser.add_argument("run_id", type=uuid.UUID, help="ResearchRun UUID")
    workflow_status_parser.set_defaults(handler=_handle_workflow_status)
    workflow_step_parser = subparsers.add_parser(
        "workflow-step", help="execute at most one next workflow stage and save its status"
    )
    workflow_step_parser.add_argument("run_id", type=uuid.UUID, help="ResearchRun UUID")
    workflow_step_parser.add_argument(
        "--stage", choices=[stage.value for stage in STAGE_ORDER],
        help="optional guard: require this to be the next eligible stage",
    )
    workflow_step_parser.set_defaults(handler=_handle_workflow_step)
    return parser


def _handle_research(args: argparse.Namespace) -> int:
    """Persist a ResearchRun requested from the CLI."""
    try:
        research_run = create_research_run(args.question, scope=args.scope_json)
    except (DatabaseConfigurationError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"Created ResearchRun {research_run.id} [{research_run.status.value}]")
    return 0


def _handle_plan(args: argparse.Namespace) -> int:
    """Return a strictly validated, persistent ResearchPlan as JSON."""
    try:
        plan = plan_research_run(args.run_id)
    except (DatabaseConfigurationError, LookupError, LLMError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"run_id": str(args.run_id), "plan": plan.model_dump(mode="json")},
                     ensure_ascii=False, indent=2))
    return 0


def _handle_search(args: argparse.Namespace) -> int:
    """Search all discovery sources and show fused paper counts."""
    try:
        summary = discover_papers(
            args.run_id,
            args.query,
            limit_per_source=args.limit,
        )
    except (DatabaseConfigurationError, LookupError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"ResearchRun: {summary.run_id}")
    print(f"Query: {summary.query}")
    for search in summary.searches:
        display_name = {"openalex": "OpenAlex", "crossref": "Crossref"}.get(
            search.source, search.source
        )
        print(
            f"{display_name} returned: {search.results_returned} "
            f"(persisted: {search.results_persisted}, SearchQuery: {search.search_query_id})"
        )
        if search.results_skipped:
            print(f"  Skipped malformed records: {search.results_skipped}")
    for failure in summary.failures:
        print(f"{failure.source}: failed ({failure.reason})", file=sys.stderr)
    print(f"Total raw hits: {summary.total_raw_hits}")
    print(f"SearchResults persisted: {summary.total_results_persisted}")
    print(f"Unique papers: {summary.unique_papers}")
    print(f"Duplicate papers merged: {summary.duplicate_papers_merged}")
    print(f"Papers in run: {summary.papers_in_run}")
    return 0 if summary.searches else 2


def _handle_discover(args: argparse.Namespace) -> int:
    """Run or resume automatic multi-source discovery for a frozen plan."""
    try:
        summary = run_planned_discovery(args.run_id, limit_per_source=args.limit)
    except (DatabaseConfigurationError, LookupError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"ResearchRun: {summary.run_id}")
    print(f"Research tracks: {summary.research_tracks}")
    print(f"Generated queries: {summary.generated_queries}")
    print(f"Newly executed: {summary.newly_executed}")
    print(f"Already executed: {summary.skipped_existing}")
    for source, hits in summary.source_hits.items():
        display_name = {"openalex": "OpenAlex", "crossref": "Crossref"}.get(
            source, source
        )
        print(f"{display_name} hits: {hits}")
    print(f"Total raw hits: {summary.total_raw_hits}")
    print(f"Scope-filtered hits: {summary.scope_filtered_count}")
    print(f"Unique papers: {summary.unique_papers}")
    print(f"Duplicate papers merged: {summary.duplicate_papers_merged}")
    print(f"Papers in run: {summary.papers_in_run}")
    for failure in summary.failures:
        print(f"{failure.source}: failed ({failure.reason})", file=sys.stderr)
    return 0 if not summary.failures else 2


def _handle_screen(args: argparse.Namespace) -> int:
    """Screen run papers and report durable label counts and retryable failures."""
    try:
        summary = screen_research_run(args.run_id, limit=args.limit)
    except (DatabaseConfigurationError, LookupError, ValueError, LLMError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"ResearchRun: {summary.run_id}")
    print(f"Papers: {summary.total_papers}")
    print(f"Newly screened: {summary.newly_screened}")
    print(f"Skipped unchanged: {summary.skipped_unchanged}")
    print(f"Include: {summary.include}")
    print(f"Maybe: {summary.maybe}")
    print(f"Exclude: {summary.exclude}")
    print(f"Pending: {summary.pending}")
    for failure in summary.failures:
        print(f"{failure.paper_id}: failed ({failure.reason})", file=sys.stderr)
    return 0 if not summary.failures else 2


def _handle_acquire(args: argparse.Namespace) -> int:
    """Report legal OA availability and durable, versioned PDF acquisition."""
    try:
        summary = acquire_fulltext(
            args.run_id, limit=args.limit, retry_unavailable=args.retry_unavailable,
        )
    except (DatabaseConfigurationError, LookupError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"ResearchRun: {summary.run_id}")
    print(f"Candidate papers: {summary.candidate_papers}")
    print(f"Full text available: {summary.full_text_available}")
    print(f"Downloaded: {summary.downloaded}")
    print(f"Already cached: {summary.already_cached}")
    print(f"Unavailable: {summary.unavailable}")
    print(f"Failed: {summary.failed}")
    print(f"Pending: {summary.pending}")
    if summary.stale_screening:
        print(f"Needs re-screening: {summary.stale_screening}")
    return 0 if summary.failed == 0 else 2


def _handle_parse(args: argparse.Namespace) -> int:
    """Persist page-aware PDF text and report per-version failures."""
    try:
        summary = parse_research_run(args.run_id, limit=args.limit)
    except (DatabaseConfigurationError, LookupError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"ResearchRun: {summary.run_id}")
    print(f"PaperVersions: {summary.paper_versions}")
    print(f"Parsed successfully: {summary.parsed_successfully}")
    print(f"Failed: {summary.failed}")
    print(f"Newly parsed: {summary.newly_parsed}")
    print(f"Skipped unchanged: {summary.skipped_existing}")
    print(f"Total pages: {summary.total_pages}")
    print(f"Total chunks: {summary.total_chunks}")
    for failure in summary.failures:
        print(f"{failure.paper_version_id}: failed ({failure.code}: {failure.detail})", file=sys.stderr)
    return 0 if summary.failed == 0 else 2


def _handle_chunk(args: argparse.Namespace) -> int:
    """Print an auditable PDF locator for one stored chunk."""
    try:
        item = get_chunk_provenance(args.chunk_id)
    except (DatabaseConfigurationError, LookupError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "chunk_id": str(item.chunk_id), "paper_id": str(item.paper_id),
        "paper_title": item.paper_title,
        "paper_version_id": str(item.paper_version_id),
        "storage_path": item.storage_path, "content_hash": item.content_hash,
        "page_start": item.page_start, "page_end": item.page_end,
        "section": item.section, "ordinal": item.ordinal,
        "locator": item.locator, "text": item.text,
    }, ensure_ascii=False, indent=2))
    return 0


def _handle_embed(args: argparse.Namespace) -> int:
    """Report resumable, versioned chunk embedding outcomes."""
    try:
        summary = embed_research_run(args.run_id, limit=args.limit)
    except (DatabaseConfigurationError, LookupError, ValueError, EmbeddingError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"ResearchRun: {summary.run_id}")
    print(f"Chunks: {summary.total_chunks}")
    print(f"Model: {summary.provider}/{summary.model_name} ({summary.dimensions}d)")
    print(f"Model revision: {summary.model_revision}")
    print(f"New embeddings: {summary.newly_embedded}")
    print(f"Skipped unchanged: {summary.skipped_unchanged}")
    print(f"Failed: {summary.failed}")
    print(f"Pending: {summary.pending}")
    for failure in summary.failures:
        print(f"{failure.chunk_id}: failed ({failure.code}: {failure.detail})", file=sys.stderr)
    return 0 if summary.failed == 0 else 2


def _handle_retrieve(args: argparse.Namespace) -> int:
    """Print scored passages with complete PDF locator chain."""
    try:
        results = search_chunks(args.run_id, args.query, top_k=args.top_k)
    except (DatabaseConfigurationError, LookupError, ValueError, EmbeddingError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"ResearchRun: {args.run_id}")
    print(f"Query: {args.query}")
    print(f"Results: {len(results)}")
    for rank, item in enumerate(results, 1):
        print(f"{rank}. {item.paper_title}")
        print(f"   Page: {item.page_start} | Section: {item.section} | "
              f"Similarity: {item.similarity:.4f} | Distance: {item.cosine_distance:.4f}")
        print(f"   Chunk: {item.chunk_id} | PaperVersion: {item.paper_version_id}")
        print(f"   Locator: {json.dumps(item.locator, ensure_ascii=False)}")
        print(f"   Text: {item.text[:350].replace(chr(10), ' ')}")
    return 0


def _handle_evidence(args: argparse.Namespace) -> int:
    try:
        results = search_plan_evidence(
            args.run_id, args.paper_id, args.sub_question_index, top_k=args.top_k,
        )
    except (DatabaseConfigurationError, LookupError, ValueError, EmbeddingError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "run_id": str(args.run_id), "paper_id": str(args.paper_id),
        "sub_question_index": args.sub_question_index,
        "results": [{
            "chunk_id": str(item.chunk_id), "paper_version_id": str(item.paper_version_id),
            "paper_title": item.paper_title, "page_start": item.page_start,
            "page_end": item.page_end, "section": item.section,
            "similarity": item.similarity, "locator": item.locator, "text": item.text,
        } for item in results],
    }, ensure_ascii=False, indent=2))
    return 0


def _handle_screen_fulltext(args: argparse.Namespace) -> int:
    try:
        summary = screen_fulltext_research_run(args.run_id, limit=args.limit)
    except (DatabaseConfigurationError, LookupError, ValueError, EmbeddingError, LLMError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"ResearchRun: {summary.run_id}")
    print(f"Candidates: {summary.candidates}")
    print(f"New full-text screenings: {summary.newly_screened}")
    print(f"Skipped unchanged: {summary.skipped_unchanged}")
    print(f"Final include: {summary.include}")
    print(f"Final exclude: {summary.exclude}")
    print(f"Uncertain: {summary.uncertain}")
    print(f"Failed: {summary.failed}")
    print(f"Pending: {summary.pending}")
    for failure in summary.failures:
        print(f"{failure.paper_id}: failed ({failure.code}: {failure.detail})", file=sys.stderr)
    return 0 if summary.failed == 0 else 2


def _handle_fulltext_decision(args: argparse.Namespace) -> int:
    try:
        decision = get_fulltext_decision(args.run_id, args.paper_id)
    except (DatabaseConfigurationError, LookupError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


def _handle_studies(args: argparse.Namespace) -> int:
    try:
        summary = normalize_studies(args.run_id)
    except (DatabaseConfigurationError, LookupError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    for label, value in (
        ("ResearchRun", summary.run_id), ("Paper records", summary.paper_records),
        ("Paper versions", summary.paper_versions),
        ("Potential version groups", summary.potential_version_groups),
        ("Canonical studies", summary.canonical_studies),
        ("Linked records", summary.linked_records),
        ("Collapsed surplus records", summary.collapsed_surplus),
        ("Unresolved", summary.unresolved), ("Newly linked", summary.newly_linked),
        ("Result comparisons not assessed", summary.result_pairs_not_assessed),
        ("Skipped unchanged pairs", summary.skipped_unchanged),
    ):
        print(f"{label}: {value}")
    return 0


def _handle_study(args: argparse.Namespace) -> int:
    try:
        details = get_study_details(args.run_id, args.study_id)
    except (DatabaseConfigurationError, LookupError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(details, ensure_ascii=False, indent=2))
    return 0


def _handle_study_compare(args: argparse.Namespace) -> int:
    try:
        set_version_result_relation(
            args.run_id, args.study_id, args.version_a_id, args.version_b_id,
            args.relation, args.note,
        )
    except (DatabaseConfigurationError, LookupError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"Recorded {args.relation} result comparison for study {args.study_id}")
    return 0


def _handle_study_link(args: argparse.Namespace) -> int:
    try:
        decide_study_link(args.run_id, args.paper_a_id, args.paper_b_id,
                          args.decision, args.reason)
    except (DatabaseConfigurationError, LookupError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"Recorded {args.decision} link; rerun `tracescholar studies {args.run_id}` to apply")
    return 0


def _handle_claims(args: argparse.Namespace) -> int:
    try:
        summary = generate_claims(args.run_id)
    except (DatabaseConfigurationError, LookupError, ValueError, LLMError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"ResearchRun: {summary.run_id}")
    print(f"Included studies: {summary.included_studies}")
    print(f"Claims: {summary.claims}")
    print(f"Newly generated: {summary.newly_generated}")
    print(f"Skipped unchanged: {summary.skipped_unchanged}")
    print(f"Generation: {summary.generation_id}")
    return 0


def _handle_extract_evidence(args: argparse.Namespace) -> int:
    try:
        summary = extract_evidence(args.run_id, limit=args.limit)
    except (DatabaseConfigurationError, LookupError, ValueError, LLMError,
            EmbeddingError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    for label, value in (
        ("ResearchRun", summary.run_id), ("Claims", summary.claims),
        ("Included studies", summary.included_studies),
        ("Extraction tasks", summary.extraction_tasks),
        ("Newly extracted", summary.newly_extracted),
        ("Skipped unchanged", summary.skipped_unchanged),
        ("No evidence", summary.no_evidence), ("Failed", summary.failed),
        ("Pending", summary.pending), ("EvidenceSpans", summary.spans),
    ):
        print(f"{label}: {value}")
    for failure in summary.failures:
        print(f"{failure.claim_id}/{failure.study_id}: {failure.code}: {failure.detail}",
              file=sys.stderr)
    return 0 if summary.failed == 0 else 2


def _handle_ledger(args: argparse.Namespace) -> int:
    try:
        result = get_evidence_ledger(args.run_id)
    except (DatabaseConfigurationError, LookupError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _handle_evidence_span(args: argparse.Namespace) -> int:
    try:
        result = get_evidence_span(args.run_id, args.span_id)
    except (DatabaseConfigurationError, LookupError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _handle_synthesize(args: argparse.Namespace) -> int:
    try:
        result = write_synthesis(args.run_id)
    except (DatabaseConfigurationError, LookupError, LLMError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "run_id": str(result.run_id), "draft_id": str(result.draft_id),
        "generated": result.generated, "attempt_count": result.attempt_count,
        "document": result.document.model_dump(mode="json"),
    }, ensure_ascii=False, indent=2))
    return 0


def _handle_render_synthesis(args: argparse.Namespace) -> int:
    try:
        markdown = render_synthesis(args.run_id, draft_id=args.draft_id)
    except (DatabaseConfigurationError, LookupError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(markdown, end="")
    return 0


def _handle_audit_citations(args: argparse.Namespace) -> int:
    try:
        result = audit_synthesis_citations(args.run_id, draft_id=args.draft_id)
    except (DatabaseConfigurationError, LookupError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "run_id": str(result.run_id), "draft_id": str(result.draft_id),
        "audit_id": str(result.audit_id), "status": result.status,
        "created": result.created, "sentences": result.sentence_count,
        "citations": result.citation_count,
        "unique_evidence": result.unique_evidence_count,
        "issues": result.issues,
        "sentence_audits": [{
            "sentence_audit_id": str(item.sentence_audit_id),
            "section_index": item.section_index,
            "paragraph_index": item.paragraph_index,
            "sentence_index": item.sentence_index,
            "input_hash": item.input_hash, "status": item.status,
            "issues": item.issues, "created": item.created,
        } for item in result.sentence_results],
    }, ensure_ascii=False, indent=2))
    return 0 if result.status == "passed" else 2


def _handle_audit_semantics(args: argparse.Namespace) -> int:
    try:
        result = audit_synthesis_semantics(args.run_id, draft_id=args.draft_id)
    except (DatabaseConfigurationError, LookupError, LLMError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "run_id": str(result.run_id), "draft_id": str(result.draft_id),
        "citation_audit_id": str(result.citation_audit_id),
        "counts": result.counts,
        "sentences": [{
            "audit_id": str(item.audit_id),
            "section_index": item.section_index,
            "paragraph_index": item.paragraph_index,
            "sentence_index": item.sentence_index,
            "status": item.status, "verdict": item.verdict,
            "entailment": item.entailment, "scope": item.scope,
            "strength": item.strength, "rationale": item.rationale,
            "minimal_revision": item.minimal_revision,
            "failure_code": item.failure_code,
            "attempt_count": item.attempt_count, "generated": item.generated,
        } for item in result.sentences],
    }, ensure_ascii=False, indent=2))
    return 2 if result.counts["failed"] else 0


def _handle_audit_omissions(args: argparse.Namespace) -> int:
    try:
        result = audit_omitted_counterevidence(args.run_id, draft_id=args.draft_id)
    except (DatabaseConfigurationError, LookupError, LLMError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "run_id": str(result.run_id), "draft_id": str(result.draft_id),
        "citation_audit_id": str(result.citation_audit_id),
        "counts": result.counts,
        "sentences": [{
            "audit_id": str(item.audit_id),
            "section_index": item.section_index,
            "paragraph_index": item.paragraph_index,
            "sentence_index": item.sentence_index,
            "status": item.status, "verdict": item.verdict,
            "omitted_evidence_ids": [str(value) for value in item.omitted_evidence_ids],
            "impacts": [{"evidence_id": str(impact.evidence_id),
                         "impact_type": impact.impact_type}
                        for impact in item.impacts],
            "rationale": item.rationale, "failure_code": item.failure_code,
            "attempt_count": item.attempt_count, "generated": item.generated,
        } for item in result.sentences],
    }, ensure_ascii=False, indent=2))
    return 2 if result.counts["failed"] else 0


def _handle_manifest(args: argparse.Namespace) -> int:
    """Build an immutable manifest exclusively from persisted stage records."""
    try:
        result = create_run_manifest(args.run_id)
    except (DatabaseConfigurationError, LookupError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "manifest_id": str(result.manifest_id),
        "run_id": str(result.run_id),
        "content_hash": result.content_hash,
        "created": result.created,
        "manifest": result.manifest.model_dump(mode="json"),
    }, ensure_ascii=False, indent=2))
    return 0


def _handle_show_manifest(args: argparse.Namespace) -> int:
    """Read and schema-validate one persisted immutable manifest."""
    try:
        result = get_run_manifest(args.manifest_id)
    except (DatabaseConfigurationError, LookupError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "manifest_id": str(result.manifest_id),
        "run_id": str(result.run_id),
        "content_hash": result.content_hash,
        "manifest": result.manifest.model_dump(mode="json"),
    }, ensure_ascii=False, indent=2))
    return 0


def _handle_export(args: argparse.Namespace) -> int:
    """Export only the saved draft and saved audit/manifest state."""
    output_dir = args.output_dir or Path("data") / "exports" / str(args.run_id)
    try:
        result = build_grounded_review_export(
            args.run_id, manifest_id=args.manifest_id
        )
        markdown_path, json_path = write_grounded_review_export(result, output_dir)
    except (DatabaseConfigurationError, LookupError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    sentences = [
        sentence
        for section in result.content.sections
        for paragraph in section.paragraphs
        for sentence in paragraph.sentences
    ]
    semantic_counts: dict[str, int] = {}
    for sentence in sentences:
        label = sentence.semantic_audit.verdict or sentence.semantic_audit.status
        semantic_counts[label] = semantic_counts.get(label, 0) + 1
    print(json.dumps({
        "run_id": str(result.run_id),
        "draft_id": str(result.draft_id),
        "manifest_id": str(result.manifest_id),
        "markdown_path": str(markdown_path),
        "json_path": str(json_path),
        "sentences": len(sentences),
        "semantic_audit": semantic_counts,
    }, ensure_ascii=False, indent=2))
    return 0


def _handle_workflow_status(args: argparse.Namespace) -> int:
    """Show durable stage progress without executing any stage."""
    try:
        snapshot = inspect_workflow(args.run_id)
    except (DatabaseConfigurationError, LookupError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(snapshot.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


def _handle_workflow_step(args: argparse.Namespace) -> int:
    """Advance one stage only; this command never loops through the pipeline."""
    try:
        result = run_next_stage(args.run_id, stage=args.stage)
    except (DatabaseConfigurationError, LookupError, WorkflowOrderError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "run_id": str(result.run_id),
        "stage": result.stage.value if result.stage else None,
        "status": result.status,
        "skipped": result.skipped,
        "detail": result.detail,
        "result": result.result,
        "workflow": result.snapshot.model_dump(mode="json"),
    }, ensure_ascii=False, indent=2))
    return 0 if result.status in {"completed", "running"} else 2


def _handle_run(args: argparse.Namespace) -> int:
    """Continue synchronously and stop on completion, failure, or a dependency block."""
    def report_step(step) -> None:
        stage_name = step.stage.value if step.stage else "workflow"
        suffix = f": {step.detail}" if step.detail else ""
        print(f"[{stage_name}] {step.status}{suffix}", file=sys.stderr, flush=True)

    try:
        result = run_workflow(args.run_id, on_step=report_step)
    except (DatabaseConfigurationError, LookupError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0 if result.status == "completed" else 2


def main(argv: Sequence[str] | None = None) -> int:
    """Run the TraceScholar command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "handler"):
        return args.handler(args)
    print("TraceScholar is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
