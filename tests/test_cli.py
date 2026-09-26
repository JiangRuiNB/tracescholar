"""Tests for the TraceScholar command-line interface."""

from __future__ import annotations

import contextlib
import io
import json
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tracescholar.cli import main
from tracescholar.llm import LLMConfigurationError
from tracescholar.models import ResearchRunStatus


class CliTestCase(unittest.TestCase):
    @patch("tracescholar.cli.run_workflow")
    def test_run_command_invokes_synchronous_workflow(self, run_workflow) -> None:
        run_id = uuid.uuid4()
        run_workflow.return_value = SimpleNamespace(
            status="completed",
            model_dump=lambda **kwargs: {"run_id": str(run_id), "status": "completed", "steps": []},
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = main(["run", str(run_id)])
        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "completed")
        self.assertEqual(run_workflow.call_args.args, (run_id,))
        self.assertIn("on_step", run_workflow.call_args.kwargs)

    @patch("tracescholar.cli.inspect_workflow")
    def test_workflow_status_command_reports_next_stage(self, inspect) -> None:
        run_id = uuid.uuid4()
        inspect.return_value = SimpleNamespace(model_dump=lambda **kwargs: {
            "run_id": str(run_id),
            "completed_stages": ["planned", "discovered", "screened"],
            "next_stage": "acquired", "stages": [], "last_failure": None,
        })
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = main(["workflow-status", str(run_id)])
        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(output.getvalue())["next_stage"], "acquired")
        inspect.assert_called_once_with(run_id)

    @patch("tracescholar.cli.run_next_stage")
    def test_workflow_step_executes_only_one_requested_next_stage(self, run_next) -> None:
        run_id = uuid.uuid4()
        run_next.return_value = SimpleNamespace(
            run_id=run_id, stage=SimpleNamespace(value="acquired"), status="completed",
            skipped=False, result={"downloaded": 4}, detail=None,
            snapshot=SimpleNamespace(model_dump=lambda **kwargs: {"next_stage": "parsed"}),
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = main(["workflow-step", str(run_id), "--stage", "acquired"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(output.getvalue())["stage"], "acquired")
        run_next.assert_called_once_with(run_id, stage="acquired")

    @patch("tracescholar.cli.audit_omitted_counterevidence")
    def test_omission_audit_command_reports_missing_evidence_ids(self, audit) -> None:
        run_id, draft_id, chain_id, sentence_audit_id, evidence_id = (
            uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), uuid.uuid4())
        audit.return_value = SimpleNamespace(
            run_id=run_id, draft_id=draft_id, citation_audit_id=chain_id,
            counts={"pass": 0, "revise": 1, "flag": 0, "failed": 0},
            sentences=(SimpleNamespace(
                audit_id=sentence_audit_id, section_index=0, paragraph_index=0,
                sentence_index=0, status="success", verdict="revise",
                omitted_evidence_ids=(evidence_id,), rationale="Scope needs narrowing.",
                impacts=(SimpleNamespace(evidence_id=evidence_id, impact_type="limits"),),
                failure_code=None, attempt_count=1, generated=True,
            ),),
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = main(["audit-omissions", str(run_id), "--draft-id", str(draft_id)])
        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(output.getvalue())["sentences"][0]
                         ["omitted_evidence_ids"], [str(evidence_id)])
        audit.assert_called_once_with(run_id, draft_id=draft_id)

    @patch("tracescholar.cli.audit_synthesis_semantics")
    def test_semantic_audit_command_reports_sentence_verdicts(self, audit) -> None:
        run_id, draft_id, chain_id, sentence_audit_id = (
            uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), uuid.uuid4())
        audit.return_value = SimpleNamespace(
            run_id=run_id, draft_id=draft_id, citation_audit_id=chain_id,
            counts={"pass": 0, "revise": 1, "reject": 0, "failed": 0},
            sentences=(SimpleNamespace(
                audit_id=sentence_audit_id, section_index=0, paragraph_index=0,
                sentence_index=0, status="success", verdict="revise",
                entailment="partial", scope="too_broad", strength="overstated",
                rationale="One benchmark only.", minimal_revision="On this benchmark...",
                failure_code=None, attempt_count=1, generated=True,
            ),),
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = main(["audit-semantics", str(run_id), "--draft-id", str(draft_id)])
        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(output.getvalue())["sentences"][0]["verdict"], "revise")
        audit.assert_called_once_with(run_id, draft_id=draft_id)

    @patch("tracescholar.cli.audit_synthesis_citations")
    def test_citation_audit_command_reports_persisted_result(self, audit) -> None:
        run_id, draft_id, audit_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        audit.return_value = SimpleNamespace(
            run_id=run_id, draft_id=draft_id, audit_id=audit_id,
            status="passed", created=False, sentence_count=3,
            citation_count=4, unique_evidence_count=2, issues=(), sentence_results=(),
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = main(["audit-citations", str(run_id), "--draft-id", str(draft_id)])
        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(output.getvalue())["audit_id"], str(audit_id))
        audit.assert_called_once_with(run_id, draft_id=draft_id)

    def test_cli_reports_ready(self) -> None:
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            exit_code = main([])

        self.assertEqual(exit_code, 0)
        self.assertEqual(output.getvalue().strip(), "TraceScholar is ready.")

    @patch("tracescholar.cli.create_research_run")
    def test_research_command_persists_and_reports_run(self, create_run) -> None:
        run_id = uuid.uuid4()
        create_run.return_value = SimpleNamespace(
            id=run_id,
            status=ResearchRunStatus.PENDING,
        )
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            exit_code = main(
                [
                    "research",
                    "How reliable is retrieval-augmented generation?",
                    "--scope-json",
                    '{"year_from": 2024}',
                ]
            )

        self.assertEqual(exit_code, 0)
        create_run.assert_called_once_with(
            "How reliable is retrieval-augmented generation?",
            scope={"year_from": 2024},
        )
        self.assertEqual(
            output.getvalue().strip(),
            f"Created ResearchRun {run_id} [pending]",
        )

    @patch("tracescholar.cli.plan_research_run")
    def test_plan_command_returns_structured_json(self, plan_research_run) -> None:
        run_id = uuid.uuid4()
        plan_research_run.return_value = SimpleNamespace(
            model_dump=lambda **kwargs: {"normalized_question": "RAG 的效果如何？", "scope_snapshot": {}}
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = main(["plan", str(run_id)])
        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(output.getvalue())["plan"]["normalized_question"], "RAG 的效果如何？")
        plan_research_run.assert_called_once_with(run_id)

    @patch("tracescholar.cli.plan_research_run", side_effect=LLMConfigurationError("key missing"))
    def test_plan_command_reports_missing_llm_configuration(self, plan_research_run) -> None:
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            exit_code = main(["plan", str(uuid.uuid4())])
        self.assertEqual(exit_code, 2)
        self.assertIn("key missing", output.getvalue())

    @patch("tracescholar.cli.discover_papers")
    def test_search_command_reports_fused_counts(self, discover_papers) -> None:
        run_id = uuid.uuid4()
        discover_papers.return_value = SimpleNamespace(
            run_id=run_id,
            query="retrieval augmented generation",
            searches=(
                SimpleNamespace(
                    source="openalex", results_returned=20, results_persisted=20,
                    search_query_id=uuid.uuid4(), results_skipped=0,
                ),
                SimpleNamespace(
                    source="crossref", results_returned=20, results_persisted=19,
                    search_query_id=uuid.uuid4(), results_skipped=1,
                ),
            ),
            failures=(),
            total_raw_hits=40,
            total_results_persisted=39,
            unique_papers=35,
            duplicate_papers_merged=4,
            papers_in_run=18,
        )
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            exit_code = main(["search", str(run_id), "retrieval augmented generation", "--limit", "20"])

        self.assertEqual(exit_code, 0)
        self.assertIn("OpenAlex returned: 20", output.getvalue())
        self.assertIn("Crossref returned: 20", output.getvalue())
        self.assertIn("Total raw hits: 40", output.getvalue())
        self.assertIn("Unique papers: 35", output.getvalue())
        self.assertIn("Duplicate papers merged: 4", output.getvalue())
        discover_papers.assert_called_once_with(
            run_id, "retrieval augmented generation", limit_per_source=20
        )

    @patch("tracescholar.cli.discover_papers")
    def test_search_command_reports_partial_failure(self, discover_papers) -> None:
        discover_papers.return_value = SimpleNamespace(
            run_id=uuid.uuid4(), query="query",
            searches=(SimpleNamespace(source="openalex", results_returned=1,
                                     results_persisted=1, search_query_id=uuid.uuid4(),
                                     results_skipped=0),),
            failures=(SimpleNamespace(source="crossref", reason="HTTP 503"),),
            total_raw_hits=1, total_results_persisted=1, unique_papers=1,
            duplicate_papers_merged=0, papers_in_run=1,
        )
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = main(["search", str(discover_papers.return_value.run_id), "query"])
        self.assertEqual(exit_code, 0)
        self.assertIn("crossref: failed", stderr.getvalue())

    @patch("tracescholar.cli.run_planned_discovery")
    def test_discover_command_reports_resumable_plan_counts(self, run_planned_discovery) -> None:
        run_id = uuid.uuid4()
        run_planned_discovery.return_value = SimpleNamespace(
            run_id=run_id, research_tracks=4, generated_queries=8,
            newly_executed=0, skipped_existing=16, source_hits={"openalex": 40, "crossref": 40},
            total_raw_hits=80, scope_filtered_count=36, unique_papers=32,
            duplicate_papers_merged=12, papers_in_run=32, failures=(),
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["discover", str(run_id), "--limit", "5"])
        self.assertEqual(result, 0)
        self.assertIn("Generated queries: 8", output.getvalue())
        self.assertIn("Newly executed: 0", output.getvalue())
        self.assertIn("Unique papers: 32", output.getvalue())
        run_planned_discovery.assert_called_once_with(run_id, limit_per_source=5)

    @patch("tracescholar.cli.screen_research_run")
    def test_screen_command_reports_saved_label_counts(self, screen_research_run) -> None:
        run_id = uuid.uuid4()
        screen_research_run.return_value = SimpleNamespace(
            run_id=run_id, total_papers=32, newly_screened=0, skipped_unchanged=32,
            include=5, maybe=20, exclude=7, pending=0, failures=(),
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["screen", str(run_id), "--limit", "10"])
        self.assertEqual(result, 0)
        self.assertIn("Include: 5", output.getvalue())
        self.assertIn("Maybe: 20", output.getvalue())
        self.assertIn("Exclude: 7", output.getvalue())
        screen_research_run.assert_called_once_with(run_id, limit=10)

    @patch("tracescholar.cli.acquire_fulltext")
    def test_acquire_command_reports_oa_outcomes(self, acquire_fulltext) -> None:
        run_id = uuid.uuid4()
        acquire_fulltext.return_value = SimpleNamespace(
            run_id=run_id, candidate_papers=24, full_text_available=15,
            downloaded=10, already_cached=3, unavailable=8, failed=1,
            pending=2, stale_screening=0,
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["acquire", str(run_id), "--limit", "10", "--retry-unavailable"])
        self.assertEqual(result, 2)
        self.assertIn("Candidate papers: 24", output.getvalue())
        self.assertIn("Downloaded: 10", output.getvalue())
        self.assertIn("Failed: 1", output.getvalue())
        acquire_fulltext.assert_called_once_with(
            run_id, limit=10, retry_unavailable=True,
        )

    @patch("tracescholar.cli.parse_research_run")
    def test_parse_command_reports_page_and_chunk_counts(self, parse_research_run) -> None:
        run_id = uuid.uuid4()
        parse_research_run.return_value = SimpleNamespace(
            run_id=run_id, paper_versions=24, parsed_successfully=23, failed=1,
            newly_parsed=23, skipped_existing=0, total_pages=288,
            total_chunks=1200,
            failures=(SimpleNamespace(paper_version_id=uuid.uuid4(), code="invalid_pdf",
                                      detail="broken PDF"),),
        )
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = main(["parse", str(run_id), "--limit", "24"])
        self.assertEqual(result, 2)
        self.assertIn("PaperVersions: 24", stdout.getvalue())
        self.assertIn("Total chunks: 1200", stdout.getvalue())
        self.assertIn("invalid_pdf", stderr.getvalue())
        parse_research_run.assert_called_once_with(run_id, limit=24)

    @patch("tracescholar.cli.get_chunk_provenance")
    def test_chunk_command_shows_auditable_source(self, get_chunk_provenance) -> None:
        chunk_id = uuid.uuid4()
        version_id = uuid.uuid4()
        get_chunk_provenance.return_value = SimpleNamespace(
            chunk_id=chunk_id, paper_id=uuid.uuid4(), paper_title="Research paper",
            paper_version_id=version_id, storage_path="fulltext/sha256/example.pdf",
            content_hash="a" * 64, page_start=3, page_end=3,
            section="Methods", ordinal=5,
            locator={"page": 3, "char_start": 100, "char_end": 110,
                     "bbox": [40, 80, 200, 95]}, text="Exact text",
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["chunk", str(chunk_id)])
        self.assertEqual(result, 0)
        saved = json.loads(output.getvalue())
        self.assertEqual(saved["paper_version_id"], str(version_id))
        self.assertEqual(saved["locator"]["page"], 3)

    @patch("tracescholar.cli.embed_research_run")
    def test_embed_command_reports_idempotent_counts(self, embed_research_run) -> None:
        run_id = uuid.uuid4()
        embed_research_run.return_value = SimpleNamespace(
            run_id=run_id, total_chunks=5460, provider="openai-compatible",
            model_name="qwen3.7-text-embedding", dimensions=1024,
            model_revision="a" * 64, newly_embedded=0, skipped_unchanged=5460,
            failed=0, pending=0, failures=(),
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["embed", str(run_id), "--limit", "20"])
        self.assertEqual(result, 0)
        self.assertIn("New embeddings: 0", output.getvalue())
        self.assertIn("Skipped unchanged: 5460", output.getvalue())
        embed_research_run.assert_called_once_with(run_id, limit=20)

    @patch("tracescholar.cli.search_chunks")
    def test_retrieve_command_reports_score_and_locator(self, search_chunks) -> None:
        run_id = uuid.uuid4()
        search_chunks.return_value = [SimpleNamespace(
            paper_title="A RAG paper", page_start=7, section="Experiments",
            similarity=0.81, cosine_distance=0.19,
            chunk_id=uuid.uuid4(), paper_version_id=uuid.uuid4(),
            locator={"page": 7, "char_start": 42, "char_end": 98},
            text="Query rewriting improves multi-hop QA.",
        )]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["retrieve", str(run_id), "multi-hop query rewriting", "--top-k", "1"])
        self.assertEqual(result, 0)
        self.assertIn("Page: 7", output.getvalue())
        self.assertIn("Similarity: 0.8100", output.getvalue())
        self.assertIn("char_start", output.getvalue())
        search_chunks.assert_called_once_with(run_id, "multi-hop query rewriting", top_k=1)

    @patch("tracescholar.cli.search_plan_evidence")
    def test_evidence_command_is_paper_and_subquestion_scoped(self, search_plan_evidence) -> None:
        run_id, paper_id = uuid.uuid4(), uuid.uuid4()
        search_plan_evidence.return_value = [SimpleNamespace(
            chunk_id=uuid.uuid4(), paper_version_id=uuid.uuid4(),
            paper_title="Paper", page_start=4, page_end=4, section="Results",
            similarity=0.82, locator={"page": 4, "char_start": 5, "char_end": 20},
            text="Supporting passage.",
        )]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["evidence", str(run_id), str(paper_id),
                           "--sub-question-index", "1", "--top-k", "2"])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue())["results"][0]["page_start"], 4)
        search_plan_evidence.assert_called_once_with(run_id, paper_id, 1, top_k=2)

    @patch("tracescholar.cli.screen_fulltext_research_run")
    def test_screen_fulltext_reports_final_corpus(self, screen_fulltext_research_run) -> None:
        run_id = uuid.uuid4()
        screen_fulltext_research_run.return_value = SimpleNamespace(
            run_id=run_id, candidates=24, newly_screened=0, skipped_unchanged=24,
            include=8, exclude=10, uncertain=6, failed=0, pending=0, failures=(),
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["screen-fulltext", str(run_id), "--limit", "10"])
        self.assertEqual(result, 0)
        self.assertIn("Final include: 8", output.getvalue())
        self.assertIn("Skipped unchanged: 24", output.getvalue())
        screen_fulltext_research_run.assert_called_once_with(run_id, limit=10)

    @patch("tracescholar.cli.get_fulltext_decision")
    def test_fulltext_decision_command_shows_cited_page(self, get_fulltext_decision) -> None:
        run_id, paper_id = uuid.uuid4(), uuid.uuid4()
        get_fulltext_decision.return_value = {
            "run_id": str(run_id), "paper_id": str(paper_id), "label": "include",
            "evidence": [{"page_start": 5, "chunk_id": str(uuid.uuid4())}],
        }
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["fulltext-decision", str(run_id), str(paper_id)])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue())["evidence"][0]["page_start"], 5)
        get_fulltext_decision.assert_called_once_with(run_id, paper_id)

    @patch("tracescholar.cli.generate_claims")
    def test_claims_command_reports_bounded_generation(self, generate_claims) -> None:
        run_id, generation_id = uuid.uuid4(), uuid.uuid4()
        generate_claims.return_value = SimpleNamespace(
            run_id=run_id, generation_id=generation_id, claims=2,
            newly_generated=0, skipped_unchanged=2, included_studies=10,
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["claims", str(run_id)])
        self.assertEqual(result, 0)
        self.assertIn("Claims: 2", output.getvalue())
        self.assertIn(str(generation_id), output.getvalue())
        generate_claims.assert_called_once_with(run_id)

    @patch("tracescholar.cli.extract_evidence")
    def test_extraction_command_reports_retryable_counts(self, extract_evidence) -> None:
        run_id = uuid.uuid4()
        extract_evidence.return_value = SimpleNamespace(
            run_id=run_id, claims=2, included_studies=10, extraction_tasks=20,
            newly_extracted=3, skipped_unchanged=0, no_evidence=1,
            failed=0, pending=17, spans=2, failures=(),
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["extract-evidence", str(run_id), "--limit", "3"])
        self.assertEqual(result, 0)
        self.assertIn("Extraction tasks: 20", output.getvalue())
        self.assertIn("EvidenceSpans: 2", output.getvalue())
        extract_evidence.assert_called_once_with(run_id, limit=3)

    @patch("tracescholar.cli.get_evidence_ledger")
    def test_ledger_command_emits_study_counts(self, get_evidence_ledger) -> None:
        run_id = uuid.uuid4()
        get_evidence_ledger.return_value = {
            "run_id": str(run_id), "claim_count": 1,
            "claims": [{"independent_evidence_studies": 2}],
        }
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["ledger", str(run_id)])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue())["claims"][0]
                         ["independent_evidence_studies"], 2)
        get_evidence_ledger.assert_called_once_with(run_id)

    @patch("tracescholar.cli.create_run_manifest")
    def test_manifest_command_reports_saved_snapshot(self, create_manifest) -> None:
        run_id, manifest_id = uuid.uuid4(), uuid.uuid4()
        create_manifest.return_value = SimpleNamespace(
            run_id=run_id, manifest_id=manifest_id, content_hash="f" * 64,
            created=True,
            manifest=SimpleNamespace(model_dump=lambda mode: {"manifest_version": 1}),
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["manifest", str(run_id)])
        self.assertEqual(result, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["manifest_id"], str(manifest_id))
        self.assertEqual(payload["manifest"], {"manifest_version": 1})
        create_manifest.assert_called_once_with(run_id)

    @patch("tracescholar.cli.get_run_manifest")
    def test_show_manifest_command_reads_saved_snapshot(self, get_manifest) -> None:
        run_id, manifest_id = uuid.uuid4(), uuid.uuid4()
        get_manifest.return_value = SimpleNamespace(
            run_id=run_id, manifest_id=manifest_id, content_hash="e" * 64,
            manifest=SimpleNamespace(model_dump=lambda mode: {"manifest_version": 1}),
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["show-manifest", str(manifest_id)])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue())["run_id"], str(run_id))
        get_manifest.assert_called_once_with(manifest_id)

    @patch("tracescholar.cli.write_grounded_review_export")
    @patch("tracescholar.cli.build_grounded_review_export")
    def test_export_command_reports_markdown_json_and_audit_counts(
        self, build_export, write_export,
    ) -> None:
        run_id, draft_id, manifest_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        sentence = SimpleNamespace(semantic_audit=SimpleNamespace(verdict="revise"))
        build_export.return_value = SimpleNamespace(
            run_id=run_id, draft_id=draft_id, manifest_id=manifest_id,
            content=SimpleNamespace(sections=[SimpleNamespace(paragraphs=[
                SimpleNamespace(sentences=[sentence]),
            ])]),
        )
        write_export.return_value = (Path("report.md"), Path("report.json"))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["export", str(run_id), "--manifest-id", str(manifest_id)])
        self.assertEqual(result, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["semantic_audit"], {"revise": 1})
        self.assertEqual(payload["manifest_id"], str(manifest_id))
        build_export.assert_called_once_with(run_id, manifest_id=manifest_id)
        write_export.assert_called_once()


if __name__ == "__main__":
    unittest.main()
