# TraceScholar

[简体中文](README.md) | English

**Version: 0.1.0**

TraceScholar is an evidence-first research assistant that turns a research
question into a reproducible, citation-grounded literature review draft. A
run records its plan, searches, screening decisions, paper versions, evidence,
draft, audits, and export so readers can trace factual sentences back to
quoted text and PDF pages.

## What v0.1 does

- Searches OpenAlex and Crossref, merges duplicate paper records, and keeps
  source-level search provenance.
- Screens papers in two stages: title/abstract, then relevant full-text chunks.
- Retrieves verified open-access PDFs when available; it does not bypass
  paywalls or login restrictions.
- Parses text into page-aware, section-aware chunks and stores cloud-generated
  embeddings in PostgreSQL with pgvector for run-scoped retrieval.
- Groups preprint and published records into canonical Studies without
  deleting the original records or PDF versions.
- Builds a Claim / Evidence Ledger with verbatim quotes, stances, and page
  locations.
- Writes a structured Grounded Review and exports deterministic Markdown and
  JSON.
- Runs deterministic citation-chain, semantic citation, and omitted-evidence
  audits, and records a Run Manifest.
- Supports a synchronous, resumable workflow that stops on completion, failure,
  or a blocked prerequisite.

## Limits and human review

- OCR is not implemented. Image-only PDFs without extractable text cannot be
  parsed.
- TraceScholar never bypasses paywalls, logins, or other access controls, and
  cannot guarantee that every candidate paper has an obtainable full text.
- Semantic Audit can return `revise` or `reject`. A suggested revision is not
  silently applied; review the original wording and recommendation yourself.
- TraceScholar is not a replacement for a systematic review or meta-analysis.
  Search coverage, provider metadata, model judgments, and extracted evidence
  can be incomplete or wrong and require researcher review.
- An open-access location permits retrieval under the locator's checks; it does
  not by itself grant permission to redistribute the PDF.

## Requirements

- Python 3.12 or newer
- Docker with Docker Compose, for the included PostgreSQL 16 + pgvector service
- Network access and credentials for the configured LLM and embedding services

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Run the CLI:

```bash
tracescholar
tracescholar --version
```

Expected output:

```text
TraceScholar is ready.
```

## Configuration

All application configuration is defined in `tracescholar.config.Settings`.
Application modules should call `get_settings()` instead of reading environment
variables directly.

Copy the example file for local development:

```bash
cp .env.example .env
```

Configuration can be provided through `.env` or environment variables. All
variables use the `TRACESCHOLAR_` prefix, and environment variables take
precedence over values from `.env`.

Set at least `TRACESCHOLAR_LLM_BASE_URL`, `TRACESCHOLAR_LLM_API_KEY`, and
`TRACESCHOLAR_LLM_MODEL` for an OpenAI-compatible Chat Completions service.
Embedding uses a separate compatible service and key; the example defaults to
Alibaba Cloud's `qwen3.7-text-embedding` at 1024 dimensions. OpenAlex API key
and Crossref email are optional. Never commit `.env` or put real credentials
in `.env.example`. Configuration is centralized in `Settings`, and secret
values are masked in logs and object representations.

## Database

The local database runs PostgreSQL 16 with the pgvector extension through
Docker Compose.

Start the database:

```bash
docker compose up -d postgres
```

Apply all migrations:

```bash
alembic upgrade head
```

This applies the complete migration chain, including the pgvector extension
and all current tables. Schema changes are managed through Alembic migrations.

Create a research run:

```bash
tracescholar research \
  "How does query rewriting affect multi-hop RAG?" \
  --scope-json '{"year_from": 2023, "languages": ["en"]}'
```

Example output:

```text
Created ResearchRun <RUN_ID> [pending]
```

Copy the generated ID and run the complete workflow:

```bash
tracescholar run <RUN_ID>
tracescholar workflow-status <RUN_ID>
```

`run` advances stages in order and stops when the run is `completed`, a stage
`failed`, or a prerequisite is `blocked`. Rerun the same command to resume from
the unfinished stage; completed stages are not needlessly repeated.
This workflow makes requests to the configured paper sources, LLM, and
embedding service as needed, and may consume their quotas or incur usage costs.

After the workflow creates a manifest, `manifest` prints its ID and content
hash, while `show-manifest` reads and validates that saved snapshot:

```bash
tracescholar manifest <RUN_ID>
tracescholar show-manifest <MANIFEST_ID>
```

Export the saved draft and audit state without rerunning research or calling an
LLM:

```bash
tracescholar export <RUN_ID> --output-dir ./review-export
```

This writes `report.md` and `report.json` under `./review-export`. By default,
exports go to `data/exports/<RUN_ID>/`, which is excluded from Git.

## Scope Planner

Configure your OpenAI-compatible provider in `.env` (never in `.env.example`):

```dotenv
TRACESCHOLAR_LLM_BASE_URL=https://your-provider.example/v1
TRACESCHOLAR_LLM_MODEL=your-model-id
TRACESCHOLAR_LLM_API_KEY=your-secret-key
TRACESCHOLAR_LLM_STRUCTURED_MODE=json_schema
```

`json_schema` asks the provider to enforce a strict schema. If the provider
implements only JSON mode, set `TRACESCHOLAR_LLM_STRUCTURED_MODE=json_object`;
TraceScholar still validates the returned JSON with Pydantic and rejects
invalid or incomplete plans. The configured endpoint must support the
OpenAI-compatible Chat Completions API and the selected response-format mode.
The official OpenAI endpoint also works when configured with its own key;
`TRACESCHOLAR_OPENAI_API_KEY` is retained as a fallback for that endpoint.

Generate the plan for an existing run:

```bash
tracescholar plan <RESEARCH_RUN_UUID>
```

The CLI prints a JSON object containing `run_id` and the typed `ResearchPlan`.
The plan has a normalized question, sub-questions, concepts with synonyms and
abbreviations, exclusion terms, inclusion/exclusion criteria, constraints,
ambiguities, search tracks, a counter-evidence track, stop conditions, and an
exact copy of the user's scope. It does **not** answer the question or execute
its suggested searches. Later Python stages can call
`tracescholar.planning.get_research_plan(run_id)` and receive the same fixed
Pydantic schema, not raw model prose.

Planning is frozen: calling `plan` again returns the stored plan without an
additional LLM request. The database record also keeps the input question and
scope, model name, prompt version, and schema version. If the run's question or
scope changes afterward, reloading raises `PlanFrozenError` instead of silently
using a stale plan.

The LLM gateway follows the [OpenAI Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs)
for JSON Schema formatting and checks completion/refusal before local schema
validation. Provider-specific compatibility should be verified with that
provider's API documentation.

## Automatic discovery from a frozen plan

After `tracescholar plan <RESEARCH_RUN_UUID>`, run:

```bash
tracescholar discover <RESEARCH_RUN_UUID> --limit 5
```

The deterministic Query Generator reads the frozen plan; it makes no additional
LLM call. It creates at most two plain keyword queries per search track and
at most 12 overall (six tracks), including a distinct counter-evidence query.
The `PlannedQuery` inventory records each query's purpose, precise/expanded
variant, source track, sub-question index, and generator version. Every
successful provider execution creates an existing `SearchQuery` linked to that
inventory, followed by source hits and deduplicated papers. Failed executions
leave the planned query intact and are retried on the next run; successful
query/source pairs are never re-executed by this command.

Supported hard scope keys are `year_from`, `year_to`, and either `language` or
`languages`. The generator cannot change them. Both adapters send date filters
to their providers, OpenAlex also receives a language filter, and TraceScholar
post-filters every result before persistence. Crossref does not provide a
language query filter, so results with absent or mismatched language metadata
are excluded when the frozen scope specifies a language. Unsupported scope keys
cause a clear error before any request rather than being silently ignored.

Search and paper counts depend on provider data, the frozen scope, and the
selected limit. Successful query/source pairs are reused on repeat runs;
unsuccessful pairs can be retried.

## Manual discovery data

You can record fake search results without an external paper API:

```python
from tracescholar.database import get_session_factory, session_scope
from tracescholar.models import ResearchRun
from tracescholar.repositories import (
    add_search_result,
    create_research_run,
    create_search_query,
    upsert_paper,
)

run = create_research_run("How does query rewriting affect RAG?")

with session_scope() as session:
    query = create_search_query(
        session,
        run_id=run.id,
        query="query rewriting RAG",
        source="openalex",
        filters={"from_year": 2024},
    )
    paper = upsert_paper(
        session,
        title="A Study of Query Rewriting",
        doi="https://doi.org/10.5555/example",
        year=2025,
        authors=["Example Author"],
        abstract="Example abstract.",
    )
    add_search_result(
        session,
        search_query=query,
        paper=paper,
        source_record_id="W123",
    )

# Reload through a new session to verify persistence.
with get_session_factory()() as session:
    saved_run = session.get(ResearchRun, run.id)
    print(saved_run.search_queries[0].source)
    print(saved_run.papers[0].title)
```

`SearchQuery.source` records which provider executed the query, while each
`SearchResult` stores a paper hit and its provider record ID. A paper may be
found by several queries or in several research runs without duplicating its
canonical metadata.

## Multi-source discovery

Run the same keyword search against OpenAlex and Crossref, then save both
providers' results in an existing run:

```bash
tracescholar search <RESEARCH_RUN_UUID> "retrieval augmented generation" --limit 20
```

Provider result counts and rankings change over time. The command reports the
actual counts for each execution rather than assuming fixed search results.

The command reports returned and persisted hits per provider, total raw hits,
unique papers, and merged duplicates. Each provider gets a separate
`SearchQuery`. Its results retain the OpenAlex work ID or Crossref DOI in
`SearchResult.source_record_id`. If both providers find the same paper, the
two source hits point to a single canonical `Paper`.

The unified entry point for other Python modules is
`tracescholar.discovery.discover_papers(run_id, query)`. It accepts any sequence
of `PaperSource` implementations. Each provider is persisted in its own
transaction, so a failed provider does not discard successful results from
another. Failures appear in the returned `DiscoverySummary`; if every provider
fails, the CLI exits with a nonzero status. A failed provider does not leave a
`SearchQuery`, matching the single-source behavior.

`TRACESCHOLAR_OPENALEX_API_KEY` is optional for small experiments. Configure a
free key in `.env` for regular use; OpenAlex can temporarily limit anonymous
search traffic. Requests use a timeout and one bounded retry for temporary
HTTP errors. OpenAlex currently supports 1–100 results per page, and this
command fetches the first page only.

Both adapters convert provider-specific fields into the common `PaperMetadata`
structure before any database writes. Rows without a usable source ID or title
are counted as skipped; optional metadata may remain empty. An empty search
is still recorded as a `SearchQuery`.

Crossref uses its public `/works` endpoint with `query.bibliographic`. Set
`TRACESCHOLAR_CROSSREF_EMAIL` in `.env` to identify the application to
Crossref's polite pool; no Crossref API key is needed for this public API.
The adapter maps DOI, title, authors, venue, publication year, and abstract;
Crossref's tagged abstracts are converted to plain text. It uses a timeout and
one bounded retry for temporary HTTP errors.

OpenAlex references: [search syntax](https://help.openalex.org/api/searching/),
[work fields](https://help.openalex.org/data/works/attributes/), and
[API authentication](https://help.openalex.org/api/authentication/).
Crossref references: [REST API](https://www.crossref.org/documentation/retrieve-metadata/rest-api/),
[query and result options](https://www.crossref.org/documentation/retrieve-metadata/rest-api/tips-for-using-the-crossref-rest-api/), and
[polite-pool identification](https://www.crossref.org/documentation/retrieve-metadata/rest-api/access-and-authentication/).

## Title/abstract screening

After planning and discovery, screen the run's deduplicated papers:

```bash
tracescholar screen <RESEARCH_RUN_UUID>
```

Use `--limit N` to process at most N pending papers per invocation. The
screener sends each paper separately through the same configured LLM gateway
used by the Scope Planner. Its strict decision schema has only `include`,
`maybe`, and `exclude` labels, a 0–1 relevance score, rationale, matched
inclusion/exclusion criteria, full-text-needed flag, sub-question index, and
evidence role. The prompt prioritizes recall: uncertain papers remain `maybe`;
papers with no abstract are always `maybe` and marked for full-text review.
The screener does not download or analyze full text.

Every successful decision is saved in `screening_results`, including excluded
papers. The row records the exact plan and paper metadata snapshot, model,
prompt version, and input hash. Re-running screening skips unchanged papers;
changed paper metadata or a different model/prompt version triggers a new
decision. Individual model/schema failures are reported without fabricating a
screening result and can be retried by running the same command again. The
stored matched criteria are the exact frozen-plan strings, not free-form
criteria invented by the model.

These are provisional screening judgments, not verified findings or full-text
eligibility decisions. Unchanged successful decisions are skipped on reruns.

## Open-access PDF acquisition

Only `include` and `maybe` papers with current screening inputs are eligible:

```bash
tracescholar acquire <RESEARCH_RUN_UUID>
```

Use `--limit N` for a bounded batch. `--retry-unavailable` rechecks papers
inside the 24-hour unavailable cache window. A verified OA location is required:
OpenAlex must mark the work and location as open access, or the paper must have
an arXiv ID. The locator may use OpenAlex's OA PDF content endpoint when the
configured key is available. DOI publisher landing pages are never treated as
PDF authorization; no paywall or login is bypassed. Unknown license values are
stored as unknown, not interpreted as reuse permission.

The downloader accepts only approved HTTPS OA hosts and safe redirects, caps
files at `TRACESCHOLAR_FULLTEXT_MAX_BYTES` (20 MB by default), and checks PDF
headers and EOF before saving. Every file is stored under `data/fulltext/sha256/`
by SHA-256 hash, so identical content occupies one local path. `PaperVersion`
records the paper, hash, source URL, source name, license, version label,
retrieval time, and size. `FullTextAcquisition` stores each run's current
`downloaded`, `cached`, `unavailable`, or `failed` outcome, attempt count,
source, and failure reason. Failed downloads retry on the next run; unavailable
locations are rechecked after the configured delay. A valid cached success is
not downloaded again.

This stage only fetches and validates files. It does not parse PDF content,
screen full text, create chunks, embeddings, claims, or evidence spans.

OpenAlex documents its [OA location evidence](https://help.openalex.org/data/works/open-access/)
and [full-text endpoint and copyright boundary](https://help.openalex.org/access/fulltext/).

OA availability varies by paper and source. Every result is recorded per run;
valid cached files are reused instead of downloaded again.

## Page-aware PDF parsing

Parse the acquired versions of a run after applying the latest migration:

```bash
alembic upgrade head
tracescholar parse <RESEARCH_RUN_UUID>
```

Inspect any stored chunk with `tracescholar chunk <CHUNK_UUID>`; the JSON
response identifies its paper, PDF version, physical page, section, exact
page-text offsets, and PDF-point bounding box.

`--limit N` bounds a batch. The parser checks the cached file's size and
SHA-256 before reading it, extracts text in page order (including common
two-column layouts), omits repeated page margins and the references section,
and groups adjacent lines by section and paragraph before length-bounding
chunks. This is text extraction, not OCR: image-only PDFs are recorded as
`no_text` failures. This version does not include an OCR stage.

`ParsedPage` retains the normalized text of every physical page, including
empty pages. `Chunk` links to one immutable `PaperVersion` and stores a section,
ordinal, page number, document character range, estimated token count, and a
locator. `locator.char_start`/`char_end` index the corresponding
`ParsedPage.text` exactly; `locator.bbox` is `[x0, top, x1, bottom]` in PDF
points from the page's top-left corner. A chunk can therefore be checked
against its page text and located visually in the source PDF. Page-local chunks
avoid ambiguous cross-page coordinates. No embedding or vector write occurs.

`PdfParseRecord` stores parser version, input hash, page/chunk counts, quality
flags, success/failure state, and failure reason. Successful unchanged
versions are skipped on rerun. A damaged or missing PDF affects only its own
version; failed versions are retried next time. Layout heuristics are
best-effort: complex tables, equations, and unusual typography may still need
manual review before downstream evidence extraction.

Successful unchanged versions are skipped on reruns. Parser quality flags and
failure reasons remain attached to each version for inspection.

## Cloud embeddings and run-scoped retrieval

Configure the separate Embeddings API key in the untracked `.env` file:

```dotenv
TRACESCHOLAR_EMBEDDING_API_KEY=your-private-key
```

The default endpoint/model is the user-selected Alibaba Cloud OpenAI-compatible
service (`https://dashscope.aliyuncs.com/compatible-mode/v1`,
`qwen3.7-text-embedding`, 1024 dimensions). Override `TRACESCHOLAR_EMBEDDING_BASE_URL`,
`TRACESCHOLAR_EMBEDDING_MODEL`, and `TRACESCHOLAR_EMBEDDING_DIMENSIONS` if your
service uses a different deployment. The chat LLM key is never reused for
embeddings. No local model or model cache is required. Alibaba Cloud's
[Embeddings documentation](https://help.aliyun.com/en/model-studio/embedding-interfaces-compatible-with-openai)
specifies this model's 20-input synchronous batch limit; TraceScholar caps its
batch size accordingly.

```bash
alembic upgrade head
tracescholar embed <RESEARCH_RUN_UUID>
tracescholar retrieve <RESEARCH_RUN_UUID> \
  "Does query rewriting improve multi-hop question answering?" --top-k 10
```

`tracescholar embed ... --limit N` processes a bounded, resumable subset. Each
`ChunkEmbedding` stores the vector in PostgreSQL `pgvector`, its source Chunk,
text SHA-256, endpoint, model, configured model version, output dimension, SDK
version, success/failure state, and retry count. Unchanged successes skip the
API call; failed chunks retry. If the provider offers a pinned model version,
set `TRACESCHOLAR_EMBEDDING_MODEL_VERSION` to it. A mutable model alias cannot
guarantee identical future vectors even when its name stays unchanged.

Retrieval uses cosine distance in PostgreSQL and only joins PDF versions
acquired by the requested `ResearchRun`. Results carry the full
`Chunk → PaperVersion → Paper` chain, physical page, section, character locator,
distance, and `1 - distance` similarity. At the current corpus size, the
database performs exact top-k vector search rather than approximate indexing,
so filtering by run does not discard candidates. This stage does not perform
reranking, claim extraction, or report generation.

Embedding and retrieval are scoped to the requested run. Unchanged successful
embeddings are reused; failed chunks can be retried independently.

## Full-text screening and evidence retrieval

The second screening stage only considers papers whose title/abstract decision
is `include` or `maybe`. It retrieves the top two page-located chunks per
frozen ResearchPlan sub-question **inside each paper's PDF version**, caps the
model context at eight distinct chunks, and asks the shared LLM gateway for a
strict `include`, `exclude`, or `uncertain` decision. `exclude` requires an
explicit plan exclusion criterion and a cited chunk; absence from a sampled
passage is not treated as proof of exclusion. Parser quality warnings (including
`low_page_coverage`) stay visible to the model and in the saved decision.

```bash
alembic upgrade head
tracescholar evidence <RESEARCH_RUN_UUID> <PAPER_UUID> --sub-question-index 0 --top-k 3
tracescholar screen-fulltext <RESEARCH_RUN_UUID>
tracescholar fulltext-decision <RESEARCH_RUN_UUID> <PAPER_UUID>
```

`fulltext-decision` reloads the decision rationale, matched plan criteria,
supported sub-questions, evidence role, warnings, and cited Chunk IDs with PDF
version, page, section and character/bounding-box locator. Per-paper LLM or
retrieval failures are recorded and retried independently. A successful result
is skipped when its frozen plan, PDF version, parser state, retrieval model,
and screening prompt/model have not changed. Full-text screening uses a
separate 180-second LLM timeout by default, configurable with
`TRACESCHOLAR_FULLTEXT_SCREENING_LLM_TIMEOUT_SECONDS`; the regular Planner and
title/abstract screening timeout is unchanged.

This step finds candidate evidence only. It does not create `EvidenceSpan` or
`Claim` records, analyze conflicts, or write a review.

## Canonical studies and publication versions

`Paper` remains a bibliographic/source record and `PaperVersion` remains an
immutable PDF acquisition. A `CanonicalStudy` groups records that represent
one independent research contribution; `StudyPaper` records each membership
and its publication role. A confirmed link requires strong identifier,
content, or title/author/abstract agreement. Similar but insufficiently
corroborated pairs stay as `unresolved` `StudyLinkCandidate` rows and are
counted separately until reviewed. A reviewer can confirm or reject such a
pair with `tracescholar study-link <RUN_UUID> <PAPER_A_UUID> <PAPER_B_UUID>
confirmed --reason "..."` (or `rejected`), then rerun `studies`. No `Paper`
or PDF is deleted.

```bash
alembic upgrade head
tracescholar studies <RESEARCH_RUN_UUID>
tracescholar study <RESEARCH_RUN_UUID> <STUDY_UUID>
```

The per-run `StudyRunSelection` picks one default PDF for Claim/Evidence
extraction: a well-parsed published version is preferred; a usable preprint
can take precedence over a damaged or incomplete published PDF. The choice
stores a policy and input fingerprint, so reruns reuse unchanged decisions.
The detailed view includes every source URL, acquisition time, license and
content hash. `StudyVersionComparison` records whether PDFs are byte-identical,
have not been assessed, or have human-reviewed equivalent/changed results.
Different hashes alone **do not** imply a changed scientific result. To
record a reviewed comparison:

```bash
tracescholar study-compare <RUN_UUID> <STUDY_UUID> <PDF_A_UUID> <PDF_B_UUID> \
  changed --note "Table 3 reports a changed F1 result"
```

A `changed` comparison causes the extraction-version plan to retain both PDFs
for separate evidence extraction, while the independent-study count remains one.
This stage does not create claims, evidence spans, conflict analyses or reports.

## Evidence Ledger

Only final-`include` canonical studies enter this stage. Claim generation reads
the frozen ResearchPlan and a small, result-section-prioritized set of full-text
screening citations. It asks the shared structured LLM gateway for at most two
atomic candidate assertions, each backed by an exact substring of an offered
Chunk and its physical parsed PDF page. Each candidate declares whether it is
`study_specific` (a paper's own dataset/method/result, checked only within
that Study) or `cross_study` (an operationally comparable proposition, checked
across the included corpus). These are **candidate** claims, not established
conclusions. The second pass evaluates each applicable claim using semantic
search limited to the Study's selected, legally acquired PDF version. A human-reviewed
`changed` version comparison can make both PDF versions eligible, but the
ledger still counts one independent Study.

```bash
alembic upgrade head
tracescholar claims <RESEARCH_RUN_UUID>
tracescholar extract-evidence <RESEARCH_RUN_UUID> --limit 10
tracescholar extract-evidence <RESEARCH_RUN_UUID>
tracescholar ledger <RESEARCH_RUN_UUID>
tracescholar evidence-span <RESEARCH_RUN_UUID> <EVIDENCE_SPAN_UUID>
```

An `EvidenceSpan` stores the verbatim quote, stance (`supports`, `contradicts`,
`qualifies`, or `unrelated`), confidence, rationale, context, limitations,
Study, PaperVersion, Chunk, physical PDF page, section, and both chunk-local
and page-local character offsets. Before persistence, code verifies the exact
quote against **both** the retrieved Chunk and its `ParsedPage` at those offsets.
An insufficient passage produces a durable `no_evidence` result, not an
invented quote; malformed model output becomes a retryable per-study failure.
Different benchmark lists or a paper's silence on a method are not treated as
contradictions. Cross-study `contradicts` labels require an explicit opposing
result in the quote; this conservative guard may miss implicit contradictions.
The Evidence Ledger is an auditable record, not a complete conflict analysis.
Input fingerprints skip successful unchanged work, while preserving prior
attempts when a PDF, retrieval model, or prompt changes. The Evidence stage
uses `TRACESCHOLAR_EVIDENCE_LLM_TIMEOUT_SECONDS` independently of other LLM
tasks. The ledger reports both raw span counts and distinct-study counts, so
preprint/published versions cannot inflate independent support.

The read-only Evidence aggregation layer behind `tracescholar ledger` groups
saved EvidenceSpans by Claim and canonical Study. Its structured `outcomes`
include span counts and deduplicated Study IDs/counts for `supports`,
`contradicts`, and `qualifies`, plus both version-level task and Study-level
`no_evidence` counts. `study_results` retains version IDs, stance labels,
no-evidence reasons, and incomplete/failed version status. A Study with two
eligible PDFs is `no_evidence` only if **both** completed without evidence;
different stances from one Study may overlap, so stance-study counts must not
be summed as independent studies. Aggregation only reads persisted rows: it
does not call an LLM, perform retrieval, or write a review.

The Evidence Ledger stage does not build a Claim Graph, classify conflicts,
write a review, audit citations, or produce a gap report.

## Structured synthesis and Markdown rendering

The Synthesis Writer reads the complete Evidence Ledger and frozen research
question, then asks the shared LLM gateway for a `SynthesisDocument` containing
sections, paragraphs, sentences, `claim_ids`, and `evidence_ids`. It checks
that each sentence cites at least one Claim and EvidenceSpan, every referenced
Claim belongs to the current generation, and every EvidenceSpan belongs to a
Claim cited in the same sentence. A validated draft
is stored with its input snapshot, model, prompt version, and schema version;
unchanged inputs reuse the saved draft. Failed attempts can be retried.

```bash
alembic upgrade head
tracescholar synthesize <RESEARCH_RUN_UUID>
tracescholar render-synthesis <RESEARCH_RUN_UUID>
tracescholar render-synthesis <RESEARCH_RUN_UUID> --draft-id <SYNTHESIS_DRAFT_UUID>
tracescholar audit-citations <RESEARCH_RUN_UUID>
tracescholar audit-citations <RESEARCH_RUN_UUID> --draft-id <SYNTHESIS_DRAFT_UUID>
tracescholar audit-semantics <RESEARCH_RUN_UUID>
tracescholar audit-semantics <RESEARCH_RUN_UUID> --draft-id <SYNTHESIS_DRAFT_UUID>
tracescholar audit-omissions <RESEARCH_RUN_UUID>
tracescholar audit-omissions <RESEARCH_RUN_UUID> --draft-id <SYNTHESIS_DRAFT_UUID>
```

The Renderer reads a saved structured draft and resolves cited EvidenceSpan
IDs against database records. It deterministically emits Markdown to standard
output, with footnotes built from the stored paper metadata, PDF version,
page, section, stance, exact quote, Claim ID, and Study ID. It does not call an
LLM or export a report file.

The deterministic Citation Auditor independently checks each factual
sentence's Claim and EvidenceSpan IDs against the saved draft generation,
then verifies the EvidenceSpan → Study → PaperVersion → Chunk → ParsedPage
chain, including PDF page number, section, exact quote, and character offsets.
Each sentence stores its own Claim/Evidence/lineage snapshot, input hash, and
passed/failed issues. Unchanged sentences reuse their records while changed
sentences are re-audited; the draft-level status and counts are derived from
those sentence records. It does not call an LLM, change the draft, judge
whether prose is semantically supported, or inspect the PDF bytes themselves.

The single-sentence Semantic Citation Auditor runs only after the deterministic
chain audit passes. It sends one sentence, its cited Claims, and the exact
EvidenceSpan quotes/context to the configured LLM, then persists entailment,
scope, strength, a `pass`/`revise`/`reject` verdict, and an optional suggested
minimal revision. The original draft is never changed. Successful unchanged
sentences are cached; failed calls are recorded and can be retried. This stage
does not search for omitted counterevidence or audit the report as a whole.

The sentence-level Omission Auditor reads every current EvidenceSpan for each
cited Claim that the sentence does not cite, regardless of its stored stance.
After verifying candidate provenance, it checks whether the quote materially
contradicts the sentence, limits its scope, or weakens its stated strength. It
persists `pass`/`revise`/`flag`, omitted Evidence IDs, per-ID impact types, and a
short rationale. Sentences with no candidates pass deterministically. It does
not modify EvidenceSpan stance or the draft, propose replacement prose, or
classify study conflicts.

## Run Manifest

The Run Manifest is a versioned JSON snapshot assembled only from persisted
records for one `ResearchRun`. It captures the frozen plan, planned and
executed queries, providers, final study decisions, used PDF versions, the
current synthesis draft, evidence citations, audit summaries, stored model and
prompt/schema versions, and known embedding configuration. It never calls an
LLM. Repeating the command against unchanged records reuses the same
content-addressed snapshot; changed run facts create a new immutable manifest.

```bash
alembic upgrade head
tracescholar manifest <RESEARCH_RUN_UUID>
tracescholar show-manifest <RUN_MANIFEST_UUID>
```

The JSON is schema-validated both when created and when read back. Secrets are
redacted and URL query parameters are omitted. Historical fields that were not
stored at the time (such as LLM endpoint URLs and some stage schema versions)
are listed as capture gaps instead of being inferred from today's environment.
Stage time spans are computed from persisted event timestamps and are not
wall-clock execution measurements.

## Grounded Review export

Export a saved draft and its selected RunManifest without rerunning research,
LLM calls, or audits:

```bash
tracescholar export <RESEARCH_RUN_UUID>
tracescholar export <RESEARCH_RUN_UUID> --manifest-id <RUN_MANIFEST_UUID>
tracescholar export <RESEARCH_RUN_UUID> --output-dir ./review-export
```

The command writes deterministic `report.md` and `report.json`. Each Markdown
sentence keeps its saved wording and cites EvidenceSpan, Study, PaperVersion,
and PDF page provenance. Semantic `revise` suggestions appear only as separate
“not applied” review notes; `reject` sentences remain explicitly marked and
are never presented as ordinary supported facts. JSON retains original
paragraph/sentence structure, Claim/Evidence IDs, evidence provenance, and
sentence-level citation, semantic, and omission audit states. The selected
Manifest ID and content hash are included in JSON.

DOI and versionless arXiv ID are normalized and unique in the database. When
neither identifier finds a record, the repository looks for a matching
normalized title and year. The `(normalized_title, year)` index accelerates
that lookup but is not unique: different papers can share a title. Conflicting
identifiers raise `PaperIdentityConflict` instead of silently merging records.

## Resumable workflow

The workflow service reads saved stage outputs and reports the first unfinished
stage. It records each single-stage attempt, start/finish times, duration, and
failure reason. A failed stage stays the next stage to retry; earlier research
records are left intact. Stages are dispatched through their existing services.

Use `workflow-step` to advance one stage for inspection, or `run` to continue
through all remaining stages synchronously:

```bash
alembic upgrade head
tracescholar workflow-status <RESEARCH_RUN_UUID>
tracescholar workflow-step <RESEARCH_RUN_UUID>
tracescholar workflow-step <RESEARCH_RUN_UUID> --stage acquired
tracescholar run <RESEARCH_RUN_UUID>
```

`workflow-step` executes at most one eligible stage. The optional `--stage`
acts as an order check and refuses to jump ahead. `run` synchronously repeats
that same one-stage operation until the run is complete or a stage fails or is
blocked; it reports stage results as they finish and can be rerun to resume.
It does not start a background job. The ordered stages are
`planned → discovered → screened → acquired → parsed → embedded →
fulltext_screened → studies_normalized → evidence_extracted → synthesized →
audited → manifested → exported`. Stages with no applicable inputs (for
example, embedding when there are no parsed chunks) are reported as completed
with a reason; PDF parsing itself requires at least one acquired PDF version.

Stop the database without deleting its volume:

```bash
docker compose stop postgres
```

## Run tests

The test suite uses Python's standard library and requires no extra
testing dependencies:

```bash
python -m unittest discover -s tests
```

## v0.1 implemented scope

Included:

- `src`-layout Python package
- installable `tracescholar` command
- automated tests for the CLI and research stages
- centralized environment and `.env` configuration
- PostgreSQL 16 with pgvector through Docker Compose
- SQLAlchemy session and transaction management
- Alembic migrations
- persisted `ResearchRun` records
- `SearchQuery` and `Paper` models with query-level source provenance
- identifier and title-based paper deduplication
- OpenAlex and Crossref keyword search through a provider-neutral source interface
- multi-source discovery, partial-failure handling, and provenance-preserving deduplication
- a reusable structured-output LLM gateway for compatible Chat Completions APIs
- frozen, validated `ResearchPlan` records and a Scope Planner CLI command
- deterministic planned queries and scoped, resumable multi-source discovery
- persistent title/abstract screening with `include`/`maybe`/`exclude`
- verified OA PDF acquisition with versioned, hash-addressed local cache
- PDF text parsing with physical pages, section-aware chunks, locators, and retryable status
- separate cloud Chunk Embeddings API, versioned pgvector storage, and run-scoped semantic retrieval
- paper-scoped, plan-sub-question evidence retrieval and auditable full-text screening
- non-destructive canonical-study grouping, version provenance and extraction PDF selection
- quote-grounded Claim generation and page-verified, study-aware Evidence Ledger
- fixed structured synthesis schema linking sentences to Claims and EvidenceSpans
- saved structured synthesis drafts and deterministic, database-backed Markdown rendering
- per-sentence deterministic, semantic, and omitted-evidence audit persistence
- versioned, content-addressed RunManifest snapshots of persisted run facts
- deterministic Grounded Review Markdown and JSON export with audit-state annotations
- synchronous resumable workflow, one-stage controls, order checks, and persisted retry history

Out of scope for v0.1:

- OCR, reranking, Claim Graph, complex conflict analysis, and gap reports
- systematic-review protocol management or meta-analysis
- BibTeX/CSV export, a web UI, and an HTTP API
- academic paper sources beyond OpenAlex and Crossref

## License

TraceScholar is licensed under the Apache License, Version 2.0. See
[LICENSE](LICENSE) for the complete license text.
