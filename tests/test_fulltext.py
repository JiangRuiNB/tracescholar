"""OA location, bounded PDF download, caching and acquisition state tests."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from sqlalchemy import create_engine, func, select
from sqlalchemy.pool import StaticPool

from tracescholar.config import Settings
from tracescholar.database import Base, create_session_factory, session_scope
from tracescholar.fulltext import (
    DocumentFetcher, FetchError, FetchedPDF, FullTextLocation, FullTextPaper,
    OpenAlexOALocator, acquire_fulltext,
)
from tracescholar.models import FullTextAcquisition, Paper, PaperVersion, ResearchRun, ScreeningResult
from tracescholar.planning import ResearchPlan
from tracescholar.repositories import (
    add_search_result, create_research_run, create_search_query, save_research_plan,
    upsert_paper,
)
from tracescholar.screening.service import _fingerprint, _input_snapshot, _paper_snapshot


PDF = b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\n%%EOF\n"
URL = "https://aclanthology.org/2025.acl-long.1.pdf"


def _plan() -> ResearchPlan:
    return ResearchPlan.model_validate({
        "normalized_question": "RAG query rewriting?", "sub_questions": ["Does it help?"],
        "concepts": [{"term": "RAG", "synonyms": [], "abbreviations": []}],
        "exclusion_terms": [], "inclusion_criteria": ["Relevant RAG"],
        "exclusion_criteria": ["Unrelated"], "constraints": [], "ambiguity_items": [],
        "search_tracks": [{"label": "negative", "intent": "counter_evidence",
                           "query": "RAG failures", "rationale": "Counter evidence"}],
        "stop_conditions": ["Saturation"], "scope_snapshot": {},
    })


class LocatorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(_env_file=None, openalex_api_key="test-key")
        self.paper = FullTextPaper(uuid.uuid4(), "RAG paper", "10.5555/test", "2501.12345", ("W123",))

    def test_only_verified_oa_pdf_locations_and_license_are_returned(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/works/W123")
            return httpx.Response(200, json={
                "id": "https://openalex.org/W123", "doi": "https://doi.org/10.5555/test",
                "open_access": {"is_oa": True},
                "best_oa_location": {"is_oa": True, "pdf_url": URL,
                    "license": "cc-by", "version": "publishedVersion",
                    "source": {"display_name": "ACL Anthology"}},
                "locations": [
                    {"is_oa": False, "pdf_url": "https://aclanthology.org/closed.pdf"},
                    {"is_oa": True, "pdf_url": "https://localhost/internal.pdf"},
                ],
                "has_content": {"pdf": True},
                "content_urls": {"pdf": "https://content.openalex.org/works/W123.pdf"},
            })
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            locations = OpenAlexOALocator(settings=self.settings, client=client).locate(self.paper)
        self.assertEqual([item.url for item in locations], [
            URL, "https://content.openalex.org/works/W123.pdf",
            "https://arxiv.org/pdf/2501.12345",
        ])
        self.assertEqual(locations[0].license, "cc-by")
        self.assertEqual(locations[0].version_label, "publishedVersion")

    def test_non_oa_work_is_unavailable_and_404_keeps_arxiv_fallback(self) -> None:
        with httpx.Client(transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={
                "id": "https://openalex.org/W123", "open_access": {"is_oa": False},
                "locations": [{"is_oa": False, "pdf_url": URL}],
            })
        )) as client:
            locations = OpenAlexOALocator(settings=self.settings, client=client).locate(self.paper)
        self.assertEqual([item.source_name for item in locations], ["arXiv"])
        with httpx.Client(transport=httpx.MockTransport(
            lambda _: httpx.Response(404)
        )) as client:
            locations = OpenAlexOALocator(settings=self.settings, client=client).locate(self.paper)
        self.assertEqual(len(locations), 1)
        self.assertEqual(locations[0].source_name, "arXiv")


class FetcherTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(_env_file=None, fulltext_max_bytes=100_000)
        self.location = FullTextLocation(URL, "ACL Anthology", "cc-by", "publishedVersion")

    def _fetch(self, handler):
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            return DocumentFetcher(settings=self.settings, client=client).fetch(self.location)

    def test_accepts_real_pdf_bytes_and_hashes_them(self) -> None:
        document = self._fetch(lambda _: httpx.Response(
            200, content=PDF, headers={"Content-Type": "application/pdf"}
        ))
        self.assertEqual(document.content_hash, hashlib.sha256(PDF).hexdigest())
        self.assertEqual(document.content, PDF)

    def test_records_404_timeout_non_pdf_and_oversize(self) -> None:
        cases = (
            (lambda _: httpx.Response(404), "not_found"),
            (lambda _: httpx.Response(200, text="<html>paywall</html>",
                                     headers={"Content-Type": "text/html"}), "not_pdf"),
            (lambda _: httpx.Response(200, content=b"not a pdf",
                                     headers={"Content-Type": "application/pdf"}), "not_pdf"),
            (lambda _: httpx.Response(200, content=b"%PDF-1.4\ntruncated",
                                     headers={"Content-Type": "application/pdf"}), "not_pdf"),
            (lambda _: httpx.Response(200, content=PDF,
                                     headers={"Content-Length": "100001"}), "too_large"),
            (lambda _: httpx.Response(200, content=b"x" * 100001,
                                     headers={"Content-Type": "application/pdf"}), "too_large"),
            (lambda _: httpx.Response(403), "access_denied"),
        )
        for handler, code in cases:
            with self.subTest(code=code), self.assertRaises(FetchError) as raised:
                self._fetch(handler)
            self.assertEqual(raised.exception.code, code)

        def timeout(request):
            raise httpx.ReadTimeout("timeout", request=request)
        with self.assertRaises(FetchError) as raised:
            self._fetch(timeout)
        self.assertEqual(raised.exception.code, "timeout")

    def test_refuses_unsafe_redirect_and_non_approved_origin(self) -> None:
        with self.assertRaises(FetchError) as raised:
            self._fetch(lambda _: httpx.Response(302, headers={
                "Location": "http://127.0.0.1/private"
            }))
        self.assertEqual(raised.exception.code, "unsafe_redirect")
        with self.assertRaises(FetchError) as raised:
            DocumentFetcher(settings=self.settings).fetch(FullTextLocation(
                "https://doi.org/10.5555/test", "DOI landing page", None, None
            ))
        self.assertEqual(raised.exception.code, "unsafe_url")

    def test_openalex_content_key_is_sent_only_on_request_not_saved_url(self) -> None:
        content_url = "https://content.openalex.org/works/W123.pdf"
        location = FullTextLocation(content_url, "OpenAlex OA content", "cc-by", None)
        settings = Settings(_env_file=None, openalex_api_key="test-secret")

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.params["api_key"], "test-secret")
            return httpx.Response(200, content=PDF, headers={"Content-Type": "application/pdf"})

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            document = DocumentFetcher(settings=settings, client=client).fetch(location)
        self.assertEqual(document.content, PDF)
        self.assertEqual(location.url, content_url)
        self.assertNotIn("test-secret", repr(location))


class FakeLocator:
    def __init__(self, mapping: dict[str, tuple[FullTextLocation, ...]]) -> None:
        self.mapping = mapping
        self.calls: list[str] = []

    def locate(self, paper: FullTextPaper) -> tuple[FullTextLocation, ...]:
        self.calls.append(paper.title)
        return self.mapping.get(paper.title, ())


class FakeFetcher:
    def __init__(self, failures: set[str] | None = None) -> None:
        self.failures = failures or set()
        self.calls: list[str] = []

    def fetch(self, location: FullTextLocation) -> FetchedPDF:
        self.calls.append(location.url)
        if location.url in self.failures:
            raise FetchError("not_found", "OA PDF URL returned HTTP 404.")
        return FetchedPDF(PDF, hashlib.sha256(PDF).hexdigest(), "application/pdf")


class AcquisitionServiceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.settings = Settings(_env_file=None, data_dir=Path(self.tempdir.name))
        self.engine = create_engine("sqlite+pysqlite:///:memory:",
                                    connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(self.engine)
        self.factory = create_session_factory(self.engine)
        self.run = create_research_run("RAG query rewriting?", session_factory=self.factory)
        plan = _plan()
        with session_scope(self.factory) as session:
            save_research_plan(session, run=session.get(ResearchRun, self.run.id), plan=plan,
                               model_name="test", prompt_version="test")
            query = create_search_query(session, run_id=self.run.id, query="RAG query rewriting",
                                        source="openalex")
            for index, (title, label) in enumerate((
                ("Available paper", "include"), ("Missing OA paper", "maybe"),
                ("Broken URL paper", "maybe"), ("Excluded paper", "exclude"),
            )):
                paper = upsert_paper(session, title=title, doi=f"10.5555/fulltext-{index}",
                                     year=2025, abstract="RAG study")
                add_search_result(session, search_query=query, paper=paper,
                                  source_record_id=f"W{index + 1}")
                snapshot = _input_snapshot(plan, _paper_snapshot(paper), "test-llm")
                session.add(ScreeningResult(
                    run_id=self.run.id, paper_id=paper.id, plan_id=session.get(ResearchRun, self.run.id).research_plan.id,
                    label=label, relevance_score=0.5, rationale="Screening test",
                    matched_inclusion_criteria=["Relevant RAG"], matched_exclusion_criteria=[],
                    needs_full_text=True, sub_question_index=0, evidence_role="method",
                    input_hash=_fingerprint(snapshot), input_snapshot=snapshot,
                    prompt_version="test", llm_model="test-llm",
                ))

    def tearDown(self) -> None:
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()
        self.tempdir.cleanup()

    def _location(self, url: str = URL) -> FullTextLocation:
        return FullTextLocation(url, "ACL Anthology", "cc-by", "publishedVersion")

    def test_statuses_retry_and_idempotent_cache(self) -> None:
        broken_url = "https://aclanthology.org/broken.pdf"
        locator = FakeLocator({
            "Available paper": (self._location(),),
            "Broken URL paper": (self._location(broken_url),),
        })
        fetcher = FakeFetcher({broken_url})
        first = acquire_fulltext(self.run.id, locator=locator, fetcher=fetcher,
                                 settings=self.settings, session_factory=self.factory)
        self.assertEqual((first.candidate_papers, first.downloaded, first.unavailable, first.failed),
                         (3, 1, 1, 1))
        self.assertEqual(first.full_text_available, 2)
        self.assertEqual(first.pending, 0)
        self.assertNotIn("Excluded paper", locator.calls)
        with self.factory() as session:
            rows = list(session.scalars(select(FullTextAcquisition)))
            self.assertEqual(len(rows), 3)
            by_title = {session.get(Paper, row.paper_id).title: row for row in rows}
            self.assertEqual(by_title["Missing OA paper"].status, "unavailable")
            self.assertIsNone(by_title["Missing OA paper"].failure_code)
            self.assertEqual(by_title["Broken URL paper"].failure_code, "not_found")
            self.assertEqual(by_title["Available paper"].paper_version.content_hash,
                             hashlib.sha256(PDF).hexdigest())
            self.assertEqual(session.scalar(select(func.count()).select_from(PaperVersion)), 1)
        self.assertEqual(len(fetcher.calls), 2)

        fetcher.failures.clear()
        second = acquire_fulltext(self.run.id, locator=locator, fetcher=fetcher,
                                  settings=self.settings, session_factory=self.factory)
        self.assertEqual(second.already_cached, 1)
        self.assertEqual(second.downloaded, 1)
        self.assertEqual(second.unavailable, 1)
        self.assertEqual(second.failed, 0)
        self.assertEqual(len(fetcher.calls), 3)
        self.assertEqual(locator.calls.count("Available paper"), 1)
        self.assertEqual(locator.calls.count("Missing OA paper"), 1)
        with self.factory() as session:
            versions = list(session.scalars(select(PaperVersion)))
            self.assertEqual(len(versions), 2)
            self.assertEqual(len({item.storage_path for item in versions}), 1)
            self.assertTrue((self.settings.data_dir / versions[0].storage_path).is_file())

        third = acquire_fulltext(self.run.id, locator=locator, fetcher=fetcher,
                                 settings=self.settings, session_factory=self.factory)
        self.assertEqual(third.downloaded, 0)
        self.assertEqual(third.already_cached, 2)
        self.assertEqual(len(fetcher.calls), 3)
        forced = acquire_fulltext(
            self.run.id, locator=locator, fetcher=fetcher, settings=self.settings,
            session_factory=self.factory, retry_unavailable=True,
        )
        self.assertEqual(forced.unavailable, 1)
        self.assertEqual(locator.calls.count("Missing OA paper"), 2)
        with self.factory() as session:
            unavailable = session.scalar(select(FullTextAcquisition).join(Paper).where(
                Paper.title == "Missing OA paper"
            ))
            self.assertEqual(unavailable.attempt_count, 2)

    def test_same_oa_url_is_reused_without_second_download(self) -> None:
        locator = FakeLocator({
            "Available paper": (self._location(),),
            "Broken URL paper": (self._location(),),
        })
        fetcher = FakeFetcher()
        summary = acquire_fulltext(self.run.id, locator=locator, fetcher=fetcher,
                                   settings=self.settings, session_factory=self.factory)
        self.assertEqual(summary.downloaded, 1)
        self.assertEqual(summary.already_cached, 1)
        self.assertEqual(fetcher.calls, [URL])
        with self.factory() as session:
            versions = list(session.scalars(select(PaperVersion)))
            self.assertEqual(len(versions), 2)
            self.assertEqual({item.content_hash for item in versions},
                             {hashlib.sha256(PDF).hexdigest()})

    def test_corrupt_cache_is_redownloaded_and_metadata_change_requires_rescreen(self) -> None:
        locator = FakeLocator({"Available paper": (self._location(),)})
        fetcher = FakeFetcher()
        acquire_fulltext(self.run.id, locator=locator, fetcher=fetcher,
                         settings=self.settings, session_factory=self.factory)
        with self.factory() as session:
            version = session.scalar(select(PaperVersion))
            path = self.settings.data_dir / version.storage_path
        path.write_bytes(b"corrupted local cache")
        retried = acquire_fulltext(self.run.id, locator=locator, fetcher=fetcher,
                                   settings=self.settings, session_factory=self.factory)
        self.assertEqual(retried.downloaded, 1)
        self.assertEqual(fetcher.calls, [URL, URL])
        self.assertEqual(path.read_bytes(), PDF)
        with session_scope(self.factory) as session:
            paper = session.scalar(select(Paper).where(Paper.title == "Available paper"))
            paper.abstract = "Changed since title/abstract screening"
        stale = acquire_fulltext(self.run.id, locator=locator, fetcher=fetcher,
                                 settings=self.settings, session_factory=self.factory)
        self.assertEqual(stale.stale_screening, 1)
        self.assertEqual(stale.candidate_papers, 2)
        self.assertEqual(fetcher.calls, [URL, URL])

    def test_unknown_run_and_unplanned_run_fail_before_location(self) -> None:
        locator = FakeLocator({})
        with self.assertRaises(LookupError):
            acquire_fulltext(uuid.uuid4(), locator=locator, settings=self.settings,
                             session_factory=self.factory)
        unplanned = create_research_run("unplanned", session_factory=self.factory)
        with self.assertRaisesRegex(ValueError, "frozen ResearchPlan"):
            acquire_fulltext(unplanned.id, locator=locator, settings=self.settings,
                             session_factory=self.factory)
        self.assertEqual(locator.calls, [])


if __name__ == "__main__":
    unittest.main()
