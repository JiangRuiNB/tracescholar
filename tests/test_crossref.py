"""Crossref adapter tests; all responses are mocked."""

from __future__ import annotations

import unittest

import httpx

from tracescholar.config import Settings
from tracescholar.sources import CrossrefSource, SourceError


def _item(doi: str = "10.5555/RAG.2025") -> dict:
    return {
        "DOI": doi,
        "title": ["<i>A Study</i> of RAG"],
        "author": [{"given": "Ada", "family": "Lovelace"}, {"name": "RAG Group"}],
        "container-title": ["Example Journal"],
        "published": {"date-parts": [[2025, 3, 7]]},
        "abstract": "<jats:p>Retrieval &amp; generation.</jats:p>",
        "URL": f"https://doi.org/{doi}",
    }


class CrossrefSourceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(
            _env_file=None,
            crossref_base_url="https://api.crossref.org",
            crossref_email="researcher@example.org",
            crossref_timeout_seconds=5,
        )

    def test_search_maps_metadata_and_identifies_polite_pool(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"message": {"total-results": 42, "items": [_item()]}})

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            batch = CrossrefSource(settings=self.settings, client=client).search(
                " retrieval augmented generation ", limit=20
            )

        self.assertEqual(batch.query, "retrieval augmented generation")
        self.assertEqual(batch.source, "crossref")
        self.assertEqual(batch.returned_count, 1)
        self.assertEqual(batch.total_count, 42)
        self.assertEqual(batch.filters, {"rows": 20})
        self.assertEqual(requests[0].url.params["query.bibliographic"], batch.query)
        self.assertEqual(requests[0].url.params["rows"], "20")
        self.assertEqual(requests[0].url.params["mailto"], "researcher@example.org")
        self.assertIn("mailto:researcher@example.org", requests[0].headers["User-Agent"])

        hit = batch.results[0]
        self.assertEqual(hit.source_record_id, "10.5555/rag.2025")
        self.assertEqual(hit.paper.title, "A Study of RAG")
        self.assertEqual(hit.paper.doi, "10.5555/RAG.2025")
        self.assertEqual(hit.paper.year, 2025)
        self.assertEqual(hit.paper.venue, "Example Journal")
        self.assertEqual(hit.paper.authors, ("Ada Lovelace", "RAG Group"))
        self.assertEqual(hit.paper.abstract, "Retrieval & generation.")

    def test_missing_optional_fields_are_empty_and_invalid_rows_are_skipped(self) -> None:
        rows = [{"DOI": "10.5555/minimal", "title": ["Minimal"]},
                {"DOI": "10.5555/notitle"}, {"title": ["No DOI"]}, 7]
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"message": {"items": rows}})
        )
        with httpx.Client(transport=transport) as client:
            batch = CrossrefSource(settings=self.settings, client=client).search("minimal")

        self.assertEqual(batch.returned_count, 4)
        self.assertEqual(batch.skipped_count, 3)
        self.assertEqual(batch.results[0].paper.authors, ())
        self.assertIsNone(batch.results[0].paper.year)
        self.assertIsNone(batch.results[0].paper.abstract)
        self.assertEqual(batch.results[0].source_url, "https://doi.org/10.5555/minimal")

    def test_empty_results_are_valid(self) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"message": {"total-results": 0, "items": []}})
        )
        with httpx.Client(transport=transport) as client:
            batch = CrossrefSource(settings=self.settings, client=client).search("no matches")
        self.assertEqual(batch.results, ())
        self.assertEqual(batch.returned_count, 0)

    def test_http_timeout_bad_status_json_and_envelope_are_reported(self) -> None:
        def timed_out(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        cases = (
            (timed_out, "timed out"),
            (lambda request: httpx.Response(503), "HTTP 503"),
            (lambda request: httpx.Response(200, text="bad JSON"), "invalid JSON"),
            (lambda request: httpx.Response(200, json={"message": {}}), "works envelope"),
        )
        for handler, expected in cases:
            with self.subTest(expected=expected):
                with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                    with self.assertRaisesRegex(SourceError, expected):
                        CrossrefSource(settings=self.settings, client=client, max_retries=0).search("query")

    def test_rate_limit_retry_uses_retry_after(self) -> None:
        calls = 0
        delays: list[float] = []

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(429, headers={"Retry-After": "2"})
            return httpx.Response(200, json={"message": {"items": []}})

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            CrossrefSource(settings=self.settings, client=client, sleep=delays.append).search("query")
        self.assertEqual(calls, 2)
        self.assertEqual(delays, [2.0])
