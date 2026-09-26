"""Page locators, malformed PDFs, durable parse state and retry isolation."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
import uuid
from pathlib import Path

from sqlalchemy import create_engine, func, select
from sqlalchemy.pool import StaticPool

from tracescholar.config import Settings
from tracescholar.database import Base, create_session_factory, session_scope
from tracescholar.models import Chunk, FullTextAcquisition, Paper, PaperVersion, ParsedPage, PdfParseRecord, ResearchRun
from tracescholar.pdf_parsing import PDFParser, ParseError, get_chunk_provenance, parse_research_run
from tracescholar.pdf_parsing.parser import ChunkData, ParsedDocument, ParsedPageData


class FakePage:
    def __init__(self, lines, height=700):
        self.width = 500
        self.height = height
        self.lines = lines
        self.chars = [
            {"x0": line[1], "x1": line[1] + 3, "top": line[2], "size": line[4]}
            for line in lines for _ in line[0]
        ]

    def dedupe_chars(self, **_):
        return self

    def crop(self, box):
        x0, top, x1, bottom = box
        return FakePage([line for line in self.lines if x0 <= line[1] < x1
                         and top <= line[2] < bottom], height=self.height)

    def extract_text_lines(self, return_chars=False):
        return [{"text": value, "x0": x, "x1": x + len(value) * 4,
                 "top": y, "bottom": y + size, "chars": [{"size": size}] * len(value)}
                for value, x, y, _, size in self.lines]


class FakePDF:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class PDFParserTestCase(unittest.TestCase):
    def test_two_column_reading_order_is_left_then_right(self):
        left_one = "Left-column opening explains the method with enough text to detect a column."
        left_two = "Left-column continuation belongs before the right-side content in reading order."
        right_one = "Right-column opening begins only after all left-side paragraphs are read."
        right_two = "Right-column continuation concludes the clearly separated body discussion."
        page = FakePage([
            (right_one, 290, 250, 1, 10),
            (left_one, 40, 250, 0, 10),
            (right_two, 290, 280, 1, 10),
            (left_two, 40, 280, 0, 10),
        ])
        result = PDFParser(opener=lambda _: FakePDF([page])).parse(Path("columns.pdf"))
        self.assertEqual(result.pages[0].column_count, 2)
        self.assertEqual(result.pages[0].text.splitlines(),
                         [left_one, left_two, right_one, right_two])

    def test_multipage_blank_page_sections_and_reference_omission(self):
        pages = [
            FakePage([
                ("1", 460, 20, 0, 8),
                ("Introduction", 50, 100, 0, 14),
                ("A first paragraph with enough useful scientific text to read.", 50, 130, 0, 10),
                ("It continues on the next visual line without losing its locator.", 50, 143, 0, 10),
                ("A second paragraph describes the experimental method.", 50, 180, 0, 10),
            ]),
            FakePage([]),
            FakePage([
                ("2. Results", 50, 100, 0, 14),
                ("The results report measured performance on held-out tasks.", 50, 130, 0, 10),
                ("References", 50, 170, 0, 14),
                ("[1] This citation must not pollute the正文 text.", 50, 190, 0, 8),
            ]),
        ]
        document = PDFParser(opener=lambda _: FakePDF(pages)).parse(Path("unused.pdf"))
        self.assertEqual(len(document.pages), 3)
        self.assertEqual(document.pages[1].text, "")
        self.assertTrue(any(chunk.section == "2. Results" for chunk in document.chunks))
        self.assertFalse(any("citation" in chunk.text for chunk in document.chunks))
        self.assertEqual(document.chunks[0].page_start, 1)
        for chunk in document.chunks:
            page = document.pages[chunk.page_start - 1]
            self.assertEqual(page.text[chunk.locator["char_start"]:chunk.locator["char_end"]], chunk.text)
            self.assertEqual(len(chunk.locator["bbox"]), 4)
            self.assertLess(chunk.locator["bbox"][1], chunk.locator["bbox"][3])

    def test_empty_and_invalid_pdf_are_explicit_failures(self):
        with self.assertRaises(ParseError) as raised:
            PDFParser(opener=lambda _: FakePDF([FakePage([])])).parse(Path("blank.pdf"))
        self.assertEqual(raised.exception.code, "no_text")

        def invalid(_):
            raise RuntimeError("broken xref")

        with self.assertRaises(ParseError) as raised:
            PDFParser(opener=invalid).parse(Path("bad.pdf"))
        self.assertEqual(raised.exception.code, "invalid_pdf")

    def test_repeated_margin_header_does_not_enter_chunks(self):
        pages = [FakePage([
            ("Journal of Test Research", 40, 15, 0, 8),
            ("Body content with sufficient detail for a valid page of research text.", 40, 110, 0, 10),
        ]) for _ in range(4)]
        result = PDFParser(opener=lambda _: FakePDF(pages)).parse(Path("four.pdf"))
        self.assertTrue(all("Journal of Test Research" not in page.text for page in result.pages))


class StubParser:
    version = "stub-v1"

    def __init__(self, failing: set[str] | None = None):
        self.failing = failing or set()
        self.calls: list[str] = []

    def parse(self, path: Path) -> ParsedDocument:
        self.calls.append(path.name)
        if path.name in self.failing:
            raise ParseError("invalid_pdf", "Simulated damaged PDF")
        text = "Introduction\nA verifiable result from the scientific article."
        chunk = text.split("\n", 1)[1]
        return ParsedDocument(
            (ParsedPageData(1, text, 500, 700, 1),),
            (ChunkData(0, chunk, 1, 1, "Introduction", 13, len(text),
                       {"page": 1, "char_start": 13, "char_end": len(text),
                        "bbox": [40, 100, 400, 120], "column": 0,
                        "coordinate_system": "pdf_points_top_left"}, 9),),
            (),
        )


class ParseServiceTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings = Settings(_env_file=None, data_dir=Path(self.temp.name))
        self.engine = create_engine("sqlite+pysqlite:///:memory:",
                                    connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(self.engine)
        self.factory = create_session_factory(self.engine)
        self.run_id = uuid.uuid4()
        self.ids = []
        with session_scope(self.factory) as session:
            session.add(ResearchRun(id=self.run_id, question="PDF parsing?"))
            for n in range(2):
                data = f"%PDF-1.4 test {n}".encode()
                name = f"sample-{n}.pdf"
                (Path(self.temp.name) / name).write_bytes(data)
                paper = Paper(title=f"Paper {n}", normalized_title=f"paper {n}", authors=[])
                session.add(paper)
                session.flush()
                version = PaperVersion(paper_id=paper.id, content_hash=hashlib.sha256(data).hexdigest(),
                                       storage_path=name, content_bytes=len(data), source_url="https://example.org/x.pdf",
                                       source_name="test")
                session.add(version)
                session.flush()
                self.ids.append(version.id)
                session.add(FullTextAcquisition(run_id=self.run_id, paper_id=paper.id,
                                                paper_version_id=version.id, status="downloaded",
                                                attempt_count=1))

    def tearDown(self):
        self.engine.dispose()
        self.temp.cleanup()

    def test_persist_rehydrate_and_skip_unchanged(self):
        parser = StubParser()
        first = parse_research_run(self.run_id, parser=parser, settings=self.settings,
                                   session_factory=self.factory)
        self.assertEqual((first.paper_versions, first.parsed_successfully, first.failed,
                          first.total_pages, first.total_chunks), (2, 2, 0, 2, 2))
        self.assertEqual(len(parser.calls), 2)
        second = parse_research_run(self.run_id, parser=parser, settings=self.settings,
                                    session_factory=self.factory)
        self.assertEqual(second.skipped_existing, 2)
        self.assertEqual(len(parser.calls), 2)
        with self.factory() as session:
            chunks = list(session.scalars(select(Chunk).order_by(Chunk.paper_version_id)))
            self.assertEqual(len(chunks), 2)
            for chunk in chunks:
                page = session.scalar(select(ParsedPage).where(
                    ParsedPage.paper_version_id == chunk.paper_version_id))
                self.assertEqual(page.text[chunk.locator["char_start"]:chunk.locator["char_end"]], chunk.text)
                self.assertEqual(chunk.paper_version.paper.title[:5], "Paper")
            sample_id = chunks[0].id
        provenance = get_chunk_provenance(sample_id, session_factory=self.factory)
        self.assertEqual(provenance.paper_title[:5], "Paper")
        self.assertEqual(provenance.page_start, 1)
        self.assertEqual(provenance.text, chunks[0].text)

    def test_one_failure_is_recorded_and_retryable(self):
        parser = StubParser({"sample-0.pdf"})
        first = parse_research_run(self.run_id, parser=parser, settings=self.settings,
                                   session_factory=self.factory)
        self.assertEqual((first.parsed_successfully, first.failed), (1, 1))
        with self.factory() as session:
            failure = session.scalar(select(PdfParseRecord).where(PdfParseRecord.status == "failed"))
            self.assertEqual(failure.failure_code, "invalid_pdf")
            self.assertEqual(session.scalar(select(func.count()).select_from(Chunk)), 1)
        parser.failing.clear()
        second = parse_research_run(self.run_id, parser=parser, settings=self.settings,
                                    session_factory=self.factory)
        self.assertEqual((second.parsed_successfully, second.failed, second.newly_parsed,
                          second.skipped_existing), (2, 0, 1, 1))

    def test_hash_mismatch_is_recorded_without_parser_call(self):
        (Path(self.temp.name) / "sample-0.pdf").write_bytes(b"tampered content")
        parser = StubParser()
        summary = parse_research_run(self.run_id, parser=parser, settings=self.settings,
                                     session_factory=self.factory)
        self.assertEqual(summary.failed, 1)
        self.assertEqual(summary.failures[0].code, "size_mismatch")
        self.assertEqual(len(parser.calls), 1)

    def test_unexpected_parser_crash_is_isolated(self):
        class CrashParser(StubParser):
            def parse(self, path):
                if path.name == "sample-0.pdf":
                    raise RuntimeError("library crash")
                return super().parse(path)

        summary = parse_research_run(self.run_id, parser=CrashParser(), settings=self.settings,
                                     session_factory=self.factory)
        self.assertEqual((summary.parsed_successfully, summary.failed), (1, 1))
        self.assertEqual(summary.failures[0].code, "unexpected_parse_error")
