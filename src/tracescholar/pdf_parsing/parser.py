"""Extract reading-order PDF text into verifiable page-local chunks.

The locator's character range always indexes ParsedPage.text, and its bbox is
expressed in PDF points from the top-left of the physical page.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pdfplumber


PARSER_VERSION = "pdfplumber-layout-v1"
_SPACE = re.compile(r"\s+")
_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*|[IVXLC]+)\.?\s+[A-Z][^.!?]{2,100}$", re.I)
_REFERENCES = re.compile(r"^(?:(?:\d+(?:\.\d+)*|[IVXLC]+)\.?\s+)?(?:references|bibliography|works cited)\s*$", re.I)
_PAGE_NUMBER = re.compile(r"^(?:\d{1,3}|page\s+\d{1,3}(?:\s+of\s+\d{1,3})?)$", re.I)
_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class ParseError(Exception):
    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(detail)


@dataclass(frozen=True, slots=True)
class ParsedPageData:
    page_number: int
    text: str
    width: float
    height: float
    column_count: int
    quality_flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ChunkData:
    ordinal: int
    text: str
    page_start: int
    page_end: int
    section: str
    document_char_start: int
    document_char_end: int
    locator: dict[str, Any]
    token_count: int


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    pages: tuple[ParsedPageData, ...]
    chunks: tuple[ChunkData, ...]
    quality_flags: tuple[str, ...]


@dataclass(slots=True)
class _Line:
    text: str
    x0: float
    top: float
    x1: float
    bottom: float
    size: float
    column: int
    start: int = 0
    end: int = 0
    section: str = "Body"


@dataclass(slots=True)
class _RawPage:
    number: int
    width: float
    height: float
    columns: int
    lines: list[_Line] = field(default_factory=list)


def _normalized(value: str) -> str:
    return _SPACE.sub(" ", value).strip()


def _line(raw: dict[str, Any], column: int) -> _Line | None:
    value = _normalized(raw.get("text", ""))
    if not value:
        return None
    chars = raw.get("chars") or []
    sizes = [float(c.get("size", 0)) for c in chars if c.get("size")]
    size = Counter(round(v * 2) / 2 for v in sizes).most_common(1)[0][0] if sizes else 10.0
    return _Line(value, float(raw["x0"]), float(raw["top"]),
                 float(raw["x1"]), float(raw["bottom"]), size, column)


def _body_size(chars: list[dict[str, Any]]) -> float:
    sizes = [round(float(c.get("size", 0)) * 2) / 2 for c in chars
             if 7 <= float(c.get("size", 0)) <= 13]
    return Counter(sizes).most_common(1)[0][0] if sizes else 10.0


def _two_columns(page: Any) -> bool:
    chars = [c for c in page.chars if page.height * .25 < c["top"] < page.height * .88
             and 7.0 <= float(c.get("size", 0)) <= 12.5]
    if len(chars) < 180:
        return False
    mid = page.width / 2
    left = sum(c["x0"] < mid - 15 for c in chars)
    right = sum(c["x1"] > mid + 15 for c in chars)
    center = sum(mid - 12 <= (c["x0"] + c["x1"]) / 2 <= mid + 12 for c in chars)
    return left > 60 and right > 60 and center / len(chars) < .034


def _first_page_split(page: Any) -> float:
    """Locate first sustained pair of independent body lines below the title."""
    mid = page.width / 2
    left = page.crop((0, page.height * .22, mid, page.height * .60))
    right = page.crop((mid, page.height * .22, page.width, page.height * .60))
    left_lines = left.extract_text_lines(return_chars=True)
    right_lines = right.extract_text_lines(return_chars=True)
    for line in left_lines:
        if _normalized(line["text"]).lower().startswith("abstract"):
            return max(page.height * .22, line["top"] - 3)
    for a in left_lines:
        if len(_normalized(a["text"])) < 22:
            continue
        for b in right_lines:
            if len(_normalized(b["text"])) >= 22 and abs(a["top"] - b["top"]) < 5:
                return max(page.height * .22, min(a["top"], b["top"]) - 3)
    return page.height * .22


def _extract_page(page: Any, number: int) -> _RawPage:
    deduped = page.dedupe_chars(tolerance=1)
    columns = 2 if _two_columns(deduped) else 1
    result = _RawPage(number, float(page.width), float(page.height), columns)
    if columns == 1:
        regions = [(deduped, 0)]
    else:
        mid = page.width / 2
        split = _first_page_split(deduped) if number == 1 else 0.0
        regions = []
        regions.extend([
            (deduped.crop((0, split, mid, page.height)), 0),
            (deduped.crop((mid, split, page.width, page.height)), 1),
        ])
    for region, column in regions:
        for raw in region.extract_text_lines(return_chars=True):
            line = _line(raw, column)
            if line is not None:
                result.lines.append(line)
    return result


def _is_heading(line: _Line, body_size: float) -> bool:
    value = line.text
    if len(value) > 105 or len(value) < 3:
        return False
    if value.lower() in {"abstract", "introduction", "conclusion", "conclusions", "discussion"}:
        return True
    if _HEADING.match(value):
        return True
    return line.size >= body_size * 1.19 and len(value.split()) <= 12 and not value.endswith(".")


def _bbox(lines: list[_Line]) -> list[float]:
    return [round(min(line.x0 for line in lines), 2), round(min(line.top for line in lines), 2),
            round(max(line.x1 for line in lines), 2), round(max(line.bottom for line in lines), 2)]


def _chunks_for_page(lines: list[_Line], page_number: int, doc_offset: int,
                     ordinal: int, max_chars: int) -> list[ChunkData]:
    """Pack adjacent paragraph lines of one section/column, splitting only if needed."""
    chunks: list[ChunkData] = []
    buffer: list[_Line] = []

    def flush() -> None:
        if not buffer:
            return
        start, end = buffer[0].start, buffer[-1].end
        # The caller's page text joins all retained lines with one newline.
        text = "\n".join(line.text for line in buffer)
        chunks.append(ChunkData(
            ordinal + len(chunks), text, page_number, page_number, buffer[0].section,
            doc_offset + start, doc_offset + end,
            {"page": page_number, "char_start": start, "char_end": end,
             "bbox": _bbox(buffer), "column": buffer[0].column,
             "coordinate_system": "pdf_points_top_left"}, len(_TOKEN.findall(text)),
        ))
        buffer.clear()

    previous: _Line | None = None
    for line in lines:
        if _is_heading(line, 10.0):
            flush()
            previous = line
            continue
        gap = line.top - previous.bottom if previous is not None else 0
        break_before = (previous is not None and (line.column != previous.column
                        or line.section != previous.section or gap > max(line.size * .85, 7.0)
                        or (line.x0 - previous.x0 > 10 and not previous.text.endswith("-"))))
        if buffer and (break_before or line.end - buffer[0].start > max_chars):
            flush()
        buffer.append(line)
        previous = line
    flush()
    return chunks


class PDFParser:
    """Parse a PDF with layout heuristics and reject unusable extraction."""

    version = PARSER_VERSION

    def __init__(self, *, opener: Callable[..., Any] = pdfplumber.open,
                 max_chunk_chars: int = 1800):
        self.opener = opener
        self.max_chunk_chars = max_chunk_chars

    def parse(self, path: Path) -> ParsedDocument:
        try:
            with self.opener(path) as pdf:
                raw_pages = [_extract_page(page, index) for index, page in enumerate(pdf.pages, 1)]
        except ParseError:
            raise
        except Exception as error:
            raise ParseError("invalid_pdf", f"PDF decoding failed: {error}") from error
        if not raw_pages:
            raise ParseError("empty_pdf", "PDF contains no pages.")

        all_margins = Counter()
        for raw in raw_pages:
            for line in raw.lines:
                if line.top < raw.height * .065 or line.bottom > raw.height * .95:
                    all_margins[_normalized(line.text).lower()] += 1
        repeated = {text for text, count in all_margins.items() if count >= 3 and count >= len(raw_pages) * .25}

        pages: list[ParsedPageData] = []
        chunks: list[ChunkData] = []
        flags: set[str] = set()
        section = "Front matter"
        references = False
        doc_offset = 0
        for raw in raw_pages:
            body_size = _body_size([{"size": line.size} for line in raw.lines])
            kept: list[_Line] = []
            page_flags: list[str] = []
            for line in raw.lines:
                value = line.text
                if references:
                    continue
                if _REFERENCES.fullmatch(value):
                    references = True
                    page_flags.append("references_omitted")
                    continue
                in_margin = line.top < raw.height * .065 or line.bottom > raw.height * .95
                if in_margin and (_PAGE_NUMBER.fullmatch(value) or value.lower() in repeated):
                    continue
                if line.size < max(6.5, body_size * .89) and len(value) > 15:
                    continue
                if _is_heading(line, body_size):
                    section = value
                line.section = section
                kept.append(line)
            if references and "references_omitted" not in page_flags:
                page_flags.append("references_omitted")
            if not kept:
                page_flags.append("no_body_text")
            text_parts: list[str] = []
            cursor = 0
            for line in kept:
                if text_parts:
                    cursor += 1
                line.start = cursor
                text_parts.append(line.text)
                cursor += len(line.text)
                line.end = cursor
            page_text = "\n".join(text_parts)
            pages.append(ParsedPageData(raw.number, page_text, raw.width, raw.height,
                                        raw.columns, tuple(sorted(set(page_flags)))))
            chunks.extend(_chunks_for_page(kept, raw.number, doc_offset,
                                          len(chunks), self.max_chunk_chars))
            doc_offset += len(page_text) + 1

        text_pages = sum(bool(page.text) for page in pages)
        total_text = "\n".join(page.text for page in pages)
        if not chunks or len(total_text.strip()) < 100:
            raise ParseError("no_text", "No usable body text was extracted; PDF may be scanned or encrypted.")
        if text_pages / len(pages) < .35:
            flags.add("low_page_coverage")
        bad = sum(c in "\ufffd\x00\u25a1" or (ord(c) < 32 and c not in "\n\t") for c in total_text)
        if bad / max(len(total_text), 1) > .02:
            raise ParseError("garbled_text", "Extracted text contains too many replacement/control characters.")
        lines = [line for page in pages for line in page.text.splitlines() if len(line) > 30]
        if lines and 1 - len(set(lines)) / len(lines) > .30:
            flags.add("repeated_text")
        if text_pages / len(pages) < .20:
            raise ParseError("low_page_coverage", "Too few PDF pages contained extractable body text.")
        return ParsedDocument(tuple(pages), tuple(chunks), tuple(sorted(flags)))
