"""Page-aware PDF parsing and persistence."""

from tracescholar.pdf_parsing.parser import PDFParser, ParseError
from tracescholar.pdf_parsing.service import (
    ChunkProvenance, ParseSummary, get_chunk_provenance, parse_research_run,
)

__all__ = ["PDFParser", "ParseError", "ChunkProvenance", "ParseSummary",
           "get_chunk_provenance", "parse_research_run"]
