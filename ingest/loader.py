"""Document loaders for PDF, Markdown, TXT, and HTML.

Implements metadata extraction and PII detection warnings.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Iterator

import frontmatter
import fitz  # PyMuPDF
import pdfplumber
from bs4 import BeautifulSoup

from .chunker import chunk_text

logger = logging.getLogger("agent.ingest")


PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "phone": re.compile(r"\b\+?\d[\d\-() ]{7,}\b"),
}


@dataclass
class DocumentChunk:
    id: str
    text: str
    metadata: Dict


class Ingestor:
    """High-level ingestor that detects file type and yields chunks.

    Methods
    -------
    ingest_folder(folder: Path, recursive: bool) -> List[DocumentChunk]
    """

    def __init__(self):
        self.supported = {"pdf", "md", "txt", "html", "htm"}

    def ingest_folder(
        self, folder: Path, recursive: bool = False
    ) -> List[DocumentChunk]:
        folder = Path(folder)
        chunks: List[DocumentChunk] = []
        for p in folder.glob("**/*" if recursive else "*"):
            if p.is_file() and p.suffix.lower().lstrip(".") in self.supported:
                chunks.extend(list(self._ingest_file(p)))
        logger.info("Ingested %d chunks from %s", len(chunks), folder)
        return chunks

    def _ingest_file(self, path: Path) -> Iterator[DocumentChunk]:
        ext = path.suffix.lower()
        meta = {"source": str(path), "name": path.name}
        text = ""
        if ext == ".pdf":
            text, meta_tables = self._load_pdf(path)
            meta["tables"] = meta_tables
        elif ext in (".md", ".markdown"):
            text, md_meta = self._load_markdown(path)
            meta.update(md_meta)
        elif ext == ".txt":
            text = path.read_text(encoding="utf8")
        elif ext in (".html", ".htm"):
            text = self._load_html(path)
        else:
            logger.warning("Unsupported file type: %s", path)
            return

        pii_warnings = self._detect_pii(text)
        if pii_warnings:
            meta["pii_warnings"] = pii_warnings

        for i, chunk in enumerate(chunk_text(text)):
            cid = f"{path.name}::chunk::{i}"
            yield DocumentChunk(id=cid, text=chunk, metadata={**meta, "chunk_index": i})

    def _load_pdf(self, path: Path) -> tuple[str, List[Dict]]:
        text = []
        tables: List[Dict] = []
        try:
            with fitz.open(path) as doc:
                for page in doc:
                    page_text = page.get_text("text")
                    text.append(page_text)
        except Exception:
            logger.exception("PyMuPDF failed, trying pdfplumber for %s", path)
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    try:
                        text.append(page.extract_text() or "")
                        # extract simple tables
                        for table in page.extract_tables():
                            tables.append({"page": page.page_number, "table": table})
                    except Exception:
                        logger.exception("pdfplumber page failed on %s", path)
        return "\n".join(text), tables

    def _load_markdown(self, path: Path) -> tuple[str, Dict]:
        raw = path.read_text(encoding="utf8")
        post = frontmatter.loads(raw)
        return post.content, post.metadata or {}

    def _load_html(self, path: Path) -> str:
        raw = path.read_text(encoding="utf8")
        soup = BeautifulSoup(raw, "html.parser")
        # remove scripts/styles
        for s in soup(["script", "style"]):
            s.decompose()
        return soup.get_text(separator="\n")

    def _detect_pii(self, text: str) -> Dict[str, int]:
        warnings: Dict[str, int] = {}
        for name, pat in PII_PATTERNS.items():
            found = pat.findall(text)
            if found:
                warnings[name] = len(found)
        return warnings
