"""Chunking utilities for documents."""

from __future__ import annotations

from typing import Iterator

from pathlib import Path
import yaml

try:
    cfg = yaml.safe_load(Path("config.yml").read_text())
    DEFAULT_SIZE = int(cfg.get("chunking", {}).get("size", 512))
    DEFAULT_OVERLAP = int(cfg.get("chunking", {}).get("overlap", 128))
except Exception:
    DEFAULT_SIZE = 512
    DEFAULT_OVERLAP = 128


def chunk_text(
    text: str, chunk_size: int = DEFAULT_SIZE, overlap: int = DEFAULT_OVERLAP
) -> Iterator[str]:
    """Yield overlapping chunks of `text`.

    Args:
        text: input string
        chunk_size: desired chunk size in characters
        overlap: overlap between chunks
    """
    if not text:
        return
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        yield text[start:end]
        if end == L:
            break
        start = max(0, end - overlap)
