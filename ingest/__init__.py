"""Ingest package: document loaders, chunking, and PII detection."""

from .loader import Ingestor
from .chunker import chunk_text

__all__ = ["Ingestor", "chunk_text"]
