"""Pipeline to ingest documents, generate embeddings, and add them to FAISS + metadata store."""

from __future__ import annotations

import logging
from pathlib import Path

from embed.encoder import Embedder
from index.store import Indexer
from ingest.loader import Ingestor

logger = logging.getLogger("agent.pipeline")


class Pipeline:
    def __init__(
        self, embedder: Embedder | None = None, indexer: Indexer | None = None
    ):
        self.embedder = embedder or Embedder()
        self.indexer = indexer or Indexer()

    def run_folder(
        self, folder: str | Path, recursive: bool = False, batch_size: int | None = None
    ) -> int:
        folder = Path(folder)
        ing = Ingestor()
        chunks = list(ing.ingest_folder(folder, recursive=recursive))
        logger.info("Pipeline: %d chunks to embed", len(chunks))
        texts = [c.text for c in chunks]
        ids = [c.id for c in chunks]
        metas = [c.metadata for c in chunks]

        batch_size = batch_size or self.embedder.batch_size
        n = 0
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_emb = self.embedder.encode(batch_texts)
            docs = []
            for j, t in enumerate(batch_texts):
                docs.append({"id": ids[i + j], "text": t, "metadata": metas[i + j]})
            self.indexer.add(batch_emb, docs)
            n += batch_emb.shape[0]
            logger.info("Indexed batch %d -> total %d", i, n)
        return n
