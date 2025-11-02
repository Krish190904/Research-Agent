"""FAISS-backed index with SQLite metadata storage."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import faiss
from sqlalchemy import Column, Integer, String, Text, LargeBinary, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


logger = logging.getLogger("agent.index")
Base = declarative_base()


class DocumentMeta(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    faiss_id = Column(Integer, unique=True, index=True)
    doc_id = Column(String, index=True)
    chunk_index = Column(Integer)
    text = Column(Text)
    meta = Column(Text)  # json


class Embedding(Base):
    __tablename__ = "embeddings"
    id = Column(Integer, primary_key=True)
    faiss_id = Column(Integer, unique=True, index=True)
    vector = Column(LargeBinary)  # store raw float32 bytes


@dataclass
class IndexStats:
    ntotal: int


class Indexer:
    def __init__(
        self,
        faiss_path: str | None = None,
        sqlite_path: str | None = None,
        index_type: str | None = None,
    ):
        import yaml

        try:
            cfg = yaml.safe_load(open("config.yml"))
        except Exception:
            cfg = {}
        self.faiss_path = faiss_path or cfg.get("index", {}).get(
            "faiss_path", "data/faiss.index"
        )
        self.sqlite_path = sqlite_path or cfg.get("index", {}).get(
            "sqlite_path", "data/meta.db"
        )
        self.index_type = index_type or cfg.get("index", {}).get("type", "FlatIP")
        self.dim: Optional[int] = None
        self.index: Optional[faiss.Index] = None

        os.makedirs(os.path.dirname(self.faiss_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.sqlite_path) or ".", exist_ok=True)

        self.engine = create_engine(f"sqlite:///{self.sqlite_path}")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def _make_index(self) -> faiss.Index:
        if self.dim is None:
            raise ValueError("Index dimension not set")
        if self.index_type == "FlatL2":
            idx = faiss.IndexFlatL2(self.dim)
        elif self.index_type == "HNSW":
            idx = faiss.IndexHNSWFlat(self.dim, 32)
        else:
            # default to inner product for cosine if vectors normalized
            idx = faiss.IndexFlatIP(self.dim)
        return idx

    def rebuild(self):
        # For now, rebuild is a placeholder: it creates an empty index
        if self.dim is None:
            # create a placeholder dimension; will be reset on add
            self.dim = 768
        self.index = self._make_index()
        faiss.write_index(self.index, self.faiss_path)
        logger.info("Rebuilt empty index at %s", self.faiss_path)

    def load(self):
        try:
            if Path(self.faiss_path).exists():
                self.index = faiss.read_index(self.faiss_path)
                logger.info("Loaded FAISS index from %s", self.faiss_path)
            else:
                if self.dim is None:
                    # lazy default dimension
                    self.dim = 768
                self.index = self._make_index()
                logger.info("Created new FAISS index (not persisted yet)")
        except Exception:
            logger.exception("Failed to load FAISS index; creating new one")
            self.index = self._make_index()

    def add(self, embeddings: np.ndarray, docs: List[Dict]):
        """Add embeddings and documents (docs must contain keys text, doc_id, chunk_index, meta)."""
        # ensure we have correct index for embedding dimension
        emb_dim = int(embeddings.shape[1])
        if self.dim is None or self.dim != emb_dim:
            self.dim = emb_dim
            # recreate index with new dim
            self.index = self._make_index()
        if self.index is None:
            self.load()
        n_before = int(self.index.ntotal)
        # ensure embeddings are float32
        vecs = embeddings.astype("float32")
        self.index.add(vecs)
        # update metadata mapping in SQLite
        session = self.Session()
        try:
            for i, doc in enumerate(docs):
                faiss_id = n_before + i
                meta_json = json.dumps(doc.get("metadata", {}))
                rec = DocumentMeta(
                    faiss_id=faiss_id,
                    doc_id=doc.get("id", ""),
                    chunk_index=doc.get("metadata", {}).get("chunk_index", 0),
                    text=doc.get("text", ""),
                    meta=meta_json,
                )
                session.add(rec)
                # store embedding as raw float32 bytes
                vec = vecs[i]
                bin_blob = vec.tobytes()
                eb = Embedding(faiss_id=faiss_id, vector=bin_blob)
                session.add(eb)
            session.commit()
        finally:
            session.close()
        faiss.write_index(self.index, self.faiss_path)
        logger.info(
            "Added %d vectors (total=%d)", embeddings.shape[0], int(self.index.ntotal)
        )

    def search(self, query_emb: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        if self.index is None:
            self.load()
        if query_emb.ndim == 1:
            query_emb = query_emb[None, :]
        query_emb = query_emb.astype("float32")
        D, idxs = self.index.search(query_emb, top_k)
        # return list of (faiss_id, score)
        return [(int(i), float(d)) for i, d in zip(idxs[0], D[0]) if i != -1]

    def fetch_metadata(self, faiss_ids: List[int]) -> List[Dict]:
        session = self.Session()
        try:
            rows = (
                session.query(DocumentMeta)
                .filter(DocumentMeta.faiss_id.in_(faiss_ids))
                .all()
            )
            out = []
            for r in rows:
                try:
                    meta = json.loads(r.meta or "{}")
                except Exception:
                    meta = {}
                out.append(
                    {
                        "faiss_id": r.faiss_id,
                        "doc_id": r.doc_id,
                        "chunk_index": r.chunk_index,
                        "text": r.text,
                        "meta": meta,
                    }
                )
            return out
        finally:
            session.close()

    def fetch_embeddings(self, faiss_ids: List[int]) -> List[np.ndarray]:
        """Return list of numpy float32 vectors corresponding to faiss_ids (order preserved when possible)."""
        session = self.Session()
        try:
            rows = (
                session.query(Embedding).filter(Embedding.faiss_id.in_(faiss_ids)).all()
            )
            id_to_vec = {}
            for r in rows:
                try:
                    b = r.vector
                    if b is None:
                        continue
                    arr = np.frombuffer(b, dtype=np.float32)
                    id_to_vec[r.faiss_id] = arr
                except Exception:
                    continue
            return [id_to_vec.get(fid) for fid in faiss_ids]
        finally:
            session.close()

    def stats(self) -> IndexStats:
        ntotal = int(self.index.ntotal) if self.index is not None else 0
        return IndexStats(ntotal=ntotal)
