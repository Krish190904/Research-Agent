"""Local embedding generator using sentence-transformers.

Supports batching and CPU/GPU device selection. Returns L2-normalized embeddings for cosine similarity.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

logger = logging.getLogger("agent.embed")


class Embedder:
    """Embedder with a robust fallback: prefer sentence-transformers; fallback to TF-IDF when heavy deps missing.

    The encode method returns a 2D numpy array of normalized vectors.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int = 32,
    ):
        import yaml

        try:
            cfg = yaml.safe_load(open("config.yml"))
        except Exception:
            cfg = {}
        self.model_name = model_name or cfg.get("embeddings", {}).get(
            "model", "sentence-transformers/all-mpnet-base-v2"
        )
        self.batch_size = batch_size or cfg.get("embeddings", {}).get("batch_size", 32)
        self.device = device or cfg.get("embeddings", {}).get("device", "auto")
        # Try to load SentenceTransformer; if torch or C libs are missing, fall back
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            self._backend = "sbert"
            self.device = (
                "cuda"
                if (self.device in (None, "auto") and torch.cuda.is_available())
                else "cpu"
            )
            logger.info(
                "Loading SentenceTransformer %s on %s", self.model_name, self.device
            )
            self.model = SentenceTransformer(self.model_name, device=self.device)
        except Exception:
            logger.warning(
                "SentenceTransformer unavailable; falling back to TF-IDF embeddings"
            )
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._backend = "tfidf"
            self._tfidf = TfidfVectorizer(max_features=768)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, 0), dtype="float32")
        if getattr(self, "_backend", "tfidf") == "sbert":
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            # normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms
            return embeddings.astype("float32")
        else:
            X = self._tfidf.fit_transform(texts).toarray()
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            X = X / norms
            return X.astype("float32")
