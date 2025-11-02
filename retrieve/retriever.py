"""Retriever implementing dense search and optional MMR reranking."""

from __future__ import annotations

import logging
from typing import List, Dict

import numpy as np

from index.store import Indexer

logger = logging.getLogger("agent.retrieve")


def mmr(
    doc_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    lambda_param: float = 0.7,
    k: int = 5,
) -> List[int]:
    """Maximal Marginal Relevance selection.

    Args:
        doc_embeddings: (n_docs, dim) normalized
        query_embedding: (dim,) normalized
        lambda_param: tradeoff between relevance and diversity
        k: number to select

    Returns indices (into doc_embeddings) of selected documents.
    """
    n = doc_embeddings.shape[0]
    if n == 0:
        return []
    # cosine similarities since embeddings normalized -> dot product
    sims = doc_embeddings @ query_embedding
    selected: List[int] = []
    candidates = set(range(n))
    while len(selected) < min(k, n):
        best_score = None
        best_idx = None
        for idx in list(candidates):
            rel = sims[idx]
            if not selected:
                score = lambda_param * rel
            else:
                # similarity of candidate to any selected (max)
                sim_to_selected = max(
                    float(doc_embeddings[idx] @ doc_embeddings[s]) for s in selected
                )
                score = lambda_param * rel - (1 - lambda_param) * sim_to_selected
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        candidates.remove(best_idx)
    return selected


class Retriever:
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def retrieve(
        self,
        query_emb: np.ndarray,
        top_k: int = 5,
        mmr_enabled: bool = True,
        lambda_param: float = 0.7,
        candidate_multiplier: int = 5,
    ) -> List[Dict]:
        # get a superset of candidates
        hits = self.indexer.search(query_emb, top_k=top_k * candidate_multiplier)
        if not hits:
            return []
        faiss_ids = [hid for hid, _ in hits]
        metas = self.indexer.fetch_metadata(faiss_ids)

        # fetch stored embeddings for these ids
        doc_embs = self.indexer.fetch_embeddings(faiss_ids)
        # filter out missing embeddings
        valid_idx = [i for i, e in enumerate(doc_embs) if e is not None]
        if not valid_idx:
            # fallback: return top-k by score
            out = []
            for (hid, score), m in zip(hits, metas):
                out.append(
                    {
                        "faiss_id": hid,
                        "score": score,
                        "text": m.get("text", ""),
                        "meta": m.get("meta", {}),
                    }
                )
            return out[:top_k]

        emb_matrix = np.vstack([doc_embs[i] for i in valid_idx])
        # ensure embeddings normalized
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_matrix = emb_matrix / norms

        qvec = np.array(query_emb, dtype=np.float32)
        if qvec.ndim == 2 and qvec.shape[0] == 1:
            qvec = qvec.ravel()
        qnorm = np.linalg.norm(qvec)
        if qnorm == 0:
            qnorm = 1.0
        qvec = qvec / qnorm

        if mmr_enabled:
            sel = mmr(
                emb_matrix,
                qvec,
                lambda_param=lambda_param,
                k=min(top_k, emb_matrix.shape[0]),
            )
            selected_global_idxs = [valid_idx[s] for s in sel]
        else:
            # fallback: pick highest score order
            selected_global_idxs = list(range(min(top_k, len(faiss_ids))))

        out = []
        for idx in selected_global_idxs:
            hid, score = hits[idx]
            m = metas[idx]
            out.append(
                {
                    "faiss_id": hid,
                    "score": score,
                    "text": m.get("text", ""),
                    "meta": m.get("meta", {}),
                }
            )
        return out
