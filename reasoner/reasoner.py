"""Multi-step reasoning pipeline: decomposition, retrieval orchestration, and synthesis."""

from __future__ import annotations

import logging
from typing import List, Dict

from retrieve.retriever import Retriever
from index.store import Indexer
from embed.encoder import Embedder
from .llm_adapter import LLMAdapter

logger = logging.getLogger("agent.reasoner")


class Reasoner:
    def __init__(self, indexer: Indexer | None = None):
        self.indexer = indexer or Indexer()
        self.embed = Embedder()
        self.retriever = Retriever(self.indexer)
        self.llm = LLMAdapter()

    def decompose(self, query: str) -> List[str]:
        # Simple decomposition: split by sentences
        parts = [s.strip() for s in query.replace("?", ".").split(".") if s.strip()]
        if not parts:
            return [query]
        return parts

    def answer(
        self,
        query: str,
        top_k: int = 5,
        mmr_enabled: bool = True,
        mmr_lambda: float = 0.7,
        candidate_multiplier: int = 5,
    ) -> Dict:
        parts = self.decompose(query)
        traces = []
        collected_evidence = []
        for part in parts:
            emb = self.embed.encode([part])[0]
            hits = self.retriever.retrieve(
                emb,
                top_k=top_k,
                mmr_enabled=mmr_enabled,
                lambda_param=mmr_lambda,
                candidate_multiplier=candidate_multiplier,
            )
            traces.append({"subquery": part, "hits": hits})
            for h in hits:
                collected_evidence.append((h["text"], h.get("meta", {})))

        # extractive synthesis: take top passages and cite by source
        combined = "\n\n".join([t for t, _ in collected_evidence[: top_k * 2]])
        synthesis = f"Extractive Summary:\n\n{combined}"
        if False:  # placeholder for LLM-enabled abstractive
            synthesis = self.llm.synthesize(query + "\n" + combined)

        return {"synthesis": synthesis, "traces": traces}
