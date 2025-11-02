"""LLM adapter interface for optional abstractive synthesis.

This is a lightweight adapter; by default the system uses extractive synthesis.
"""

from __future__ import annotations

from typing import Dict


class LLMAdapter:
    def __init__(self, config: Dict | None = None):
        self.config = config or {}

    def synthesize(self, prompt: str, max_tokens: int = 512) -> str:
        """Synthesize using a local model adapter.

        This is an abstraction point for local LLMs. By default, it returns the prompt.
        """
        # Placeholder: integrations for transformers/llamacpp/llama.cpp would go here.
        return "".join(["SYNTHESIS:\n", prompt])
