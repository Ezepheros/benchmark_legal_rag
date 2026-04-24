"""
GeminiQueryRewriter — rewrites a natural-language query into legal language.

Used as a pre-retrieval step: the rewritten query is embedded and used to search
the index instead of the raw user query.
"""
from __future__ import annotations

import logging
import time

from benchmark_rag.prompts.query_rewriter import REWRITER_SYSTEM_PROMPT as _SYSTEM_PROMPT

log = logging.getLogger(__name__)

_RETRY_DELAYS = [5, 10, 30, 60, 120]  # seconds between attempts


def _get_api_key(api_key: str | None) -> str:
    import os
    key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise EnvironmentError(
            "No Google API key found for GeminiQueryRewriter. "
            "Set GOOGLE_API_KEY or GEMINI_API_KEY."
        )
    return key


class GeminiQueryRewriter:
    """
    Rewrites queries into legal language using Gemini.

    Parameters
    ----------
    model_name:
        Gemini model ID, default "gemini-2.5-flash".
    api_key:
        Google API key. Falls back to GOOGLE_API_KEY / GEMINI_API_KEY env vars.
    temperature:
        Sampling temperature (0.0 = greedy).
    max_output_tokens:
        Upper limit on rewritten query tokens.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._api_key = api_key
        self._client = None
        self._call_count: int = 0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0

    def _load(self) -> None:
        if self._client is not None:
            return
        from google import genai
        self._client = genai.Client(api_key=_get_api_key(self._api_key))

    def rewrite(self, query: str) -> str:
        """Return the query rephrased in legal language, retrying on 503 errors."""
        from google.genai import types
        from google.genai.errors import ServerError
        self._load()
        config = types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        for attempt, delay in enumerate([0] + _RETRY_DELAYS):
            if delay:
                log.warning("Gemini 503 — retrying in %ds (attempt %d/%d)...", delay, attempt, len(_RETRY_DELAYS) + 1)
                time.sleep(delay)
            try:
                response = self._client.models.generate_content(  # type: ignore[union-attr]
                    model=self.model_name,
                    contents=query,
                    config=config,
                )
                break
            except ServerError as e:
                if e.code != 503 or attempt == len(_RETRY_DELAYS):
                    raise
        usage = response.usage_metadata
        self._call_count += 1
        self._total_input_tokens += usage.prompt_token_count
        self._total_output_tokens += usage.candidates_token_count
        rewritten = response.text.strip()
        log.debug("Query rewrite: %r → %r", query, rewritten)
        return rewritten

    def log_usage_summary(self) -> None:
        log.info(
            "GeminiQueryRewriter usage summary | model=%s | calls=%d"
            " | total_input_tokens=%d | total_output_tokens=%d",
            self.model_name, self._call_count,
            self._total_input_tokens, self._total_output_tokens,
        )
