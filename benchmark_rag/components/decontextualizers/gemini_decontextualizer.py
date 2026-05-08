"""
Gemini-based chunk decontextualizer.

Takes a list of statement strings and their source document, replaces vague
references (pronouns, "the applicant", "the Act", etc.) with specific entities
from the document, and returns the revised statements as a JSON array.

If the model's response is not valid JSON, the decontextualizer sends one
follow-up asking the model to fix the format.  If that also fails, it returns
the original statements unchanged with a per-statement ``failed`` flag so
downstream code can tell which chunks were not decontextualized.
"""
from __future__ import annotations

import json
import logging

from benchmark_rag.prompts.decontextualizer import (
    DECONTEXTUALIZE_INSTRUCTION,
    DECONTEXTUALIZE_REMINDER,
    DECONTEXTUALIZE_SYSTEM_PROMPT,
)

log = logging.getLogger(__name__)


def _generate_with_retry(client, **kwargs):
    """Re-use the same 429/503 retry helper from gemini.py."""
    from benchmark_rag.components.generators.gemini import _generate_with_retry as _retry
    return _retry(client, **kwargs)


def _estimate_cost_gemini_25_pro(in_tok: int, out_tok: int) -> float:
    """Tiered pricing for Gemini 2.5 Pro (both tiers keyed on prompt length).

    Prompts <= 200k tokens: $2.25/1M input, $18.00/1M output (incl. thinking).
    Prompts >  200k tokens: $4.50/1M input, $27.00/1M output (incl. thinking).
    """
    if in_tok <= 200_000:
        return in_tok * (2.25 / 1_000_000) + out_tok * (18.00 / 1_000_000)
    return in_tok * (4.50 / 1_000_000) + out_tok * (27.00 / 1_000_000)


def _get_api_key(api_key, caller):
    from benchmark_rag.components.generators.gemini import _get_api_key
    return _get_api_key(api_key, caller)


_JSON_FIX_PROMPT = (
    "Your previous response was not valid JSON. "
    "Please respond with ONLY a JSON array of revised statements, "
    "e.g. [\"statement 1\", \"statement 2\", ...]. "
    "No markdown fences, no commentary — just the raw JSON array."
)


def _try_parse_json_array(text: str) -> list[str] | None:
    """Parse text as a JSON array of strings, stripping markdown fences if present."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(parsed, list) and all(isinstance(s, str) for s in parsed):
        return parsed
    return None


class GeminiDecontextualizer:
    """
    Decontextualizes a batch of statements using the source document as context.

    Parameters
    ----------
    model_name:
        Gemini model ID, default ``gemini-2.5-pro``.
    api_key:
        Google API key.  Falls back to ``GOOGLE_API_KEY`` / ``GEMINI_API_KEY``.
    temperature:
        Sampling temperature.
    max_output_tokens:
        Upper limit on generated tokens per call (includes thinking tokens).
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: str | None = None,
        temperature: float = 0.05,
        max_output_tokens: int = 32000,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._api_key = api_key
        self._client = None

        self._call_count: int = 0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cost: float | None = None
        self._json_fix_count: int = 0
        self._json_fail_count: int = 0
        self._last_raw_responses: list[str] = []

    def _load(self) -> None:
        if self._client is not None:
            return
        from google import genai
        self._client = genai.Client(
            api_key=_get_api_key(self._api_key, "GeminiDecontextualizer"),
        )

    def _track_and_log(self, in_tok: int, out_tok: int) -> None:
        call_cost = _estimate_cost_gemini_25_pro(in_tok, out_tok)
        self._call_count += 1
        self._total_input_tokens += in_tok
        self._total_output_tokens += out_tok
        self._total_cost = (self._total_cost or 0.0) + call_cost
        cost_str = f"{call_cost:.6f}"
        total_str = f"{self._total_cost:.6f}" if self._total_cost is not None else "N/A"
        log.info(
            "GeminiDecontextualizer model=%s | call %d: in=%d out=%d cost=$%s"
            " | total: in=%d out=%d cost=$%s",
            self.model_name, self._call_count, in_tok, out_tok, cost_str,
            self._total_input_tokens, self._total_output_tokens, total_str,
        )

    def log_usage_summary(self) -> None:
        total_str = f"{self._total_cost:.6f}" if self._total_cost is not None else "N/A"
        log.info(
            "GeminiDecontextualizer usage summary | model=%s | calls=%d"
            " | total_in=%d | total_out=%d | cost=$%s"
            " | json_fixes=%d | json_failures=%d",
            self.model_name, self._call_count,
            self._total_input_tokens, self._total_output_tokens, total_str,
            self._json_fix_count, self._json_fail_count,
        )

    def decontextualize(
        self,
        statements: list[str],
        document_text: str,
    ) -> list[str] | None:
        """
        Decontextualize a batch of statements given their source document.

        Returns
        -------
        list[str]
            The revised statements on success.
        None
            If the model returned unparseable JSON on both attempts.
        """
        from google.genai import types

        self._load()

        numbered = "\n".join(
            f"{i+1}. {s}" for i, s in enumerate(statements)
        )
        prompt = (
            f"{DECONTEXTUALIZE_INSTRUCTION}\n\n"
            f"STATEMENTS TO DECONTEXTUALIZE:\n{numbered}\n\n"
            f"===== DOCUMENT =====\n{document_text}\n===== END DOCUMENT =====\n\n"
            f"STATEMENTS TO DECONTEXTUALIZE (repeated for reference):\n{numbered}\n\n"
            f"{DECONTEXTUALIZE_REMINDER}"
        )

        config = types.GenerateContentConfig(
            system_instruction=DECONTEXTUALIZE_SYSTEM_PROMPT,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

        # --- First attempt ---
        response = _generate_with_retry(
            self._client,
            model=self.model_name,
            contents=prompt,
            config=config,
        )
        usage = response.usage_metadata
        self._track_and_log(
            usage.prompt_token_count or 0,
            usage.candidates_token_count or 0,
        )

        raw_text_1 = response.text or ""
        revised = _try_parse_json_array(raw_text_1)
        if revised is not None:
            self._last_raw_responses = [raw_text_1]
            return revised

        # --- Retry once on bad JSON ---
        log.warning(
            "GeminiDecontextualizer: first response was not valid JSON, "
            "requesting fix (batch of %d statements). "
            "First 500 chars of response: %.500s",
            len(statements), raw_text_1,
        )
        self._json_fix_count += 1

        fix_response = _generate_with_retry(
            self._client,
            model=self.model_name,
            contents=[
                types.Content(role="user", parts=[types.Part(text=prompt)]),
                types.Content(
                    role="model",
                    parts=[types.Part(text=raw_text_1)],
                ),
                types.Content(
                    role="user",
                    parts=[types.Part(text=_JSON_FIX_PROMPT)],
                ),
            ],
            config=config,
        )
        fix_usage = fix_response.usage_metadata
        self._track_and_log(
            fix_usage.prompt_token_count or 0,
            fix_usage.candidates_token_count or 0,
        )
        raw_text_2 = fix_response.text or ""
        revised = _try_parse_json_array(raw_text_2)
        if revised is not None:
            self._last_raw_responses = [raw_text_1, raw_text_2]
            return revised

        # --- Both attempts failed — store raw responses for debugging ---
        self._last_raw_responses = [raw_text_1, raw_text_2]
        log.error(
            "GeminiDecontextualizer: JSON parse failed after retry "
            "(batch of %d statements). "
            "First 500 chars of retry response: %.500s",
            len(statements), raw_text_2,
        )
        self._json_fail_count += 1
        return None
