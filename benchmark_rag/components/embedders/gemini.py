"""Google Gemini embedding model via the google-genai SDK."""
from __future__ import annotations

import logging

from benchmark_rag.components.base import BaseEmbedder

log = logging.getLogger(__name__)


class GeminiEmbedder(BaseEmbedder):
    """
    Embeds text using Google's Gemini embedding API.

    Parameters
    ----------
    model_name:
        Gemini embedding model ID, e.g. "gemini-embedding-001".
    task_type:
        Gemini task type hint.  Use "RETRIEVAL_DOCUMENT" for corpus chunks,
        "RETRIEVAL_QUERY" for query embeddings.
    api_key:
        Google API key.  Falls back to GOOGLE_API_KEY / GEMINI_API_KEY env vars if None.
    batch_size:
        Max texts per API call.
    max_cost_usd:
        Maximum estimated spend in USD before raising RuntimeError.
    """

    _EMBEDDING_DIM = 3072  # gemini-embedding-001 output dim
    _COST_PER_1K_TOKENS = 0.0001  # USD estimate (chars/4 token estimate)

    def __init__(
        self,
        model_name: str = "gemini-embedding-001",
        task_type: str = "RETRIEVAL_DOCUMENT",
        api_key: str | None = None,
        batch_size: int = 50,
        max_cost_usd: float | None = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.task_type = task_type
        self.batch_size = batch_size
        self._api_key = api_key
        self.max_cost_usd = max_cost_usd
        self._client = None
        self._total_est_input_tokens: int = 0
        self._total_est_cost_usd: float = 0.0

    def _load(self):
        if self._client is not None:
            return
        import os
        from google import genai

        key = self._api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise EnvironmentError(
                "No Google API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY, "
                "or pass api_key= to GeminiEmbedder."
            )
        self._client = genai.Client(api_key=key)

    @property
    def embedding_dim(self) -> int:
        return self._EMBEDDING_DIM

    def _track_and_log(self, n_texts: int, call_est_tokens: int) -> None:
        self._total_est_input_tokens += call_est_tokens
        call_est_cost = call_est_tokens / 1000 * self._COST_PER_1K_TOKENS
        self._total_est_cost_usd += call_est_cost
        log.info(
            "GeminiEmbedder model=%s task=%s | embed call: texts=%d est_input_tokens=%d est_cost=$%.4f"
            " | running total est_input_tokens=%d est_cost=$%.4f"
            " (note: embedding API does not report token counts; estimate = chars/4)",
            self.model_name, self.task_type, n_texts, call_est_tokens, call_est_cost,
            self._total_est_input_tokens, self._total_est_cost_usd,
        )

    def log_usage_summary(self) -> None:
        log.info(
            "GeminiEmbedder usage summary | model=%s | total_est_input_tokens=%d"
            " total_est_cost=$%.4f (limit=$%s)"
            " (note: embedding API does not report token counts; estimate = chars/4)",
            self.model_name, self._total_est_input_tokens, self._total_est_cost_usd,
            f"{self.max_cost_usd:.2f}" if self.max_cost_usd is not None else "none",
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        from google.genai import types

        self._load()
        all_embeddings: list[list[float]] = []
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        call_est_tokens = 0

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_est_tokens = sum(len(t) for t in batch) // 4
            batch_est_cost = batch_est_tokens / 1000 * self._COST_PER_1K_TOKENS
            if self.max_cost_usd is not None:
                if self._total_est_cost_usd + batch_est_cost > self.max_cost_usd:
                    raise RuntimeError(
                        f"GeminiEmbedder cost limit of ${self.max_cost_usd:.2f} would be exceeded "
                        f"(accumulated: ${self._total_est_cost_usd:.4f}, "
                        f"next batch est: ${batch_est_cost:.4f}). "
                        f"Stopping before batch {i // self.batch_size + 1}/{n_batches}."
                    )
            result = self._client.models.embed_content(  # type: ignore[union-attr]
                model=self.model_name,
                contents=batch,
                config=types.EmbedContentConfig(task_type=self.task_type),
            )
            all_embeddings.extend([e.values for e in result.embeddings])
            call_est_tokens += batch_est_tokens
            log.debug(
                "GeminiEmbedder batch %d/%d: texts=%d est_input_tokens=%d",
                i // self.batch_size + 1, n_batches, len(batch), batch_est_tokens,
            )

        self._track_and_log(len(texts), call_est_tokens)
        return all_embeddings


if __name__ == "__main__":
    import os
    import numpy as np

    if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        print("GOOGLE_API_KEY / GEMINI_API_KEY not set — skipping GeminiEmbedder test.")
    else:
        texts = [
            "The accused was found guilty of fraud over $5,000.",
            "The appellant submits the trial judge erred in admitting hearsay evidence.",
            "Promissory estoppel requires a clear and unequivocal promise.",
            "The Crown must prove each element beyond a reasonable doubt.",
            "The quick brown fox jumps over the lazy dog.",
        ]

        print("Loading GeminiEmbedder ...")
        embedder = GeminiEmbedder(task_type="RETRIEVAL_DOCUMENT")
        embeddings = embedder.embed(texts)

        print(f"Embedding dim : {embedder.embedding_dim}")
        print(f"Vectors shape : {len(embeddings)} x {len(embeddings[0])}")
        print(f"Call count    : {embedder.call_count}")
        embedder.log_usage_summary()

        emb = np.array(embeddings)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb_norm = emb / norms
        sim = emb_norm @ emb_norm.T
        print("\nCosine similarity matrix:")
        for i, row in enumerate(sim):
            scores = "  ".join(f"{v:.2f}" for v in row)
            print(f"  [{i}]  {scores}  | {texts[i][:50]}")
