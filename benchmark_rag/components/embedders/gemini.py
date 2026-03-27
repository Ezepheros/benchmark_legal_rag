"""Google Gemini embedding model via google-generativeai SDK."""
from __future__ import annotations

from benchmark_rag.components.base import BaseEmbedder


class GeminiEmbedder(BaseEmbedder):
    """
    Embeds text using Google's Gemini embedding API.

    Parameters
    ----------
    model_name:
        Gemini embedding model ID, e.g. "models/gemini-embedding-exp-03-07".
    task_type:
        Gemini task type hint.  Use "RETRIEVAL_DOCUMENT" for corpus chunks,
        "RETRIEVAL_QUERY" for query embeddings.
    api_key:
        Google API key.  Falls back to GOOGLE_API_KEY env var if None.
    batch_size:
        Max texts per API call (Gemini batch endpoint limit is 100).
    """

    _EMBEDDING_DIM = 3072  # gemini-embedding-exp-03-07 output dim

    def __init__(
        self,
        model_name: str = "models/gemini-embedding-exp-03-07",
        task_type: str = "RETRIEVAL_DOCUMENT",
        api_key: str | None = None,
        batch_size: int = 50,
    ):
        super().__init__()
        self.model_name = model_name
        self.task_type = task_type
        self.batch_size = batch_size
        self._api_key = api_key
        self._client = None

    def _load(self):
        if self._client is not None:
            return
        import os
        import google.generativeai as genai

        key = self._api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise EnvironmentError(
                "No Google API key found. Set GOOGLE_API_KEY or pass api_key= to GeminiEmbedder."
            )
        genai.configure(api_key=key)
        self._client = genai

    @property
    def embedding_dim(self) -> int:
        return self._EMBEDDING_DIM

    def _embed(self, texts: list[str]) -> list[list[float]]:
        self._load()
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            result = self._client.embed_content(  # type: ignore[union-attr]
                model=self.model_name,
                content=batch,
                task_type=self.task_type,
            )
            all_embeddings.extend(result["embedding"])

        return all_embeddings


if __name__ == "__main__":
    import os
    import numpy as np

    if not os.environ.get("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY not set — skipping GeminiEmbedder test.")
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

        emb = np.array(embeddings)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb_norm = emb / norms
        sim = emb_norm @ emb_norm.T
        print("\nCosine similarity matrix:")
        for i, row in enumerate(sim):
            scores = "  ".join(f"{v:.2f}" for v in row)
            print(f"  [{i}]  {scores}  | {texts[i][:50]}")
