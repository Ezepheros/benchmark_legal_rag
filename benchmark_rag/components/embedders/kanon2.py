"""
Isaacus Kanon2 embedding model.

Kanon2 is a legal-domain embedding model served via the Isaacus API.
API reference: https://docs.isaacus.com
"""
from __future__ import annotations

from benchmark_rag.components.base import BaseEmbedder


class Kanon2Embedder(BaseEmbedder):
    """
    Embeds text using the Isaacus Kanon2 legal embedding model.

    Parameters
    ----------
    model_name:
        Kanon2 model variant, e.g. "kanon-2".
    api_key:
        Isaacus API key.  Falls back to ISAACUS_API_KEY env var if None.
    api_base_url:
        API base URL.  Override for self-hosted deployments.
    batch_size:
        Texts per API request.
    """

    _EMBEDDING_DIM = 1024  # update if the model version changes

    def __init__(
        self,
        model_name: str = "kanon-2",
        api_key: str | None = None,
        api_base_url: str = "https://api.isaacus.com/v1",
        batch_size: int = 64,
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self._api_key = api_key
        self._api_base_url = api_base_url
        self._client = None

    def _load(self):
        if self._client is not None:
            return
        import os
        import httpx

        key = self._api_key or os.environ.get("ISAACUS_API_KEY")
        if not key:
            raise EnvironmentError(
                "No Isaacus API key found. Set ISAACUS_API_KEY or pass api_key= to Kanon2Embedder."
            )
        self._client = httpx.Client(
            base_url=self._api_base_url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=60.0,
        )

    @property
    def embedding_dim(self) -> int:
        return self._EMBEDDING_DIM

    def _embed(self, texts: list[str]) -> list[list[float]]:
        self._load()
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self._client.post(  # type: ignore[union-attr]
                "/embeddings",
                json={"model": self.model_name, "input": batch},
            )
            response.raise_for_status()
            data = response.json()
            # OpenAI-compatible response format
            batch_embeddings = [item["embedding"] for item in data["data"]]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings


if __name__ == "__main__":
    import os
    import numpy as np

    if not os.environ.get("ISAACUS_API_KEY"):
        print("ISAACUS_API_KEY not set — skipping Kanon2Embedder test.")
    else:
        texts = [
            "The accused was found guilty of fraud over $5,000.",
            "The appellant submits the trial judge erred in admitting hearsay evidence.",
            "Promissory estoppel requires a clear and unequivocal promise.",
            "The Crown must prove each element beyond a reasonable doubt.",
            "The quick brown fox jumps over the lazy dog.",
        ]

        print("Loading Kanon2Embedder ...")
        embedder = Kanon2Embedder()
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
