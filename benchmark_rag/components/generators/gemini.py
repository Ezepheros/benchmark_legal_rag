"""
Gemini 2.5 Flash generator and Gemini 2.5 Pro judge.

Both share the same Google Generative AI SDK; they differ only in model name
and the prompt they receive.
"""
from __future__ import annotations

from benchmark_rag.components.base import BaseGenerator, RetrievedChunk


_DEFAULT_SYSTEM_PROMPT = (
    "You are a legal research assistant. Answer the question accurately and concisely "
    "using only the provided context passages. If the context does not contain enough "
    "information to answer, say so clearly. Cite the relevant passage(s) when possible."
)

_DEFAULT_JUDGE_PROMPT = (
    "You are an expert legal judge evaluating the quality of a RAG system's answer.\n\n"
    "Given:\n"
    "  - A legal question\n"
    "  - A generated answer\n"
    "  - The ground-truth reference answer\n\n"
    "Score the generated answer on a scale of 1–5 for each of:\n"
    "  1. Faithfulness: Is the answer grounded in the retrieved context?\n"
    "  2. Correctness: Does it match the reference answer?\n"
    "  3. Completeness: Does it cover all key points in the reference?\n\n"
    "Respond ONLY with valid JSON in this exact format:\n"
    '{"faithfulness": <1-5>, "correctness": <1-5>, "completeness": <1-5>, "rationale": "<brief explanation>"}'
)


def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        citation = chunk.metadata.get("citation_en", chunk.doc_id)
        parts.append(f"[{i}] ({citation})\n{chunk.text}")
    return "\n\n".join(parts)


class GeminiGenerator(BaseGenerator):
    """
    Generates answers using Gemini 2.5 Flash.

    Parameters
    ----------
    model_name:
        Gemini model ID, default "gemini-2.5-flash".
    system_prompt:
        Instruction prepended to every request.
    api_key:
        Google API key.  Falls back to GOOGLE_API_KEY env var.
    temperature:
        Sampling temperature (0.0 = greedy).
    max_output_tokens:
        Upper limit on generated tokens.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._api_key = api_key
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import os
        import google.generativeai as genai

        key = self._api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise EnvironmentError(
                "No Google API key found. Set GOOGLE_API_KEY or pass api_key= to GeminiGenerator."
            )
        genai.configure(api_key=key)
        self._model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.system_prompt,
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            ),
        )

    def generate(self, query: str, context_chunks: list[RetrievedChunk]) -> str:
        self._load()
        context = _build_context(context_chunks)
        prompt = f"Context:\n{context}\n\nQuestion: {query}"
        response = self._model.generate_content(prompt)  # type: ignore[union-attr]
        return response.text


class GeminiJudge:
    """
    Evaluates a (query, generated_answer, reference_answer) triple using
    Gemini 2.5 Pro as an LLM-as-a-judge.

    Returns a dict with keys: faithfulness, correctness, completeness, rationale.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-pro",
        api_key: str | None = None,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self._api_key = api_key
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import os
        import google.generativeai as genai

        key = self._api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise EnvironmentError(
                "No Google API key. Set GOOGLE_API_KEY or pass api_key= to GeminiJudge."
            )
        genai.configure(api_key=key)
        self._model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=_DEFAULT_JUDGE_PROMPT,
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                response_mime_type="application/json",
            ),
        )

    def judge(
        self,
        query: str,
        generated_answer: str,
        reference_answer: str,
    ) -> dict:
        """
        Returns
        -------
        dict with keys: faithfulness (1-5), correctness (1-5),
                        completeness (1-5), rationale (str).
        """
        import json

        self._load()
        prompt = (
            f"Question: {query}\n\n"
            f"Generated answer: {generated_answer}\n\n"
            f"Reference answer: {reference_answer}"
        )
        response = self._model.generate_content(prompt)  # type: ignore[union-attr]
        return json.loads(response.text)


if __name__ == "__main__":
    import os
    from benchmark_rag.components.base import RetrievedChunk

    if not os.environ.get("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY not set — skipping Gemini generator/judge test.")
    else:
        fake_chunks = [
            RetrievedChunk(
                text=(
                    "The trial judge found that the warrantless search of the appellant's "
                    "business premises violated s. 8 of the Charter. The documents seized "
                    "were excluded under s. 24(2) as their admission would bring the "
                    "administration of justice into disrepute."
                ),
                doc_id="2022 ONCA 100",
                chunk_idx=0,
                metadata={"citation_en": "2022 ONCA 100"},
                score=0.91,
            ),
            RetrievedChunk(
                text=(
                    "An office manager does not have actual or apparent authority to waive "
                    "an employer's Charter rights. Consent to search must come from someone "
                    "with actual authority over the premises and knowledge of the right being waived."
                ),
                doc_id="2022 ONCA 100",
                chunk_idx=1,
                metadata={"citation_en": "2022 ONCA 100"},
                score=0.87,
            ),
        ]
        query = "Can an office manager consent to a warrantless search on behalf of their employer?"

        # --- Generator ---
        print("Testing GeminiGenerator (gemini-2.5-flash) ...")
        generator = GeminiGenerator()
        answer = generator.generate(query, fake_chunks)
        print(f"\nQuery : {query}")
        print(f"Answer: {answer}")

        # --- Judge ---
        reference = (
            "No. An office manager lacks authority to waive Charter rights on behalf of their employer. "
            "Such consent must come from someone with actual authority and knowledge of the right being waived."
        )
        print("\nTesting GeminiJudge (gemini-2.5-pro) ...")
        judge = GeminiJudge()
        scores = judge.judge(query, answer, reference)
        print(f"Judge scores: {scores}")
