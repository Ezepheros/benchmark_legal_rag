"""
Agentic RAG pipeline: a Gemini LLM drives iterative keyword search over a
BM25 index, explicitly curating relevant documents before synthesising a
final answer.

The agent alternates between two phases each iteration:

  Search phase  — tools: keyword_search, answer
    The agent searches for cases using BM25 keywords, or calls answer()
    if it already has enough saved documents.

  Review phase  — tools: save_citations, answer
    After seeing the search results, the agent saves the relevant citations
    (with search-term provenance) or calls answer() to stop searching.

Each save_citations response includes a full state summary (saved documents +
queries run so far), giving the agent full context before the next search.

Three tools:
  keyword_search(query, k)   BM25 search; available in search phase only.
  save_citations(citations)  Save relevant citations; available in review phase only.
  answer()                   Exit signal; available in both phases.

When answer() is called (or max_iterations is exhausted), a separate final-
answer generation step is performed with all saved document chunks as context
and no tools.

Usage
-----
    python scripts/run_benchmark.py \\
        --config configs/experiments/agentic_gemini_bm25_recursive_1024.yaml

Requires: GOOGLE_API_KEY or GEMINI_API_KEY environment variable.
Requires a pre-built BM25 index (run run_indexing.py first).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from benchmark_rag.components.base import EmbeddedChunk, RetrievedChunk
from benchmark_rag.components.retrievers.bm25_retriever import BM25Retriever
from benchmark_rag.config.schemas import ExperimentConfig
from benchmark_rag.logging import get_logger
from benchmark_rag.pipeline.rag_pipeline import QueryResult
from benchmark_rag.prompts.agentic import (
    ANSWER_SYSTEM_PROMPT as _ANSWER_SYSTEM_PROMPT,
    REVIEW_SYSTEM_PROMPT as _REVIEW_SYSTEM_PROMPT,
    SEARCH_SYSTEM_PROMPT as _SEARCH_SYSTEM_PROMPT,
)

log = logging.getLogger(__name__)

_PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.3 / 1_000_000, 2.5 / 1_000_000),
    "gemini-2.5-pro":   (1.25 / 1_000_000, 10.0 / 1_000_000),
}

@dataclass
class _SavedDoc:
    citation: str
    search_query: str       # the keyword_search query that led to saving this document
    chunks: list[RetrievedChunk]


def _estimate_cost(model_name: str, in_tok: int, out_tok: int) -> float | None:
    for prefix, (in_price, out_price) in _PRICING.items():
        if model_name.startswith(prefix):
            return in_tok * in_price + out_tok * out_price
    return None


def _format_chunks(chunks: list[RetrievedChunk], show_score: bool = True) -> str:
    parts = []
    for c in chunks:
        header = f"[{c.doc_id} chunk {c.chunk_idx}]"
        if show_score and c.score > 0:
            header += f" (score={c.score:.3f})"
        parts.append(f"{header}\n{c.text}")
    return "\n\n".join(parts)


class AgenticRAGPipeline:
    """
    RAG pipeline where a Gemini LLM alternates between a search phase and a
    review phase to iteratively curate relevant documents from a BM25 index
    before synthesising a final answer.

    Parameters
    ----------
    retriever:
        Pre-loaded BM25Retriever.
    model_name:
        Gemini model ID used for all agent calls and the final answer.
    max_iterations:
        Hard cap on search→review iteration rounds.  If the LLM has not
        called answer() by this point, final answer generation is forced.
    max_k_per_search:
        Upper bound on k for a single keyword_search call.
    max_doc_chunks:
        Upper bound on chunks loaded per saved document (get_document style).
    api_key:
        Google API key.  Falls back to GOOGLE_API_KEY / GEMINI_API_KEY env vars.
    """

    def __init__(
        self,
        retriever: BM25Retriever,
        model_name: str = "gemini-2.5-flash",
        max_iterations: int = 5,
        max_k_per_search: int = 10,
        max_doc_chunks: int = 15,
        api_key: str | None = None,
    ):
        self.retriever = retriever
        self._model_name = model_name
        self._max_iterations = max_iterations
        self._max_k_per_search = max_k_per_search
        self._max_doc_chunks = max_doc_chunks
        self._api_key = api_key
        self._client = None
        self._doc_chunks: dict[str, list[EmbeddedChunk]] = {}
        self.log = get_logger(__name__)

        self._call_count: int = 0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cost: float | None = None

        self._build_doc_lookup()

    def _build_doc_lookup(self) -> None:
        for chunk in self.retriever._chunks:
            self._doc_chunks.setdefault(chunk.doc_id, []).append(chunk)
        for chunks in self._doc_chunks.values():
            chunks.sort(key=lambda c: c.chunk_idx)

    def _load_client(self) -> None:
        if self._client is not None:
            return
        import os
        from google import genai

        key = (
            self._api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        if not key:
            raise EnvironmentError(
                "No Google API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY, "
                "or pass api_key= to AgenticRAGPipeline."
            )
        self._client = genai.Client(api_key=key)

    def _track_and_log(self, in_tok: int, out_tok: int) -> None:
        cost = _estimate_cost(self._model_name, in_tok, out_tok)
        self._call_count += 1
        self._total_input_tokens += in_tok
        self._total_output_tokens += out_tok
        if cost is not None:
            self._total_cost = (self._total_cost or 0.0) + cost
        self.log.info(
            "AgenticRAGPipeline model=%s | call %d: in=%d out=%d cost=%s | "
            "running total: in=%d out=%d cost=%s",
            self._model_name, self._call_count, in_tok, out_tok,
            f"${cost:.6f}" if cost is not None else "N/A",
            self._total_input_tokens, self._total_output_tokens,
            f"${self._total_cost:.6f}" if self._total_cost is not None else "N/A",
        )

    def log_usage_summary(self) -> None:
        self.log.info(
            "AgenticRAGPipeline usage summary | model=%s | calls=%d "
            "| total_in=%d | total_out=%d | total_cost=%s",
            self._model_name, self._call_count,
            self._total_input_tokens, self._total_output_tokens,
            f"${self._total_cost:.6f}" if self._total_cost is not None else "N/A",
        )

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _keyword_search(self, query: str, k: int) -> tuple[str, list[RetrievedChunk]]:
        k = min(k, self._max_k_per_search)
        self.log.debug("AgenticRAGPipeline tool=keyword_search query=%r k=%d", query, k)
        if not query.strip():
            return "Empty query — please provide search terms.", []
        chunks = self.retriever.retrieve_text(query, k=k)
        if not chunks:
            return "No results found for the given query.", []
        return _format_chunks(chunks, show_score=True), chunks

    def _save_citations(
        self,
        citations: list[str],
        search_query: str,
        saved_docs: dict[str, _SavedDoc],
    ) -> str:
        """
        Record citations as relevant, loading their chunks from the BM25 index.
        Skips citations already saved.  Returns a confirmation string.
        """
        saved_new: list[str] = []
        not_found: list[str] = []

        for citation in citations:
            citation = citation.strip()
            if not citation or citation in saved_docs:
                continue
            raw = self._doc_chunks.get(citation, [])
            if not raw:
                self.log.warning(
                    "AgenticRAGPipeline save_citations: no chunks for citation %r", citation
                )
                not_found.append(citation)
                continue
            chunks = [
                RetrievedChunk(
                    text=c.text,
                    doc_id=c.doc_id,
                    chunk_idx=c.chunk_idx,
                    metadata=c.metadata,
                    embedding=c.embedding,
                    score=0.0,
                )
                for c in raw[: self._max_doc_chunks]
            ]
            saved_docs[citation] = _SavedDoc(
                citation=citation,
                search_query=search_query,
                chunks=chunks,
            )
            saved_new.append(citation)

        parts: list[str] = []
        if saved_new:
            parts.append(f"Saved {len(saved_new)} citation(s): {', '.join(saved_new)}.")
        if not_found:
            parts.append(
                f"Not found in database (check spelling): {', '.join(not_found)}."
            )
        if not parts:
            parts.append(
                "No new citations saved (all were already saved or list was empty)."
            )
        return " ".join(parts)

    # ------------------------------------------------------------------
    # State formatting helpers
    # ------------------------------------------------------------------

    def _format_state(
        self,
        saved_docs: dict[str, _SavedDoc],
        searches_run: list[str],
    ) -> str:
        """
        Compact state summary injected into save_citations responses so the
        agent has full context before choosing the next search.
        """
        lines: list[str] = ["=== Research State ==="]

        if searches_run:
            lines.append(
                "Searches run: " + " | ".join(f'"{s}"' for s in searches_run)
            )
        else:
            lines.append("No searches run yet.")

        if saved_docs:
            lines.append(f"\nSaved documents ({len(saved_docs)}):")
            for doc in saved_docs.values():
                lines.append(f"\n[{doc.citation}]  (found via: \"{doc.search_query}\")")
                # Show first chunk truncated so the agent can remember what each doc is
                if doc.chunks:
                    snippet = doc.chunks[0].text[:300].replace("\n", " ")
                    lines.append(f"  {snippet}...")
        else:
            lines.append("\nNo documents saved yet.")

        return "\n".join(lines)

    def _format_saved_for_answer(self, saved_docs: dict[str, _SavedDoc]) -> str:
        """Full chunk text for all saved documents, used in the final answer call."""
        parts: list[str] = []
        for doc in saved_docs.values():
            header = f"=== {doc.citation} (found via: \"{doc.search_query}\") ==="
            body = _format_chunks(doc.chunks, show_score=False)
            parts.append(f"{header}\n{body}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: ExperimentConfig) -> "AgenticRAGPipeline":
        """
        Build an AgenticRAGPipeline from an ExperimentConfig.

        Loads the BM25 index from cfg.indexing.output_dir.  The index must
        have been built by run_indexing.py using a HybridRetriever or
        BM25Retriever config with matching dataset + chunker + embedder
        settings (same index_id).
        """
        assert cfg.agentic is not None, (
            "AgenticRAGPipeline requires an 'agentic:' section in the experiment config."
        )
        retriever = BM25Retriever()
        index_path = Path(cfg.indexing.output_dir) / "index"
        retriever.load_index(index_path)

        return cls(
            retriever=retriever,
            model_name=cfg.agentic.model_name,
            max_iterations=cfg.agentic.max_iterations,
            max_k_per_search=cfg.agentic.max_k_per_search,
            max_doc_chunks=cfg.agentic.max_doc_chunks,
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, query_text: str, k: int | None = None) -> QueryResult:
        """
        Run the agentic search loop for one query.

        Alternates between search phase (keyword_search | answer) and review
        phase (save_citations | answer) for up to max_iterations rounds.
        When answer() is called or iterations are exhausted, a separate final-
        answer generation call synthesises the answer from all saved chunks.

        Returns a QueryResult where retrieved_chunks is the ordered union of
        all saved document chunks (save order, chunk order within each doc).
        """
        from google.genai import types

        self._load_client()

        # Per-query mutable state
        saved_docs: dict[str, _SavedDoc] = {}   # citation → _SavedDoc
        searches_run: list[str] = []
        last_search_query: str = ""

        # ---- Tool declarations ----------------------------------------
        keyword_search_decl = types.FunctionDeclaration(
            name="keyword_search",
            description=(
                "Search the legal document database by keyword using BM25 ranking. "
                "Returns the most relevant text passages for the given keywords."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description=(
                            "Keyword search query — legal terms, case names, legal concepts."
                        ),
                    ),
                    "k": types.Schema(
                        type=types.Type.INTEGER,
                        description=(
                            f"Number of passages to return (max {self._max_k_per_search})."
                        ),
                    ),
                },
                required=["query"],
            ),
        )
        save_citations_decl = types.FunctionDeclaration(
            name="save_citations",
            description=(
                "Save citations identified as relevant from the last search. "
                "Saved documents (with the search query that found them) will be shown "
                "at the start of your next search."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "citations": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.STRING),
                        description=(
                            "Legal citations to save, e.g. ['2022 BCSC 100', '2021 SCC 5']."
                        ),
                    ),
                },
                required=["citations"],
            ),
        )
        answer_decl = types.FunctionDeclaration(
            name="answer",
            description=(
                "Stop searching and generate the final answer from your saved documents."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
            ),
        )

        all_tools = types.Tool(
            function_declarations=[keyword_search_decl, save_citations_decl, answer_decl]
        )

        # ---- Phase configs (differ only in allowed_function_names) ----
        search_cfg = types.GenerateContentConfig(
            system_instruction=_SEARCH_SYSTEM_PROMPT,
            tools=[all_tools],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=["keyword_search", "answer"],
                )
            ),
            temperature=0.0,
        )
        review_cfg = types.GenerateContentConfig(
            system_instruction=_REVIEW_SYSTEM_PROMPT,
            tools=[all_tools],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=["save_citations", "answer"],
                )
            ),
            temperature=0.0,
        )

        contents: list[types.Content] = [
            types.Content(role="user", parts=[types.Part(text=query_text)])
        ]

        done = False
        n_iterations = 0

        for n_iterations in range(1, self._max_iterations + 1):

            # ---- Search phase -----------------------------------------
            response = self._client.models.generate_content(  # type: ignore[union-attr]
                model=self._model_name,
                contents=contents,
                config=search_cfg,
            )
            usage = response.usage_metadata
            self._track_and_log(usage.prompt_token_count, usage.candidates_token_count)

            candidate = response.candidates[0].content
            contents.append(candidate)

            fc = next((p.function_call for p in candidate.parts if p.function_call), None)
            if fc is None or fc.name == "answer":
                self.log.info(
                    "AgenticRAGPipeline: answer() in search phase — iteration=%d",
                    n_iterations,
                )
                done = True
                break

            # Execute keyword_search
            search_query = str(fc.args.get("query", ""))
            k_req = int(fc.args.get("k", self._max_k_per_search))
            result_text, _ = self._keyword_search(search_query, k_req)
            last_search_query = search_query
            searches_run.append(search_query)

            self.log.debug(
                "AgenticRAGPipeline iteration=%d search=%r", n_iterations, search_query
            )

            contents.append(types.Content(
                role="user",
                parts=[types.Part(
                    function_response=types.FunctionResponse(
                        name="keyword_search",
                        response={"result": result_text},
                    )
                )],
            ))

            # ---- Review phase -----------------------------------------
            response = self._client.models.generate_content(  # type: ignore[union-attr]
                model=self._model_name,
                contents=contents,
                config=review_cfg,
            )
            usage = response.usage_metadata
            self._track_and_log(usage.prompt_token_count, usage.candidates_token_count)

            candidate = response.candidates[0].content
            contents.append(candidate)

            fc = next((p.function_call for p in candidate.parts if p.function_call), None)
            if fc is None or fc.name == "answer":
                self.log.info(
                    "AgenticRAGPipeline: answer() in review phase — iteration=%d",
                    n_iterations,
                )
                done = True
                break

            # Execute save_citations
            raw_citations: list[str] = list(fc.args.get("citations", []))
            confirm_text = self._save_citations(raw_citations, last_search_query, saved_docs)

            # Append state summary to the tool response so the agent has full
            # context before deciding what to search next.
            state_text = self._format_state(saved_docs, searches_run)
            full_response = f"{confirm_text}\n\n{state_text}"

            self.log.debug(
                "AgenticRAGPipeline iteration=%d save_citations=%r — %s",
                n_iterations, raw_citations, confirm_text,
            )

            contents.append(types.Content(
                role="user",
                parts=[types.Part(
                    function_response=types.FunctionResponse(
                        name="save_citations",
                        response={"result": full_response},
                    )
                )],
            ))

        if not done:
            self.log.info(
                "AgenticRAGPipeline: max_iterations=%d reached — forcing final answer",
                self._max_iterations,
            )

        # ---- Step 5: final answer generation (no tools) ---------------
        final_text = self._generate_final_answer(query_text, saved_docs)

        # Flatten saved docs into retrieved_chunks in save order
        retrieved_chunks: list[RetrievedChunk] = []
        for doc in saved_docs.values():
            retrieved_chunks.extend(doc.chunks)

        self.log.info(
            "AgenticRAGPipeline: complete | iterations=%d | saved_docs=%d | "
            "retrieved_chunks=%d | searches=%d",
            n_iterations, len(saved_docs), len(retrieved_chunks), len(searches_run),
        )

        return QueryResult(
            query=query_text,
            retrieved_chunks=retrieved_chunks,
            answer=final_text or None,
            metadata={
                "iterations": n_iterations,
                "saved_docs": list(saved_docs.keys()),
                "searches_run": searches_run,
            },
        )

    def _generate_final_answer(
        self,
        query_text: str,
        saved_docs: dict[str, _SavedDoc],
    ) -> str:
        """Separate generation call — no tools — from all saved document chunks."""
        from google.genai import types

        if saved_docs:
            context = self._format_saved_for_answer(saved_docs)
        else:
            context = "No documents were saved during the research phase."

        prompt = f"Question: {query_text}\n\n{context}"

        response = self._client.models.generate_content(  # type: ignore[union-attr]
            model=self._model_name,
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
            config=types.GenerateContentConfig(
                system_instruction=_ANSWER_SYSTEM_PROMPT,
                temperature=0.0,
            ),
        )
        usage = response.usage_metadata
        self._track_and_log(usage.prompt_token_count, usage.candidates_token_count)
        return response.text or ""

    def batch_query(self, queries: list[str], k: int | None = None) -> list[QueryResult]:
        return [self.query(q, k=k) for q in queries]
