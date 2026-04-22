"""
Agentic RAG pipeline: an LLM iteratively searches a BM25 index using keyword
tools, then synthesises a final answer from the accumulated passages.

Two tools are available to the agent:

  keyword_search(query, k)
      BM25 search over all indexed chunks.  Analogous to grep — the agent
      picks the search terms and refines them across iterations.

  get_document(citation, max_chunks)
      Returns all chunks for a specific case in document order, up to
      max_chunks.  Analogous to cat — drill into a case the agent already
      identified as relevant.
      # TODO: add pagination support (page parameter, return max_chunks per
      #       page) so very long cases can be browsed incrementally.

The agent loop uses Gemini's native function-calling API.  The conversation
history grows each iteration, giving the LLM full context of what it has
already found before deciding what to search for next.

Usage
-----
    python scripts/run_benchmark.py \
        --config configs/experiments/agentic_gemini_bm25_recursive_1024.yaml

Requires: GOOGLE_API_KEY or GEMINI_API_KEY environment variable.
Requires a pre-built BM25 index (run run_indexing.py with a HybridRetriever
or BM25Retriever config that matches the dataset + chunker + embedder settings).
"""
from __future__ import annotations

import logging
from pathlib import Path

from benchmark_rag.components.base import EmbeddedChunk, RetrievedChunk
from benchmark_rag.components.retrievers.bm25_retriever import BM25Retriever
from benchmark_rag.config.schemas import ExperimentConfig
from benchmark_rag.logging import get_logger
from benchmark_rag.pipeline.rag_pipeline import QueryResult

log = logging.getLogger(__name__)

# Per-token pricing for cost tracking
_PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.075 / 1_000_000, 0.30 / 1_000_000),
    "gemini-2.5-pro":   (1.25  / 1_000_000, 10.0  / 1_000_000),
    "gemini-2.0-flash": (0.10  / 1_000_000, 0.40  / 1_000_000),
}

_SYSTEM_PROMPT = """\
You are a legal research assistant with access to a searchable database of Canadian legal cases.

You have two search tools:

keyword_search(query, k)
  Search the database for text passages matching the given keywords (BM25 ranking).
  Results show the case citation and the matching passage.
  Use this to discover which cases are relevant to the question.

get_document(citation, max_chunks)
  Retrieve passages from a specific case in document order.
  Use this after keyword_search has identified a promising case.

Strategy:
1. Begin with keyword_search using key legal terms from the question.
2. When results mention a promising case, use get_document to read more of it.
3. Refine your searches based on what you find — try different terminology, narrower \
terms, or related legal concepts.
4. When you have enough context to answer the question accurately, stop searching \
and write your answer.

Answer concisely and accurately. Cite relevant cases by their citation when possible.
If the database does not contain sufficient information to answer the question, say so clearly.\
"""


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
    RAG pipeline where a Gemini LLM iteratively drives keyword search over a
    BM25 index, accumulating retrieved chunks across iterations before
    producing a final answer.

    Parameters
    ----------
    retriever:
        Pre-loaded BM25Retriever.  The pipeline also builds an internal
        doc_id → chunks lookup for efficient get_document calls.
    model_name:
        Gemini model ID used for both the agent loop and final answer.
    max_iterations:
        Hard cap on the number of tool-call rounds.  If the LLM has not
        produced a text answer by this point, a final answer is forced.
    max_k_per_search:
        Upper bound on k for a single keyword_search call (guards against
        the agent requesting enormous result sets).
    max_doc_chunks:
        Upper bound on chunks returned by a single get_document call.
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

        # Cost tracking
        self._call_count: int = 0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cost: float | None = None

        # Build doc_id → sorted chunk list once at construction
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

        key = self._api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
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

    def _get_document(self, citation: str, max_chunks: int) -> tuple[str, list[RetrievedChunk]]:
        max_chunks = min(max_chunks, self._max_doc_chunks)
        self.log.debug(
            "AgenticRAGPipeline tool=get_document citation=%r max_chunks=%d",
            citation, max_chunks,
        )
        raw = self._doc_chunks.get(citation.strip(), [])
        if not raw:
            return f"No document found with citation '{citation}'.", []
        chunks = [
            RetrievedChunk(
                text=c.text, doc_id=c.doc_id, chunk_idx=c.chunk_idx,
                metadata=c.metadata, embedding=c.embedding, score=0.0,
            )
            for c in raw[:max_chunks]
        ]
        return _format_chunks(chunks, show_score=False), chunks

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

        The LLM is given the query and two tools.  It issues tool calls until
        it decides it has enough context, or until max_iterations is reached.
        All chunks retrieved across iterations are deduplicated and returned
        as retrieved_chunks (used by the evaluation harness for recall/MRR).
        """
        from google.genai import types

        self._load_client()
        accumulated: list[RetrievedChunk] = []

        # Build Gemini tool declarations
        tools = [types.Tool(function_declarations=[
            types.FunctionDeclaration(
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
                            description="Keyword search query — legal terms, case names, concepts.",
                        ),
                        "k": types.Schema(
                            type=types.Type.INTEGER,
                            description=f"Number of passages to return (max {self._max_k_per_search}).",
                        ),
                    },
                    required=["query"],
                ),
            ),
            types.FunctionDeclaration(
                name="get_document",
                description=(
                    "Retrieve text passages from a specific legal case by its citation. "
                    "Use this after keyword_search has identified a promising case."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "citation": types.Schema(
                            type=types.Type.STRING,
                            description="Legal citation, e.g. '2022 BCSC 100'.",
                        ),
                        "max_chunks": types.Schema(
                            type=types.Type.INTEGER,
                            description=f"Maximum passages to return (max {self._max_doc_chunks}).",
                        ),
                    },
                    required=["citation"],
                ),
            ),
        ])]

        gen_cfg = types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            tools=tools,
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO")
            ),
            temperature=0.0,
        )

        contents: list[types.Content] = [
            types.Content(role="user", parts=[types.Part(text=query_text)])
        ]

        final_text = ""
        n_iterations = 0

        for n_iterations in range(1, self._max_iterations + 1):
            response = self._client.models.generate_content(  # type: ignore[union-attr]
                model=self._model_name,
                contents=contents,
                config=gen_cfg,
            )
            usage = response.usage_metadata
            self._track_and_log(usage.prompt_token_count, usage.candidates_token_count)

            candidate = response.candidates[0].content
            function_calls = [p.function_call for p in candidate.parts if p.function_call]
            text_parts = [p.text for p in candidate.parts if p.text]

            if not function_calls:
                # LLM decided it has enough context — final answer
                final_text = "".join(text_parts)
                break

            # Append model turn and execute tools
            contents.append(candidate)
            tool_response_parts: list[types.Part] = []

            for fc in function_calls:
                if fc.name == "keyword_search":
                    result_text, chunks = self._keyword_search(
                        query=str(fc.args.get("query", "")),
                        k=int(fc.args.get("k", self._max_k_per_search)),
                    )
                elif fc.name == "get_document":
                    result_text, chunks = self._get_document(
                        citation=str(fc.args.get("citation", "")),
                        max_chunks=int(fc.args.get("max_chunks", self._max_doc_chunks)),
                    )
                else:
                    result_text, chunks = f"Unknown tool: {fc.name}", []
                    self.log.warning("AgenticRAGPipeline: unknown tool call '%s'", fc.name)

                accumulated.extend(chunks)
                tool_response_parts.append(types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response={"result": result_text},
                    )
                ))
                self.log.debug(
                    "AgenticRAGPipeline iteration=%d tool=%s chunks_returned=%d total_accumulated=%d",
                    n_iterations, fc.name, len(chunks), len(accumulated),
                )

            contents.append(types.Content(role="user", parts=tool_response_parts))

        # If max_iterations exhausted without a text answer, force one
        if not final_text:
            self.log.info(
                "AgenticRAGPipeline: max_iterations=%d reached — forcing final answer",
                self._max_iterations,
            )
            contents.append(types.Content(
                role="user",
                parts=[types.Part(text=(
                    "Based on the search results above, please provide your final answer "
                    "to the original question."
                ))],
            ))
            response = self._client.models.generate_content(  # type: ignore[union-attr]
                model=self._model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    temperature=0.0,
                ),
            )
            usage = response.usage_metadata
            self._track_and_log(usage.prompt_token_count, usage.candidates_token_count)
            final_text = response.text

        # Deduplicate accumulated chunks by (doc_id, chunk_idx), preserving rank order
        seen: set[tuple[str, int]] = set()
        unique_chunks: list[RetrievedChunk] = []
        for chunk in accumulated:
            key = (chunk.doc_id, chunk.chunk_idx)
            if key not in seen:
                seen.add(key)
                unique_chunks.append(chunk)

        self.log.info(
            "AgenticRAGPipeline: complete | iterations=%d | unique_chunks=%d | "
            "total_accumulated=%d",
            n_iterations, len(unique_chunks), len(accumulated),
        )

        return QueryResult(
            query=query_text,
            retrieved_chunks=unique_chunks,
            answer=final_text or None,
            metadata={
                "iterations": n_iterations,
                "total_chunks_accumulated": len(accumulated),
            },
        )

    def batch_query(self, queries: list[str], k: int | None = None) -> list[QueryResult]:
        return [self.query(q, k=k) for q in queries]
