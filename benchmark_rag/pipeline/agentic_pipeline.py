"""
Agentic RAG pipeline: a Gemini LLM drives iterative keyword search over a
BM25 index, curating relevant documents before synthesising a final answer.

Each iteration has two phases plus an enforcement step:

  Search phase  — the agent chooses a keyword_search query.
  Review phase  — the agent chooses which citations to save.
  Enforcement   — if the agent saved fewer than min_docs_per_iter, the top
                  BM25 results from that search are auto-saved.

The loop runs until target_saved_docs documents are saved or max_iterations
is reached — the agent cannot call answer() early.

After citations are saved, a summarize call produces a short summary of each
newly saved document.  These summaries form the agent's memory for subsequent
iterations, keeping input tokens bounded.

When the loop exits, saved chunks are re-scored against the original query
via BM25 and returned in descending relevance order, consistent with the
dense and hybrid retrieval pipelines.

Usage
-----
    python scripts/run_benchmark.py \\
        --config configs/experiments/agentic_gemini_bm25_recursive_1024.yaml

Requires: GOOGLE_API_KEY or GEMINI_API_KEY environment variable.
Requires a pre-built BM25 index (run run_indexing.py first).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from benchmark_rag.components.base import EmbeddedChunk, RetrievedChunk
from benchmark_rag.components.generators.gemini import _generate_with_retry
from benchmark_rag.components.retrievers.bm25_retriever import BM25Retriever, _tokenize
from benchmark_rag.config.schemas import ExperimentConfig
from benchmark_rag.logging import get_logger
from benchmark_rag.pipeline.rag_pipeline import QueryResult
from benchmark_rag.prompts.agentic import (
    ANSWER_SYSTEM_PROMPT as _ANSWER_SYSTEM_PROMPT,
    REVIEW_SYSTEM_PROMPT as _REVIEW_SYSTEM_PROMPT,
    SEARCH_SYSTEM_PROMPT as _SEARCH_SYSTEM_PROMPT,
    SUMMARIZE_INSTRUCTION as _SUMMARIZE_INSTRUCTION,
    SUMMARIZE_SYSTEM_PROMPT as _SUMMARIZE_SYSTEM_PROMPT,
)

log = logging.getLogger(__name__)

_PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.3 / 1_000_000, 2.5 / 1_000_000),
    "gemini-2.5-pro":   (1.25 / 1_000_000, 10.0 / 1_000_000),
}


_DOC_PREFIX_CHARS = 10_000


@dataclass
class _SavedDoc:
    citation: str
    title: str
    search_query: str
    chunks: list[RetrievedChunk]
    doc_prefix: str = ""       # first 10k chars of the document (or full text if short)
    is_short_doc: bool = False # True when full doc text fits in the prefix
    summary: str = ""


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
    RAG pipeline where a Gemini LLM drives iterative BM25 search and document
    curation before synthesising a final answer.

    The agent searches and saves documents each iteration until
    ``target_saved_docs`` are collected or ``max_iterations`` is reached.
    If the agent saves fewer than ``min_docs_per_iter`` in an iteration,
    the top BM25 results are auto-saved to ensure progress.

    After the loop, all saved chunks are re-scored against the original query
    via BM25 and returned in descending score order.
    """

    def __init__(
        self,
        retriever: BM25Retriever,
        model_name: str = "gemini-2.5-flash",
        max_iterations: int = 5,
        max_k_per_search: int = 10,
        max_doc_chunks: int = 15,
        max_cost_usd: float | None = 15.0,
        target_saved_docs: int = 25,
        min_docs_per_iter: int = 4,
        api_key: str | None = None,
    ):
        self.retriever = retriever
        self._model_name = model_name
        self._max_iterations = max_iterations
        self._max_k_per_search = max_k_per_search
        self._max_doc_chunks = max_doc_chunks
        self._max_cost_usd = max_cost_usd
        self._target_saved_docs = target_saved_docs
        self._min_docs_per_iter = min_docs_per_iter
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

    def _track_and_log(self, in_tok: int | None, out_tok: int | None) -> None:
        in_tok = in_tok or 0
        out_tok = out_tok or 0
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

    def _budget_exceeded(self) -> bool:
        if self._max_cost_usd is None:
            return False
        return (self._total_cost or 0.0) >= self._max_cost_usd

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
    ) -> tuple[str, list[str]]:
        """
        Record citations as relevant, loading their chunks from the BM25 index.
        Returns (confirmation_message, list_of_newly_saved_citations).
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
            title = chunks[0].metadata.get("name", "") if chunks else ""

            # Build document prefix from ALL chunks (not just max_doc_chunks)
            full_text = "\n\n".join(c.text for c in raw)
            is_short = len(full_text) <= _DOC_PREFIX_CHARS
            doc_prefix = full_text if is_short else full_text[:_DOC_PREFIX_CHARS]

            saved_docs[citation] = _SavedDoc(
                citation=citation,
                title=title,
                search_query=search_query,
                chunks=chunks,
                doc_prefix=doc_prefix,
                is_short_doc=is_short,
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
        return " ".join(parts), saved_new

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    def _summarize_document(self, query_text: str, doc: _SavedDoc) -> str:
        """Generate a concise summary of a saved document and its query relevance."""
        from google.genai import types

        doc_text = self._format_doc_text(doc)
        prompt = (
            f"QUESTION: {query_text}\n\n"
            f"DOCUMENT [{doc.citation}] — {doc.title}\n\n"
            f"{doc_text}\n\n"
            f"{_SUMMARIZE_INSTRUCTION}"
        )

        response = _generate_with_retry(self._client,  # type: ignore[union-attr]
            model=self._model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SUMMARIZE_SYSTEM_PROMPT,
                temperature=0.0,
                max_output_tokens=1024,
            ),
        )
        usage = response.usage_metadata
        self._track_and_log(usage.prompt_token_count, usage.candidates_token_count)
        return response.text or ""

    # ------------------------------------------------------------------
    # State formatting helpers
    # ------------------------------------------------------------------

    def _format_state(
        self,
        saved_docs: dict[str, _SavedDoc],
        searches_run: list[str],
    ) -> str:
        """Compact research state shown to the agent each iteration."""
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
                lines.append(
                    f"\n[{doc.citation}] {doc.title}  "
                    f"(found via: \"{doc.search_query}\")"
                )
                if doc.summary:
                    lines.append(doc.summary)
                elif doc.chunks:
                    snippet = doc.chunks[0].text[:300].replace("\n", " ")
                    lines.append(f"  {snippet}...")
        else:
            lines.append("\nNo documents saved yet.")

        return "\n".join(lines)

    @staticmethod
    def _format_doc_text(doc: _SavedDoc) -> str:
        """Format a saved document for LLM context.

        Short docs (full text <= 10k chars): just the full text.
        Long docs: the 10k-char prefix followed by the saved chunks.
        """
        if doc.is_short_doc:
            return doc.doc_prefix
        parts = [
            f"--- Document overview (first {_DOC_PREFIX_CHARS:,d} chars) ---",
            doc.doc_prefix,
            "--- Relevant chunks ---",
            _format_chunks(doc.chunks, show_score=False),
        ]
        return "\n\n".join(parts)

    def _format_saved_for_answer(self, saved_docs: dict[str, _SavedDoc]) -> str:
        """Full document context for all saved documents, used in the final answer call."""
        parts: list[str] = []
        for doc in saved_docs.values():
            header = f"=== {doc.citation} — {doc.title} ==="
            body = self._format_doc_text(doc)
            parts.append(f"{header}\n{body}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Final ranking
    # ------------------------------------------------------------------

    def _rank_saved_chunks(
        self, query_text: str, saved_docs: dict[str, _SavedDoc],
    ) -> list[RetrievedChunk]:
        """Re-score all saved chunks by BM25 relevance to the original query."""
        all_scores = self.retriever._bm25.get_scores(_tokenize(query_text))

        chunk_to_idx: dict[tuple[str, int], int] = {}
        for i, c in enumerate(self.retriever._chunks):
            chunk_to_idx[(c.doc_id, c.chunk_idx)] = i

        ranked: list[RetrievedChunk] = []
        for doc in saved_docs.values():
            for chunk in doc.chunks:
                idx = chunk_to_idx.get((chunk.doc_id, chunk.chunk_idx))
                chunk.score = float(all_scores[idx]) if idx is not None else 0.0
                ranked.append(chunk)

        ranked.sort(key=lambda c: c.score, reverse=True)
        return ranked

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: ExperimentConfig) -> "AgenticRAGPipeline":
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
            max_cost_usd=cfg.agentic.max_cost_usd,
            target_saved_docs=cfg.agentic.target_saved_docs,
            min_docs_per_iter=cfg.agentic.min_docs_per_iter,
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, query_text: str, k: int | None = None) -> QueryResult:
        """
        Run the agentic search loop for one query.

        The agent searches and saves documents every iteration until
        target_saved_docs are collected or max_iterations is reached.
        If the agent saves fewer than min_docs_per_iter, the top BM25
        results from that search are auto-saved.

        After the loop, all saved chunks are re-scored against the original
        query via BM25 and returned in descending relevance order.
        """
        from google.genai import types

        self._load_client()

        target = self._target_saved_docs
        saved_docs: dict[str, _SavedDoc] = {}
        searches_run: list[str] = []

        # ---- Tool declarations (no answer tool) --------------------------
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
                "Saved documents will be summarized and shown in your research "
                "state for subsequent searches."
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

        search_tools = types.Tool(function_declarations=[keyword_search_decl])
        review_tools = types.Tool(function_declarations=[save_citations_decl])

        search_cfg = types.GenerateContentConfig(
            system_instruction=_SEARCH_SYSTEM_PROMPT,
            tools=[search_tools],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=["keyword_search"],
                )
            ),
            temperature=0.0,
        )
        review_cfg = types.GenerateContentConfig(
            system_instruction=_REVIEW_SYSTEM_PROMPT,
            tools=[review_tools],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=["save_citations"],
                )
            ),
            temperature=0.0,
        )

        n_iterations = 0

        for n_iterations in range(1, self._max_iterations + 1):

            if len(saved_docs) >= target:
                self.log.info(
                    "AgenticRAGPipeline: target %d docs reached (%d saved) "
                    "— stopping search loop",
                    target, len(saved_docs),
                )
                break

            if self._budget_exceeded():
                self.log.warning(
                    "AgenticRAGPipeline: budget cap $%.2f reached "
                    "(spent $%.6f) — stopping search loop",
                    self._max_cost_usd, self._total_cost or 0.0,
                )
                break

            # ---- Search phase (fresh prompt) -----------------------------
            state_text = self._format_state(saved_docs, searches_run)
            search_prompt = (
                f"Question: {query_text}\n\n"
                f"{state_text}\n\n"
                f"You need to find {target - len(saved_docs)} more document(s). "
                f"Choose your next search."
            )

            response = _generate_with_retry(self._client,  # type: ignore[union-attr]
                model=self._model_name,
                contents=search_prompt,
                config=search_cfg,
            )
            usage = response.usage_metadata
            self._track_and_log(usage.prompt_token_count, usage.candidates_token_count)

            candidate = response.candidates[0].content if response.candidates else None
            if candidate is None or not getattr(candidate, "parts", None):
                finish = getattr(response.candidates[0], "finish_reason", "UNKNOWN") if response.candidates else "NO_CANDIDATES"
                self.log.warning(
                    "AgenticRAGPipeline: empty response in search phase "
                    "(iteration=%d, finish_reason=%s) — skipping to next iteration",
                    n_iterations, finish,
                )
                continue
            fc = next((p.function_call for p in candidate.parts if p.function_call), None)
            if fc is None:
                self.log.warning(
                    "AgenticRAGPipeline: no function call in search phase "
                    "(iteration=%d) — skipping to next iteration",
                    n_iterations,
                )
                continue

            search_query = str(fc.args.get("query", ""))
            k_req = int(fc.args.get("k", self._max_k_per_search))
            result_text, search_chunks = self._keyword_search(search_query, k_req)
            searches_run.append(search_query)

            self.log.debug(
                "AgenticRAGPipeline iteration=%d search=%r", n_iterations, search_query
            )

            if not search_chunks:
                self.log.debug(
                    "AgenticRAGPipeline iteration=%d: no search results", n_iterations
                )
                continue

            if self._budget_exceeded():
                self.log.warning(
                    "AgenticRAGPipeline: budget cap reached after search — "
                    "stopping (iteration=%d)", n_iterations,
                )
                break

            # ---- Review phase (fresh prompt) -----------------------------
            review_prompt = (
                f"Question: {query_text}\n\n"
                f"SEARCH RESULTS for \"{search_query}\":\n{result_text}\n\n"
                f"{state_text}"
            )

            response = _generate_with_retry(self._client,  # type: ignore[union-attr]
                model=self._model_name,
                contents=review_prompt,
                config=review_cfg,
            )
            usage = response.usage_metadata
            self._track_and_log(usage.prompt_token_count, usage.candidates_token_count)

            candidate = response.candidates[0].content if response.candidates else None
            fc = None
            if candidate is not None and getattr(candidate, "parts", None):
                fc = next((p.function_call for p in candidate.parts if p.function_call), None)
            else:
                finish = getattr(response.candidates[0], "finish_reason", "UNKNOWN") if response.candidates else "NO_CANDIDATES"
                self.log.warning(
                    "AgenticRAGPipeline: empty response in review phase "
                    "(iteration=%d, finish_reason=%s) — falling through to auto-save",
                    n_iterations, finish,
                )

            newly_saved: list[str] = []
            if fc is not None:
                raw_citations: list[str] = list(fc.args.get("citations", []))
                confirm_text, newly_saved = self._save_citations(
                    raw_citations, search_query, saved_docs,
                )
                self.log.debug(
                    "AgenticRAGPipeline iteration=%d save_citations=%r — %s",
                    n_iterations, raw_citations, confirm_text,
                )

            # ---- Auto-save enforcement -----------------------------------
            room = target - len(saved_docs)
            needed = min(self._min_docs_per_iter, room) - len(newly_saved)
            if needed > 0:
                seen: set[str] = set()
                auto_candidates: list[str] = []
                for chunk in search_chunks:
                    if chunk.doc_id not in seen and chunk.doc_id not in saved_docs:
                        auto_candidates.append(chunk.doc_id)
                        seen.add(chunk.doc_id)
                if auto_candidates:
                    auto_msg, auto_saved = self._save_citations(
                        auto_candidates[:needed], search_query, saved_docs,
                    )
                    newly_saved.extend(auto_saved)
                    self.log.debug(
                        "AgenticRAGPipeline iteration=%d auto-saved %d doc(s): %s",
                        n_iterations, len(auto_saved), auto_msg,
                    )

            # ---- Summarize newly saved documents -------------------------
            for citation in newly_saved:
                doc = saved_docs[citation]
                if self._budget_exceeded():
                    self.log.warning(
                        "AgenticRAGPipeline: budget cap reached — skipping "
                        "summary for %s", citation,
                    )
                    break
                try:
                    doc.summary = self._summarize_document(query_text, doc)
                    self.log.debug(
                        "AgenticRAGPipeline iteration=%d summarized %s (%d chars)",
                        n_iterations, citation, len(doc.summary),
                    )
                except Exception:
                    n_chunks = len(doc.chunks)
                    total_chars = sum(len(c.text) for c in doc.chunks)
                    self.log.exception(
                        "AgenticRAGPipeline: failed to summarize %s "
                        "(title=%r, n_chunks=%d, total_chars=%d, query=%r) "
                        "— continuing with snippet fallback",
                        citation, doc.title, n_chunks, total_chars,
                        query_text[:200],
                    )

        self.log.info(
            "AgenticRAGPipeline: search loop done | iterations=%d | saved_docs=%d",
            n_iterations, len(saved_docs),
        )

        # ---- Final answer generation (no tools) --------------------------
        final_text = self._generate_final_answer(query_text, saved_docs)

        # ---- Re-score and sort by BM25 relevance to original query -------
        retrieved_chunks = self._rank_saved_chunks(query_text, saved_docs)

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

        response = _generate_with_retry(self._client,  # type: ignore[union-attr]
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
