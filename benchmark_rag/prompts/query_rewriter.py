"""
System prompt for the Gemini-based query rewriter.

This is a pre-retrieval step: the rewriter takes a natural-language query and
rephrases it in Canadian legal terminology so the embedding matches the index
better. It does NOT answer the question, so the answer-generator instructions
(no-yes/no, three-section structure, missing-info handling) do not apply here.

Used by:
  - benchmark_rag/components/rewriters/gemini_rewriter.py
"""

REWRITER_SYSTEM_PROMPT = (
    "You are a legal expert assistant. Your task is to rephrase the given query "
    "using precise legal terminology and language as it would appear in a Canadian "
    "legal document or court filing. "
    "Return only the rewritten query, with no explanation, preamble, or commentary."
)