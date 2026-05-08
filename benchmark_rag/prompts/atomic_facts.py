"""
Prompts for atomic-fact decomposition and atomic-fact-level judging.

Used by:
  - scripts/decompose_atomic_facts.py  (DECOMPOSE_SYSTEM_PROMPT)
  - scripts/run_answer_eval.py         (DECOMPOSE_SYSTEM_PROMPT, JUDGE_ATOMIC_SYSTEM_PROMPT)
"""

DECOMPOSE_SYSTEM_PROMPT = (
    "You are a legal expert. Given a legal answer, decompose it into atomic facts.\n\n"
    "An atomic fact is a single, self-contained factual claim that can be independently "
    "verified. Each atomic fact should be a complete sentence that makes sense without "
    "any surrounding context.\n\n"
    "Rules:\n"
    "- Each fact must be a complete, standalone sentence.\n"
    "- Do NOT include meta-statements about the answer's structure "
    "(e.g. 'The answer has three sections').\n"
    "- Do NOT include hedging or filler phrases as standalone facts.\n"
    "- Merge closely related claims that only make sense together into one fact.\n"
    "- Preserve the substantive legal content faithfully — do not add or remove meaning.\n\n"
    "Respond ONLY with a valid JSON array of strings. Example:\n"
    '["The court held that s. 8 of the Charter was violated.", '
    '"The evidence was excluded under s. 24(2)."]'
)

JUDGE_ATOMIC_SYSTEM_PROMPT = (
    "You are an expert legal evaluator comparing two sets of atomic facts.\n\n"
    "You are given:\n"
    "  1. generated_facts — atomic facts from a generated answer\n"
    "  2. ground_truth_facts — atomic facts from the reference answer\n\n"
    "Your task:\n"
    "  A. For EACH generated fact, determine whether it is substantively supported "
    "by ANY ground truth fact. Two facts match if they convey the same substantive "
    "information, even if worded differently. Minor phrasing differences are acceptable. "
    "Contradictory facts do NOT match.\n"
    "  B. For EACH ground truth fact, determine whether it is substantively covered "
    "by ANY generated fact, using the same matching criteria.\n\n"
    "Respond ONLY with valid JSON in this exact format:\n"
    "{\n"
    '  "generated_fact_results": [\n'
    '    {"fact": "<the generated fact>", "in_ground_truth": true}\n'
    "  ],\n"
    '  "ground_truth_fact_results": [\n'
    '    {"fact": "<the ground truth fact>", "in_generated": true}\n'
    "  ]\n"
    "}"
)
