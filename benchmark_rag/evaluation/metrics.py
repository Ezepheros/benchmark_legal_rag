"""
Retrieval evaluation metrics.

All functions take a list of retrieved doc_ids (ordered, highest rank first)
and a set of relevant doc_ids, and return a scalar float.

Supported metrics
-----------------
  recall_at_k       — fraction of relevant docs found in top-k
  precision_at_k    — fraction of top-k results that are relevant
  mrr               — mean reciprocal rank of the first hit
  ndcg_at_k         — normalised discounted cumulative gain
  hit_at_k          — 1 if any relevant doc appears in top-k
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Per-query metric computation
# ---------------------------------------------------------------------------


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / k


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return float(any(r in relevant for r in retrieved[:k]))


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for rank, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    def dcg(items: list[str]) -> float:
        score = 0.0
        for i, item in enumerate(items, start=1):
            if item in relevant:
                score += 1.0 / math.log2(i + 1)
        return score

    actual_dcg = dcg(retrieved[:k])
    ideal_items = list(relevant)[:k]
    ideal_dcg = dcg(ideal_items)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Aggregate over many queries
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    experiment_id: str
    num_queries: int
    k_values: list[int]
    # Each metric name → {k → mean value}
    scores: dict[str, dict[int, float]] = field(default_factory=dict)
    # LLM judge scores (optional)
    judge_scores: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"=== {self.experiment_id} | {self.num_queries} queries ==="]
        for metric, by_k in self.scores.items():
            for k, val in sorted(by_k.items()):
                lines.append(f"  {metric}@{k}: {val:.4f}")
        for metric, val in self.judge_scores.items():
            lines.append(f"  judge/{metric}: {val:.4f}")
        return "\n".join(lines)


def evaluate_retrieval(
    experiment_id: str,
    retrieved_lists: list[list[str]],   # one list per query, ordered by rank
    relevant_sets: list[set[str]],      # one set per query
    k_values: list[int] = (5, 25, 100),
    metric_names: list[str] | None = None,
) -> EvaluationResult:
    """
    Compute aggregate retrieval metrics over a query set.

    Parameters
    ----------
    retrieved_lists:
        Outer list = queries; inner list = retrieved doc_ids in rank order.
    relevant_sets:
        Corresponding sets of relevant doc_ids (ground truth).
    k_values:
        Cutoffs at which to evaluate recall, precision, hit, ndcg.
    metric_names:
        Subset of ["recall_at_k", "precision_at_k", "hit_at_k", "mrr", "ndcg_at_k"].
        Defaults to all.
    """
    if metric_names is None:
        metric_names = ["recall_at_k", "precision_at_k", "hit_at_k", "mrr", "ndcg_at_k"]

    n = len(retrieved_lists)
    assert n == len(relevant_sets), "retrieved_lists and relevant_sets must have the same length"

    scores: dict[str, dict[int, float]] = {}

    for metric in metric_names:
        if metric == "mrr":
            vals = [mrr(ret, rel) for ret, rel in zip(retrieved_lists, relevant_sets)]
            scores["mrr"] = {-1: sum(vals) / n}
        else:
            scores[metric] = {}
            for k in k_values:
                if metric == "recall_at_k":
                    fn = recall_at_k
                elif metric == "precision_at_k":
                    fn = precision_at_k
                elif metric == "hit_at_k":
                    fn = hit_at_k
                elif metric == "ndcg_at_k":
                    fn = ndcg_at_k
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                vals = [fn(ret, rel, k) for ret, rel in zip(retrieved_lists, relevant_sets)]
                scores[metric][k] = sum(vals) / n

    # Clean up mrr key (use 0 as a sentinel k)
    if "mrr" in scores:
        scores["mrr"] = {0: scores["mrr"][-1]}

    return EvaluationResult(
        experiment_id=experiment_id,
        num_queries=n,
        k_values=list(k_values),
        scores=scores,
    )
