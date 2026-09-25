"""
evaluation/retrieval_metrics.py
=================================
Computes all retrieval-quality metrics from the output of
`evidence_matching.evaluate_chunks_against_gold`, for a single question and
aggregated across questions.

Primary metric: Evidence Recall@K (spec section 15-16).
"""
from __future__ import annotations

import numpy as np


def evidence_recall_at_k(gold_hits: list[dict], k: int) -> float:
    """Fraction of gold evidence items retrieved within top-k."""
    if not gold_hits:
        return 0.0
    hits = sum(1 for gh in gold_hits if gh["hit"] and gh["hit_rank"] is not None and gh["hit_rank"] <= k)
    return hits / len(gold_hits)


def precision_at_k(gold_hits: list[dict], k: int, num_retrieved: int) -> float:
    """Fraction of the top-k retrieved chunks that matched SOME gold item.
    (Chunk-level precision: how many of the top-k slots were "useful".)"""
    if num_retrieved == 0 or k == 0:
        return 0.0
    matched_ranks = {gh["hit_rank"] for gh in gold_hits if gh["hit"] and gh["hit_rank"] is not None and gh["hit_rank"] <= k}
    denom = min(k, num_retrieved)
    return len(matched_ranks) / denom if denom > 0 else 0.0


def f1_at_k(gold_hits: list[dict], k: int, num_retrieved: int) -> float:
    r = evidence_recall_at_k(gold_hits, k)
    p = precision_at_k(gold_hits, k, num_retrieved)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def mrr(gold_hits: list[dict]) -> float:
    """Mean reciprocal rank of the FIRST gold item found (question-level MRR)."""
    ranks = [gh["hit_rank"] for gh in gold_hits if gh["hit"] and gh["hit_rank"] is not None]
    if not ranks:
        return 0.0
    return 1.0 / min(ranks)


def hit_at_k(gold_hits: list[dict], k: int) -> float:
    """1.0 if AT LEAST ONE gold item was retrieved within top-k, else 0.0."""
    return 1.0 if any(gh["hit"] and gh["hit_rank"] is not None and gh["hit_rank"] <= k for gh in gold_hits) else 0.0


def complete_evidence_retrieval(gold_hits: list[dict], k: int) -> float:
    """1.0 if ALL gold items were retrieved within top-k, else 0.0 (CER@K)."""
    if not gold_hits:
        return 0.0
    return 1.0 if all(gh["hit"] and gh["hit_rank"] is not None and gh["hit_rank"] <= k for gh in gold_hits) else 0.0


def evidence_coverage(gold_hits: list[dict], k: int) -> float:
    """Same computation as evidence_recall_at_k; kept as an explicit alias
    since the spec treats "coverage" as a named, reported quantity
    (esp. for multi-hop analysis)."""
    return evidence_recall_at_k(gold_hits, k)


def compute_question_metrics(gold_hits: list[dict], num_retrieved: int, ks: list[int]) -> dict:
    metrics = {}
    for k in ks:
        metrics[f"recall@{k}"] = evidence_recall_at_k(gold_hits, k)
        metrics[f"precision@{k}"] = precision_at_k(gold_hits, k, num_retrieved)
        metrics[f"f1@{k}"] = f1_at_k(gold_hits, k, num_retrieved)
        metrics[f"hit@{k}"] = hit_at_k(gold_hits, k)
        metrics[f"cer@{k}"] = complete_evidence_retrieval(gold_hits, k)
        metrics[f"evidence_coverage@{k}"] = evidence_coverage(gold_hits, k)
    metrics["mrr"] = mrr(gold_hits)
    return metrics


def aggregate_metrics(list_of_question_metrics: list[dict]) -> dict:
    if not list_of_question_metrics:
        return {}
    keys = list_of_question_metrics[0].keys()
    return {k: float(np.mean([m[k] for m in list_of_question_metrics])) for k in keys}
