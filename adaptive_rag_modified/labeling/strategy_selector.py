"""
labeling/strategy_selector.py
================================
Determines the best retrieval strategy for a single question given each
strategy's evidence-based metrics (spec sections 20-21).

Two selectable methods:
  - "lexicographic" (default, recommended since evidence recall is the
    primary objective):
        1. highest Evidence Recall@5
        2. tie -> highest MRR
        3. tie -> highest F1@5
        4. tie -> lowest latency
  - "utility": single scalar score via labeling/utility.py, highest wins.

Never uses dataset identity, question_type, or gold evidence content itself
as a *feature* — only the already-computed retrieval metrics are used to
pick a label (spec section 22).
"""
from __future__ import annotations

from labeling.utility import compute_utility


def select_best_strategy(per_strategy_metrics: dict[str, dict], cfg: dict) -> tuple[str, dict]:
    """per_strategy_metrics: {"bm25": {"recall@5":..., "f1@5":..., "mrr":..., "latency_ms":...}, ...}
    Returns (best_strategy_name, all_scores_used_for_tie_breaking)."""
    method = cfg.get("method", "lexicographic")
    strategies = list(per_strategy_metrics.keys())

    if method == "utility":
        latencies = [per_strategy_metrics[s].get("latency_ms", 0.0) for s in strategies]
        lo, hi = min(latencies), max(latencies)
        scores = {}
        for s in strategies:
            lat = per_strategy_metrics[s].get("latency_ms", 0.0)
            norm_lat = 0.0 if hi - lo < 1e-9 else (lat - lo) / (hi - lo)
            scores[s] = compute_utility(per_strategy_metrics[s], norm_lat, cfg)
        best = max(scores, key=scores.get)
        return best, scores

    # ---- lexicographic (default) ----
    def sort_key(s: str):
        m = per_strategy_metrics[s]
        cer_score = max(m.get("cer@5", 0.0), m.get("cer@10", 0.0))
        recall_score = max(m.get("recall@5", 0.0), m.get("recall@10", 0.0))
        return (
            -cer_score,
            -recall_score,
            -m.get("mrr", 0.0),
            -m.get("f1@5", 0.0),
            m.get("latency_ms", 0.0),
        )

    ranked = sorted(strategies, key=sort_key)
    best = ranked[0]
    scores = {s: {"recall@5": per_strategy_metrics[s].get("recall@5", 0.0),
                  "cer@10": per_strategy_metrics[s].get("cer@10", 0.0),
                  "mrr": per_strategy_metrics[s].get("mrr", 0.0),
                  "f1@5": per_strategy_metrics[s].get("f1@5", 0.0),
                  "latency_ms": per_strategy_metrics[s].get("latency_ms", 0.0)} for s in strategies}
    return best, scores
