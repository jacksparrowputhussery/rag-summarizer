"""
evaluation/oracle.py
======================
The oracle strategy selector represents an upper bound: for each question it
picks whichever strategy achieved the best evidence-retrieval metrics
(computed using gold evidence). This is only used for evaluation/reporting —
never for inference-time retrieval — since it requires access to gold
evidence, violating the "no gold leakage at inference" rule by design and on
purpose (it's the ceiling the learned classifier is compared against).
"""
from __future__ import annotations

from labeling.strategy_selector import select_best_strategy


def oracle_metrics_for_question(per_strategy_metrics: dict[str, dict], selection_cfg: dict) -> tuple[str, dict]:
    """per_strategy_metrics: {"bm25": {...metrics...}, "semantic": {...}, ...}
    Returns (best_strategy_name, its_metrics)."""
    best_strategy, _ = select_best_strategy(per_strategy_metrics, selection_cfg)
    return best_strategy, per_strategy_metrics[best_strategy]


def compute_oracle_aggregate(all_questions_metrics: list[dict[str, dict]], selection_cfg: dict, ks: list[int]) -> dict:
    """all_questions_metrics: list of {"bm25": metrics, "semantic": metrics, ...} one per question."""
    from evaluation.retrieval_metrics import aggregate_metrics

    oracle_per_question = []
    strategy_choice_counts = {}
    for per_strategy in all_questions_metrics:
        best_strategy, best_metrics = oracle_metrics_for_question(per_strategy, selection_cfg)
        oracle_per_question.append(best_metrics)
        strategy_choice_counts[best_strategy] = strategy_choice_counts.get(best_strategy, 0) + 1

    return {
        "aggregate_metrics": aggregate_metrics(oracle_per_question),
        "oracle_strategy_distribution": strategy_choice_counts,
    }
