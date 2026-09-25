"""
labeling/utility.py
=====================
Configurable utility function used (optionally) to rank retrieval
strategies for label generation:

    utility = alpha*EvidenceRecall + beta*F1 + gamma*MRR - lambda*norm_latency
"""
from __future__ import annotations


def compute_utility(metrics: dict, normalized_latency: float, cfg: dict) -> float:
    recall = metrics.get(f"recall@{_primary_k(cfg)}", metrics.get("recall@5", 0.0))
    f1 = metrics.get(f"f1@{_primary_k(cfg)}", metrics.get("f1@5", 0.0))
    mrr = metrics.get("mrr", 0.0)

    return (
        cfg["alpha_recall"] * recall
        + cfg["beta_f1"] * f1
        + cfg["gamma_mrr"] * mrr
        - cfg["lambda_latency"] * normalized_latency
    )


def _primary_k(cfg: dict) -> int:
    # Defaults to 5 if not explicitly configured elsewhere.
    return cfg.get("primary_k", 5)
