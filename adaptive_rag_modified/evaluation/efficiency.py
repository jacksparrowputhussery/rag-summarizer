"""
evaluation/efficiency.py
==========================
Latency tracking helpers: per-question timing was already captured by each
retriever's `retrieve_timed`; this module aggregates it into summary stats
and normalizes latency for use inside the labeling utility function.
"""
from __future__ import annotations

import numpy as np


def summarize_latencies(latencies_ms: list[float]) -> dict:
    if not latencies_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    arr = np.asarray(latencies_ms)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(arr.max()),
    }


def normalize_latency(latency_ms: float, min_ms: float, max_ms: float) -> float:
    """Min-max normalize a latency value into [0, 1] for use in the utility
    function (higher = slower, so it should be subtracted, not added)."""
    if max_ms - min_ms < 1e-9:
        return 0.0
    return (latency_ms - min_ms) / (max_ms - min_ms)
