"""
retrieval/base.py
===================
Every retriever implements the same interface and returns results in the
same standardized structure:

    [{"chunk_id": ..., "doc_id": ..., "text": ..., "score": ...}, ...]

CRITICAL (spec section 35 - "No Gold Evidence Leakage"): retrievers receive
ONLY the question text (and, for multi-hop, its own previously retrieved
evidence). They must never be passed gold_sentences / gold_documents /
answer / supporting_facts. This base class's signature intentionally only
accepts a plain string query to make that misuse structurally awkward.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    name: str = "base"

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[dict]:
        """query: plain question text only. Returns standardized chunk list."""
        raise NotImplementedError

    def retrieve_batch(self, queries: list[str], top_k: int) -> list[list[dict]]:
        """Batched retrieval over multiple questions at once. Default
        implementation just loops over `retrieve()` one query at a time --
        correct but slow. Retrievers backed by a GPU encoder (semantic,
        hybrid, multihop) override this with a real batched implementation
        since encoding queries one-at-a-time is the main GPU underutilization
        bottleneck during large-scale label generation / evaluation."""
        return [self.retrieve(q, top_k) for q in queries]

    def retrieve_timed(self, query: str, top_k: int) -> tuple[list[dict], float]:
        start = time.perf_counter()
        results = self.retrieve(query, top_k)
        elapsed = time.perf_counter() - start
        return results, elapsed

    def retrieve_batch_timed(self, queries: list[str], top_k: int) -> tuple[list[list[dict]], float]:
        """Returns (per_query_results, total_elapsed_seconds_for_the_whole_batch).
        Callers that need a per-question latency estimate should divide by
        len(queries) -- this is an approximation, since batched GPU work
        doesn't decompose cleanly into per-item timings, but it's accurate
        enough for aggregate latency reporting."""
        start = time.perf_counter()
        results = self.retrieve_batch(queries, top_k)
        elapsed = time.perf_counter() - start
        return results, elapsed


def format_result(chunk: dict, score: float) -> dict:
    return {
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk["doc_id"],
        "text": chunk["text"],
        "score": float(score),
    }
