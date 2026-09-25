"""
retrieval/hybrid.py
======================
Combines BM25 and dense semantic retrieval:

    BM25 top-K_c  +  Dense top-K_c
            -> candidate union
            -> min-max score normalization (per system)
            -> hybrid_score = alpha * norm_bm25 + (1-alpha) * norm_semantic
            -> top-K
"""
from __future__ import annotations

from retrieval.base import BaseRetriever
from retrieval.bm25 import BM25Retriever
from retrieval.semantic import SemanticRetriever


def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class HybridRetriever(BaseRetriever):
    name = "hybrid"

    def __init__(self, chunks: list[dict], cfg: dict, bm25: BM25Retriever, semantic: SemanticRetriever):
        self.chunks = chunks
        self.cfg = cfg
        self.bm25 = bm25
        self.semantic = semantic
        self.chunk_lookup = {c["chunk_id"]: c for c in chunks}

    def _combine(self, bm25_results: list[dict], sem_results: list[dict], top_k: int, alpha: float) -> list[dict]:
        bm25_scores = {r["chunk_id"]: r["score"] for r in bm25_results}
        sem_scores = {r["chunk_id"]: r["score"] for r in sem_results}

        norm_bm25 = _min_max_normalize(bm25_scores)
        norm_sem = _min_max_normalize(sem_scores)

        candidate_ids = set(bm25_scores) | set(sem_scores)
        hybrid_scores = {}
        for cid in candidate_ids:
            b = norm_bm25.get(cid, 0.0)
            s = norm_sem.get(cid, 0.0)
            hybrid_scores[cid] = alpha * b + (1 - alpha) * s

        ranked = sorted(hybrid_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

        out = []
        for cid, score in ranked:
            chunk = self.chunk_lookup[cid]
            out.append({"chunk_id": cid, "doc_id": chunk["doc_id"], "text": chunk["text"], "score": float(score)})
        return out

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        hcfg = self.cfg["hybrid"]
        candidate_k = hcfg["candidate_k"]
        alpha = hcfg["alpha"]

        bm25_results = self.bm25.retrieve(query, candidate_k)
        sem_results = self.semantic.retrieve(query, candidate_k)
        return self._combine(bm25_results, sem_results, top_k, alpha)

    def retrieve_batch(self, queries: list[str], top_k: int) -> list[list[dict]]:
        """Batches the expensive semantic half (one encoder call for all
        queries) and the BM25 half (one call per query -- cheap, CPU-only,
        no GPU benefit to batching), then combines per-query. This is the
        main throughput win for hybrid/multihop during large-scale label
        generation, since hybrid is called on every hop of every multi-hop
        question."""
        if not queries:
            return []
        hcfg = self.cfg["hybrid"]
        candidate_k = hcfg["candidate_k"]
        alpha = hcfg["alpha"]

        bm25_results_batch = self.bm25.retrieve_batch(queries, candidate_k)
        sem_results_batch = self.semantic.retrieve_batch(queries, candidate_k)

        return [
            self._combine(bm25_res, sem_res, top_k, alpha)
            for bm25_res, sem_res in zip(bm25_results_batch, sem_results_batch)
        ]
