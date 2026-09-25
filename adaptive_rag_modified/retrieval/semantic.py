"""
retrieval/semantic.py
=======================
Dense retrieval over the chunk corpus. Document embeddings are computed once
and cached (see corpus/index.py); only the query is embedded per-call (or
per-batch, via `retrieve_batch`).
"""
from __future__ import annotations

from corpus.index import DenseIndex, build_or_load_dense, encode_queries, encode_query
from retrieval.base import BaseRetriever, format_result


class SemanticRetriever(BaseRetriever):
    name = "semantic"

    def __init__(self, chunks: list[dict], cfg: dict, index: DenseIndex | None = None):
        self.chunks = chunks
        self.cfg = cfg
        self.index = index or build_or_load_dense(chunks, cfg)

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        qvec = encode_query(self.index.encoder, query)
        scores, idxs = self.index.search(qvec, top_k)
        return [format_result(self.chunks[i], s) for s, i in zip(scores, idxs)]

    def retrieve_batch(self, queries: list[str], top_k: int) -> list[list[dict]]:
        if not queries:
            return []
        batch_size = self.cfg.get("semantic", {}).get("batch_size", 32)
        qvecs = encode_queries(self.index.encoder, queries, batch_size=batch_size)
        per_query = self.index.search_batch(qvecs, top_k)
        return [
            [format_result(self.chunks[i], s) for s, i in zip(scores, idxs)]
            for scores, idxs in per_query
        ]
