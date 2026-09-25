"""
retrieval/bm25.py
===================
Sparse lexical retrieval over the chunk corpus using Okapi BM25.
"""
from __future__ import annotations

import numpy as np

from corpus.index import BM25Index, build_or_load_bm25
from retrieval.base import BaseRetriever, format_result


class BM25Retriever(BaseRetriever):
    name = "bm25"

    def __init__(self, chunks: list[dict], cfg: dict, index: BM25Index | None = None):
        self.chunks = chunks
        self.cfg = cfg
        self.index = index or build_or_load_bm25(chunks, cfg)

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        scores = self.index.get_scores(query)
        top_k = min(top_k, len(self.chunks))
        top_idx = np.argpartition(-scores, top_k - 1)[:top_k] if top_k > 0 else np.array([], dtype=int)
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [format_result(self.chunks[i], scores[i]) for i in top_idx]
