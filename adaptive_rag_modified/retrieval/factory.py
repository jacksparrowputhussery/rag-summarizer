"""
retrieval/factory.py
======================
Builds all four retrievers once, sharing the underlying BM25 / dense indices
so we don't recompute embeddings or tokenize the corpus four times.
"""
from __future__ import annotations

from corpus.index import build_or_load_bm25, build_or_load_dense
from retrieval.bm25 import BM25Retriever
from retrieval.hybrid import HybridRetriever
from retrieval.multihop import MultiHopRetriever
from retrieval.semantic import SemanticRetriever


def build_all_retrievers(chunks: list[dict], cfg: dict) -> dict[str, object]:
    bm25_index = build_or_load_bm25(chunks, cfg)
    dense_index = build_or_load_dense(chunks, cfg)

    bm25 = BM25Retriever(chunks, cfg, index=bm25_index)
    semantic = SemanticRetriever(chunks, cfg, index=dense_index)
    hybrid = HybridRetriever(chunks, cfg, bm25=bm25, semantic=semantic)
    multihop = MultiHopRetriever(chunks, cfg, hybrid=hybrid)

    return {
        "bm25": bm25,
        "semantic": semantic,
        "hybrid": hybrid,
        "multihop": multihop,
    }


STRATEGY_ID_TO_NAME = {0: "bm25", 1: "semantic", 2: "hybrid", 3: "multihop"}
STRATEGY_NAME_TO_ID = {v: k for k, v in STRATEGY_ID_TO_NAME.items()}
