"""
evaluation/evidence_matching.py
=================================
The most important component of the pipeline: decides whether a retrieved
chunk counts as a "hit" against a piece of gold evidence.

Matching priority (spec sections 13-14):
  1. ID-based: gold (doc_id, sentence_id) falls within the chunk's
     (doc_id, sentence_ids) span -> HIT. Preferred whenever IDs are present.
  2. Text-based fallback (used when gold_sentences are unavailable and we
     only have gold_documents, or IDs don't line up cleanly):
       a. exact normalized text match
       b. token-overlap F1 above a threshold
       c. semantic (embedding cosine similarity) above a threshold
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from utils import normalize_for_matching, simple_word_tokenize


@dataclass
class MatchResult:
    hit: bool
    method: str  # "id" | "exact_text" | "token_overlap" | "semantic" | "none"
    score: float = 0.0


def _token_f1(a: str, b: str) -> float:
    ta, tb = simple_word_tokenize(a), simple_word_tokenize(b)
    if not ta or not tb:
        return 0.0
    set_a, set_b = set(ta), set(tb)
    overlap = len(set_a & set_b)
    if overlap == 0:
        return 0.0
    precision = overlap / len(set_a)
    recall = overlap / len(set_b)
    return 2 * precision * recall / (precision + recall)


def match_chunk_to_gold_sentence(
    chunk: dict,
    gold_sentence: dict,
    encoder=None,
    semantic_threshold: float = 0.75,
    token_f1_threshold: float = 0.5,
    emb_cache: dict | None = None,
) -> MatchResult:
    """chunk: {chunk_id, doc_id, text, sentence_ids}
    gold_sentence: {doc_id, sentence_id, text}
    """
    # 1) ID-based (preferred)
    if chunk.get("doc_id") == gold_sentence.get("doc_id") and "sentence_ids" in chunk:
        if gold_sentence.get("sentence_id") in chunk["sentence_ids"]:
            return MatchResult(hit=True, method="id", score=1.0)
        # same document but sentence id list unavailable/mismatched -> fall through to text methods

    # 2) exact normalized text containment
    norm_gold = normalize_for_matching(gold_sentence["text"])
    norm_chunk = normalize_for_matching(chunk["text"])
    if norm_gold and norm_gold in norm_chunk:
        return MatchResult(hit=True, method="exact_text", score=1.0)

    # 3) token-overlap F1
    f1 = _token_f1(gold_sentence["text"], chunk["text"])
    if f1 >= token_f1_threshold:
        return MatchResult(hit=True, method="token_overlap", score=f1)

    # 4) semantic similarity (only if an encoder is supplied)
    if encoder is not None:
        try:
            if emb_cache is not None:
                t_gold = gold_sentence["text"]
                t_chunk = chunk["text"]
                if t_gold not in emb_cache:
                    emb_cache[t_gold] = encoder.encode([t_gold], normalize_embeddings=True)[0]
                if t_chunk not in emb_cache:
                    emb_cache[t_chunk] = encoder.encode([t_chunk], normalize_embeddings=True)[0]
                sim = float(np.dot(emb_cache[t_gold], emb_cache[t_chunk]))
            else:
                vecs = encoder.encode([gold_sentence["text"], chunk["text"]], normalize_embeddings=True)
                sim = float(np.dot(vecs[0], vecs[1]))
            if sim >= semantic_threshold:
                return MatchResult(hit=True, method="semantic", score=sim)
            return MatchResult(hit=False, method="semantic", score=sim)
        except Exception:  # noqa: BLE001
            pass

    return MatchResult(hit=False, method="none", score=f1)


def match_chunk_to_gold_document(chunk: dict, gold_document: dict) -> MatchResult:
    """Fallback used when a question has only document-level gold evidence
    (no sentence-level annotation available)."""
    if chunk.get("doc_id") == gold_document.get("doc_id"):
        return MatchResult(hit=True, method="id", score=1.0)
    return MatchResult(hit=False, method="none", score=0.0)


def evaluate_chunks_against_gold(
    retrieved_chunks: list[dict],
    gold_sentences: list[dict],
    gold_documents: list[dict],
    encoder=None,
    semantic_threshold: float = 0.75,
    token_f1_threshold: float = 0.5,
    emb_cache: dict | None = None,
) -> dict:
    """For one question: returns, for every gold evidence item, whether it
    was covered by ANY retrieved chunk at each rank, plus the matching
    method used. Uses gold_sentences when available, else gold_documents.

    `encoder` (optional): anything exposing `.encode(list[str]) -> array`,
    e.g. the same dense encoder already built for semantic retrieval. When
    provided, it enables the semantic-similarity fallback (step 4 of the
    matching priority) for sentence pairs that don't match on ID, exact
    text, or token overlap -- important for real datasets where sentence
    splitting/normalization can drift from the official annotation.
    """
    use_sentences = len(gold_sentences) > 0
    gold_items = gold_sentences if use_sentences else gold_documents
    level = "sentence" if use_sentences else "document"

    gold_hits = []  # one entry per gold item: {"gold": ..., "hit": bool, "hit_rank": Optional[int], "method": ...}
    for gold in gold_items:
        hit = False
        hit_rank: Optional[int] = None
        method = "none"
        for rank, chunk in enumerate(retrieved_chunks, start=1):
            if use_sentences:
                res = match_chunk_to_gold_sentence(
                    chunk,
                    gold,
                    encoder=encoder,
                    semantic_threshold=semantic_threshold,
                    token_f1_threshold=token_f1_threshold,
                    emb_cache=emb_cache,
                )
            else:
                res = match_chunk_to_gold_document(chunk, gold)
            if res.hit:
                hit = True
                hit_rank = rank
                method = res.method
                break
        gold_hits.append({"gold": gold, "hit": hit, "hit_rank": hit_rank, "method": method})

    return {"level": level, "gold_hits": gold_hits}
