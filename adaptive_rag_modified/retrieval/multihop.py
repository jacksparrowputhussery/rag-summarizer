"""
retrieval/multihop.py
========================
Genuine iterative multi-hop retrieval. At inference time the retriever has
NO access to gold_sentences / gold_documents / answer / supporting_facts —
it only ever sees the question text and its own previously retrieved chunks,
consistent with spec section 10 and section 35 ("No Gold Evidence Leakage").

Algorithm
---------
Hop 1: retrieve top_k_per_hop chunks with a hybrid retriever using the raw
       question as the query.
Hop t>1: build a refined query = original question + salient terms extracted
       from the chunks retrieved so far (that are not already well covered),
       then retrieve top_k_per_hop *new* chunks with that refined query.
Stop when max_hops is reached or a new hop adds no new chunks.

Final ranking: chunks are ordered by (hop_first_seen ascending, score
descending) and truncated to top_k.
"""
from __future__ import annotations

import re

from retrieval.base import BaseRetriever
from retrieval.hybrid import HybridRetriever
from utils import simple_word_tokenize

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at", "to",
    "and", "or", "for", "with", "by", "as", "that", "this", "which", "who",
    "whom", "what", "when", "where", "how", "why", "did", "does", "do", "be",
    "been", "being", "it", "its", "their", "his", "her", "he", "she", "they",
    "also", "known", "born",
}

_CAPITALIZED_PHRASE_RE = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\b")


def _extract_salient_terms(text: str, exclude_terms: set[str], max_terms: int = 4) -> list[str]:
    """Cheap, dependency-free entity-ish term extraction: capitalized phrases
    (proper-noun-like spans), falling back to rare content words."""
    candidates = _CAPITALIZED_PHRASE_RE.findall(text)
    candidates = [c for c in candidates if c.lower() not in _STOPWORDS and c.lower() not in exclude_terms]
    # de-dup, preserve order
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    if unique:
        return unique[:max_terms]

    # fallback: rare content words not in the question already
    tokens = [t for t in simple_word_tokenize(text) if t not in _STOPWORDS and len(t) > 3 and t not in exclude_terms]
    seen = set()
    unique = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique[:max_terms]


class MultiHopRetriever(BaseRetriever):
    name = "multihop"

    def __init__(self, chunks: list[dict], cfg: dict, hybrid: HybridRetriever):
        self.chunks = chunks
        self.cfg = cfg
        self.hybrid = hybrid  # each hop's retrieval uses the hybrid retriever internally

    def retrieve(self, query: str, top_k: int, max_hops: int | None = None) -> list[dict]:
        return self.retrieve_batch([query], top_k, max_hops=max_hops)[0]

    def retrieve_batch(self, queries: list[str], top_k: int, max_hops: int | None = None) -> list[list[dict]]:
        """Hop-synchronized batched multi-hop retrieval: at each hop, every
        still-active question's query is sent through `hybrid.retrieve_batch`
        in a single call, instead of looping questions one at a time and
        paying per-question encoder overhead 2-3x over (once per hop). Each
        question still independently decides, after its own hop-1 result,
        whether/how to refine its query and whether to keep going -- the
        batching only changes *when* the GPU work happens, not the per-
        question retrieval logic (same sequential algorithm as before)."""
        if not queries:
            return []
        mcfg = self.cfg["multihop"]
        max_hops = max_hops or mcfg["max_hops"]
        top_k_per_hop = mcfg["top_k_per_hop"]
        n = len(queries)

        seen_chunk_ids: list[set] = [set() for _ in range(n)]
        collected: list[list[tuple[int, dict]]] = [[] for _ in range(n)]
        current_query = list(queries)
        used_terms = [set(simple_word_tokenize(q)) for q in queries]
        active = [True] * n

        for hop in range(1, max_hops + 1):
            active_indices = [i for i in range(n) if active[i]]
            if not active_indices:
                break

            batch_queries = [current_query[i] for i in active_indices]
            hop_results_batch = self.hybrid.retrieve_batch(batch_queries, top_k_per_hop)

            for pos, i in enumerate(active_indices):
                hop_results = hop_results_batch[pos]
                new_results = [r for r in hop_results if r["chunk_id"] not in seen_chunk_ids[i]]

                if not new_results and hop > 1:
                    active[i] = False  # no new evidence found -> stop this question early
                    continue

                for r in new_results:
                    seen_chunk_ids[i].add(r["chunk_id"])
                    collected[i].append((hop, r))

                if hop == max_hops:
                    active[i] = False
                    continue

                evidence_text = " ".join(r["text"] for r in new_results) if new_results else " ".join(
                    r["text"] for _, r in collected[i]
                )
                salient = _extract_salient_terms(evidence_text, exclude_terms=used_terms[i])
                if not salient:
                    active[i] = False  # nothing new to chase -> stop this question early
                    continue
                used_terms[i].update(t.lower() for t in salient)
                current_query[i] = f"{queries[i]} {' '.join(salient)}"

        final_results = []
        for i in range(n):
            ranked = sorted(collected[i], key=lambda hr: (hr[0], -hr[1]["score"]))
            final_results.append([r for _, r in ranked][:top_k])
        return final_results
