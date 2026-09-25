"""
preprocessing/musique.py
==========================
Loads MuSiQue (multi-hop, 2-4 hops). Each example provides `paragraphs` with
an `is_supporting` flag -- the supporting paragraphs ARE the gold documents.
Since MuSiQue's official annotation is paragraph-level (not sentence-level),
we treat each supporting paragraph's first sentence as the anchor gold
sentence and keep the full paragraph as `gold_documents`, consistent with the
spec's fallback rule: "gold_sentences, or when sentence-level annotations are
unavailable, gold_documents".
"""
from __future__ import annotations

from typing import Optional

from preprocessing.base import SyntheticKB, doc_to_sentences, make_unified_record, try_load_hf_dataset
from utils import deterministic_hash, get_logger, normalize_text

logger = get_logger(__name__)

DATASET_NAME = "musique"
_HF_SPLIT_MAP = {"train": "train", "validation": "validation", "test": "validation"}


def load(cfg: dict, split: str = "train", n: Optional[int] = None) -> list[dict]:
    hf_split = _HF_SPLIT_MAP.get(split, "train")
    # MuSiQue is distributed under several community mirrors on the HF Hub;
    # try the most common one and fall back offline if unavailable.
    ds = try_load_hf_dataset("dgslibisey/MuSiQue", split=hf_split)

    records = []
    if ds is not None:
        iterator = ds if n is None else ds.select(range(min(n, len(ds))))
        for ex in iterator:
            try:
                question = ex["question"]
                answer = normalize_text(ex.get("answer", ""))
                paragraphs = ex["paragraphs"]

                gold_documents, gold_sentences = [], []
                for p in paragraphs:
                    if not p.get("is_supporting"):
                        continue
                    title = p.get("title", "untitled")
                    text = normalize_text(p.get("paragraph_text", ""))
                    if not text:
                        continue
                    doc_id = f"musique_doc_{deterministic_hash(title + text[:50])}"
                    gold_documents.append({"doc_id": doc_id, "title": title, "text": text})
                    sents = doc_to_sentences(doc_id, title, text)
                    if sents:
                        gold_sentences.append(sents[0])  # anchor sentence for this supporting paragraph

                if not gold_documents:
                    continue

                records.append(
                    make_unified_record(
                        id=f"musique_{ex.get('id', deterministic_hash(question))}",
                        question=question,
                        answer=answer,
                        dataset=DATASET_NAME,
                        question_type="multi-hop",
                        split=split,
                        gold_documents=gold_documents,
                        gold_sentences=gold_sentences,
                        num_hops=len(gold_documents),
                        metadata={"source": "huggingface:dgslibisey/MuSiQue"},
                    )
                )
            except (KeyError, IndexError, TypeError):
                continue
        logger.info("musique: loaded %d real examples for split=%s", len(records), split)
        return records if n is None else records[:n]

    n = n or 100
    kb = SyntheticKB(seed=hash(("musique", split)) % (2**31))
    logger.info("musique: using synthetic fallback (%d examples, split=%s)", n, split)
    return kb.sample_multihop(n, DATASET_NAME, split, hops=3)
