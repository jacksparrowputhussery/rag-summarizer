"""
preprocessing/squad.py
=======================
Loads SQuAD v1.1 (single-hop). Gold evidence = the sentence inside the
passage/context that contains the annotated answer span.

Falls back to the offline SyntheticKB if the `datasets` library / network
access to huggingface.co is unavailable.
"""
from __future__ import annotations

from typing import Optional

from preprocessing.base import (
    SyntheticKB,
    doc_to_sentences,
    make_unified_record,
    try_load_hf_dataset,
)
from utils import deterministic_hash, get_logger, normalize_text, simple_sentence_split

logger = get_logger(__name__)

DATASET_NAME = "squad"
_HF_SPLIT_MAP = {"train": "train", "validation": "validation", "test": "validation"}


def _locate_answer_sentence(context: str, answer_start: int, answer_text: str) -> Optional[int]:
    """Map a char-offset answer span to a sentence index within the context."""
    sentences = simple_sentence_split(context)
    cursor = 0
    for idx, sent in enumerate(sentences):
        # find sentence's approx span within the (whitespace-normalized) context
        start = context.find(sent, cursor)
        if start == -1:
            continue
        end = start + len(sent)
        if start <= answer_start < end or (answer_text and answer_text in sent):
            return idx
        cursor = end
    return None


def load(cfg: dict, split: str = "train", n: Optional[int] = None) -> list[dict]:
    hf_split = _HF_SPLIT_MAP.get(split, "train")
    ds = try_load_hf_dataset("squad", split=hf_split)

    records = []
    if ds is not None:
        seen_doc_ids = {}
        iterator = ds if n is None else ds.select(range(min(n, len(ds))))
        for ex in iterator:
            try:
                context = normalize_text(ex["context"])
                title = ex.get("title", "untitled")
                doc_id = seen_doc_ids.setdefault(context, f"squad_doc_{deterministic_hash(context)}")

                answers = ex["answers"]
                if not answers or not answers.get("text"):
                    continue  # malformed example -> drop
                answer_text = normalize_text(answers["text"][0])
                answer_start = answers["answer_start"][0]

                sent_idx = _locate_answer_sentence(context, answer_start, answer_text)
                gold_sentences = []
                if sent_idx is not None:
                    sents = simple_sentence_split(context)
                    gold_sentences = [{"doc_id": doc_id, "sentence_id": sent_idx, "text": sents[sent_idx]}]

                records.append(
                    make_unified_record(
                        id=f"squad_{ex['id']}",
                        question=ex["question"],
                        answer=answer_text,
                        dataset=DATASET_NAME,
                        question_type="single-hop",
                        split=split,
                        gold_documents=[{"doc_id": doc_id, "title": title, "text": context}],
                        gold_sentences=gold_sentences,  # may be empty if span could not be localized
                        num_hops=1,
                        metadata={"source": "huggingface:squad"},
                    )
                )
            except (KeyError, IndexError, TypeError):
                continue  # malformed example -> drop
        logger.info("squad: loaded %d real examples for split=%s", len(records), split)
        return records if n is None else records[:n]

    # ---- offline fallback -------------------------------------------------
    n = n or 100
    kb = SyntheticKB(seed=hash(("squad", split)) % (2**31))
    logger.info("squad: using synthetic fallback (%d examples, split=%s)", n, split)
    return kb.sample_singlehop(n, DATASET_NAME, split)
