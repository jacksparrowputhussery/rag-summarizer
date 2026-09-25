"""
preprocessing/hotpotqa.py
===========================
Loads HotpotQA (distractor setting, multi-hop). `supporting_facts` gives the
exact (title, sentence_id) gold evidence pairs, and `context` gives, for each
title, the list of sentences -- so gold sentences are taken directly from
official annotations (never invented from the answer).
"""
from __future__ import annotations

from typing import Optional

from preprocessing.base import SyntheticKB, make_unified_record, try_load_hf_dataset
from utils import deterministic_hash, get_logger, normalize_text

logger = get_logger(__name__)

DATASET_NAME = "hotpotqa"
_HF_SPLIT_MAP = {"train": "train", "validation": "validation", "test": "validation"}


def load(cfg: dict, split: str = "train", n: Optional[int] = None) -> list[dict]:
    hf_split = _HF_SPLIT_MAP.get(split, "train")
    ds = try_load_hf_dataset("hotpot_qa", name="distractor", split=hf_split)

    records = []
    if ds is not None:
        iterator = ds if n is None else ds.select(range(min(n, len(ds))))
        for ex in iterator:
            try:
                question = ex["question"]
                answer = normalize_text(ex["answer"])

                context_titles = ex["context"]["title"]
                context_sents = ex["context"]["sentences"]  # list[list[str]]
                title_to_idx = {t: i for i, t in enumerate(context_titles)}

                gold_documents, doc_ids = [], {}
                for title, sents in zip(context_titles, context_sents):
                    doc_text = normalize_text(" ".join(sents))
                    doc_id = f"hotpotqa_doc_{deterministic_hash(title + doc_text[:50])}"
                    doc_ids[title] = doc_id
                    gold_documents.append({"doc_id": doc_id, "title": title, "text": doc_text})

                sup_titles = ex["supporting_facts"]["title"]
                sup_sent_ids = ex["supporting_facts"]["sent_id"]
                gold_sentences = []
                for title, sent_id in zip(sup_titles, sup_sent_ids):
                    if title not in title_to_idx:
                        continue
                    sents = context_sents[title_to_idx[title]]
                    if sent_id >= len(sents):
                        continue
                    gold_sentences.append(
                        {"doc_id": doc_ids[title], "sentence_id": sent_id, "text": normalize_text(sents[sent_id])}
                    )

                if not gold_sentences:
                    continue  # malformed / unmatched -> drop

                # Only keep the gold documents actually referenced by supporting facts
                referenced_doc_ids = {gs["doc_id"] for gs in gold_sentences}
                gold_documents = [d for d in gold_documents if d["doc_id"] in referenced_doc_ids]

                records.append(
                    make_unified_record(
                        id=f"hotpotqa_{ex.get('id', deterministic_hash(question))}",
                        question=question,
                        answer=answer,
                        dataset=DATASET_NAME,
                        question_type="multi-hop",
                        split=split,
                        gold_documents=gold_documents,
                        gold_sentences=gold_sentences,
                        num_hops=len({d["doc_id"] for d in gold_documents}),
                        metadata={"source": "huggingface:hotpot_qa", "level": ex.get("level"), "type": ex.get("type")},
                    )
                )
            except (KeyError, IndexError, TypeError):
                continue
        logger.info("hotpotqa: loaded %d real examples for split=%s", len(records), split)
        return records if n is None else records[:n]

    n = n or 100
    kb = SyntheticKB(seed=hash(("hotpotqa", split)) % (2**31))
    logger.info("hotpotqa: using synthetic fallback (%d examples, split=%s)", n, split)
    return kb.sample_multihop(n, DATASET_NAME, split, hops=2)
