"""
preprocessing/wiki2multi.py
=============================
Loads 2WikiMultiHopQA (multi-hop). Schema mirrors HotpotQA: `context`
(title -> sentences) and `supporting_facts` (title, sent_id) pairs, plus an
`evidences` field with (subject, relation, object) triples in some releases
-- we only use the sentence-level supporting facts to avoid inventing
evidence from the answer.
"""
from __future__ import annotations

from typing import Optional

from preprocessing.base import SyntheticKB, make_unified_record, try_load_hf_dataset
from utils import deterministic_hash, get_logger, normalize_text

logger = get_logger(__name__)

DATASET_NAME = "2wikimultihopqa"
_HF_SPLIT_MAP = {"train": "train", "validation": "validation", "test": "validation"}
_RAW_CACHE: dict[str, list[dict]] = {}


def _load_raw_json(split: str) -> Optional[list[dict]]:
    filename = "train.json" if split == "train" else "dev.json"
    if filename in _RAW_CACHE:
        return _RAW_CACHE[filename]
    try:
        from huggingface_hub import hf_hub_download
        import json
        path = hf_hub_download(repo_id="voidful/2WikiMultihopQA", filename=filename, repo_type="dataset")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            _RAW_CACHE[filename] = data
            return data
    except Exception as e:
        logger.warning("Could not download/read 2Wiki raw json (%s): %s", split, e)
        return None


def load(cfg: dict, split: str = "train", n: Optional[int] = None) -> list[dict]:
    # voidful/2WikiMultihopQA fails with DatasetGenerationError when using HF datasets
    # because of schema inconsistencies in train.json. We load raw JSON directly via HF Hub.
    raw_items = _load_raw_json(split)
    if raw_items is None:
        hf_split = _HF_SPLIT_MAP.get(split, "train")
        ds = try_load_hf_dataset("voidful/2WikiMultihopQA", split=hf_split)
        if ds is not None:
            raw_items = ds if n is None else ds.select(range(min(n * 2 if n else len(ds), len(ds))))

    records = []
    if raw_items is not None:
        for ex in raw_items:
            try:
                question = ex["question"]
                answer = normalize_text(ex.get("answer", ""))

                context = ex["context"]  # list of [title, [sentences]] pairs
                titles = [c[0] for c in context]
                sents_per_doc = [c[1] for c in context]
                title_to_idx = {t: i for i, t in enumerate(titles)}

                gold_documents, doc_ids = [], {}
                for title, sents in zip(titles, sents_per_doc):
                    doc_text = normalize_text(" ".join(sents))
                    doc_id = f"2wiki_doc_{deterministic_hash(title + doc_text[:50])}"
                    doc_ids[title] = doc_id
                    gold_documents.append({"doc_id": doc_id, "title": title, "text": doc_text})

                sup = ex.get("supporting_facts", [])  # list of [title, sent_id]
                gold_sentences = []
                for title, sent_id in sup:
                    if title not in title_to_idx:
                        continue
                    sents = sents_per_doc[title_to_idx[title]]
                    if sent_id >= len(sents):
                        continue
                    gold_sentences.append(
                        {"doc_id": doc_ids[title], "sentence_id": sent_id, "text": normalize_text(sents[sent_id])}
                    )

                if not gold_sentences:
                    continue

                referenced_doc_ids = {gs["doc_id"] for gs in gold_sentences}
                gold_documents = [d for d in gold_documents if d["doc_id"] in referenced_doc_ids]

                records.append(
                    make_unified_record(
                        id=f"2wiki_{ex.get('_id', deterministic_hash(question))}",
                        question=question,
                        answer=answer,
                        dataset=DATASET_NAME,
                        question_type="multi-hop",
                        split=split,
                        gold_documents=gold_documents,
                        gold_sentences=gold_sentences,
                        num_hops=len({d["doc_id"] for d in gold_documents}),
                        metadata={"source": "huggingface:voidful/2WikiMultihopQA", "type": ex.get("type")},
                    )
                )
                if n is not None and len(records) >= n:
                    break
            except (KeyError, IndexError, TypeError):
                continue
        logger.info("2wikimultihopqa: loaded %d real examples for split=%s", len(records), split)
        return records

    n = n or 100
    kb = SyntheticKB(seed=hash(("2wiki", split)) % (2**31))
    logger.info("2wikimultihopqa: using synthetic fallback (%d examples, split=%s)", n, split)
    return kb.sample_multihop(n, DATASET_NAME, split, hops=2)
