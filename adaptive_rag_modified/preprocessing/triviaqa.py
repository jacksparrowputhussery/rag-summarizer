"""
preprocessing/triviaqa.py
==========================
Loads TriviaQA (single-hop, "rc" / reading-comprehension config), using the
provided evidence documents (Wikipedia / web search results) and the
annotated answer aliases to locate a gold sentence.
"""
from __future__ import annotations

from typing import Optional

from preprocessing.base import SyntheticKB, make_unified_record, try_load_hf_dataset
from utils import deterministic_hash, get_logger, normalize_text, simple_sentence_split

logger = get_logger(__name__)

DATASET_NAME = "triviaqa"
_HF_SPLIT_MAP = {"train": "train", "validation": "validation", "test": "validation"}


def load(cfg: dict, split: str = "train", n: Optional[int] = None) -> list[dict]:
    hf_split = _HF_SPLIT_MAP.get(split, "train")
    ds = try_load_hf_dataset("trivia_qa", name="rc", split=hf_split)

    records = []
    if ds is not None:
        iterator = ds if n is None else ds.select(range(min(n, len(ds))))
        for ex in iterator:
            try:
                question = ex["question"]
                answer = normalize_text(ex["answer"]["value"])
                aliases = [normalize_text(a) for a in ex["answer"].get("aliases", [])] + [answer]

                entity_pages = ex.get("entity_pages", {}) or {}
                titles = entity_pages.get("title", [])
                texts = entity_pages.get("wiki_context", [])
                if not texts:
                    continue

                doc_text = normalize_text(texts[0])
                doc_title = titles[0] if titles else "untitled"
                doc_id = f"triviaqa_doc_{deterministic_hash(doc_text[:500])}"

                sentences = simple_sentence_split(doc_text)
                gold_sentences = []
                for i, s in enumerate(sentences):
                    if any(a and a.lower() in s.lower() for a in aliases):
                        gold_sentences = [{"doc_id": doc_id, "sentence_id": i, "text": s}]
                        break

                records.append(
                    make_unified_record(
                        id=f"triviaqa_{ex.get('question_id', deterministic_hash(question))}",
                        question=question,
                        answer=answer,
                        dataset=DATASET_NAME,
                        question_type="single-hop",
                        split=split,
                        gold_documents=[{"doc_id": doc_id, "title": doc_title, "text": doc_text}],
                        gold_sentences=gold_sentences,
                        num_hops=1,
                        metadata={"source": "huggingface:trivia_qa"},
                    )
                )
            except (KeyError, IndexError, TypeError):
                continue
        logger.info("triviaqa: loaded %d real examples for split=%s", len(records), split)
        return records if n is None else records[:n]

    n = n or 100
    kb = SyntheticKB(seed=hash(("triviaqa", split)) % (2**31))
    logger.info("triviaqa: using synthetic fallback (%d examples, split=%s)", n, split)
    return kb.sample_singlehop(n, DATASET_NAME, split)
