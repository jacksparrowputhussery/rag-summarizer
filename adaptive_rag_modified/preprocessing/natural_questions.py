"""
preprocessing/natural_questions.py
===================================
Loads Natural Questions (single-hop, open-domain). Uses the short answer
annotation as gold evidence when present; otherwise uses the long answer
(paragraph) and falls back to synthetic data offline.

NQ is large and its HF schema (`google-research-datasets/natural_questions`,
config "default") nests tokens + byte offsets for long/short answers. We only
keep examples that have a resolvable short answer, since that is what maps
cleanly onto a single gold sentence.
"""
from __future__ import annotations

from typing import Optional

from preprocessing.base import SyntheticKB, make_unified_record, try_load_hf_dataset
from utils import deterministic_hash, get_logger, normalize_text, simple_sentence_split

logger = get_logger(__name__)

DATASET_NAME = "natural_questions"
_HF_SPLIT_MAP = {"train": "train", "validation": "validation", "test": "validation"}


def _extract_short_answer(example: dict) -> Optional[str]:
    try:
        annotations = example["annotations"]
        short_answers = annotations["short_answers"]
        for sa in short_answers:
            texts = sa.get("text") if isinstance(sa, dict) else None
            if texts:
                return normalize_text(texts[0])
    except (KeyError, IndexError, TypeError):
        pass
    return None


def load(cfg: dict, split: str = "train", n: Optional[int] = None) -> list[dict]:
    hf_split = _HF_SPLIT_MAP.get(split, "train")
    ds = try_load_hf_dataset("natural_questions", split=hf_split)

    records = []
    if ds is not None:
        iterator = ds if n is None else ds.select(range(min(n, len(ds))))
        for ex in iterator:
            try:
                question = ex["question"]["text"] if isinstance(ex.get("question"), dict) else ex.get("question_text")
                doc_title = ex.get("document", {}).get("title", "untitled")
                doc_text = normalize_text(ex.get("document", {}).get("html", "") or ex.get("document", {}).get("text", ""))
                if not doc_text:
                    continue
                answer = _extract_short_answer(ex)
                if not answer:
                    continue  # keep only resolvable short-answer examples

                doc_id = f"nq_doc_{deterministic_hash(doc_text[:500])}"
                sentences = simple_sentence_split(doc_text)
                gold_sentences = []
                for i, s in enumerate(sentences):
                    if answer.lower() in s.lower():
                        gold_sentences = [{"doc_id": doc_id, "sentence_id": i, "text": s}]
                        break

                records.append(
                    make_unified_record(
                        id=f"nq_{ex.get('id', deterministic_hash(question))}",
                        question=question,
                        answer=answer,
                        dataset=DATASET_NAME,
                        question_type="single-hop",
                        split=split,
                        gold_documents=[{"doc_id": doc_id, "title": doc_title, "text": doc_text}],
                        gold_sentences=gold_sentences,
                        num_hops=1,
                        metadata={"source": "huggingface:natural_questions"},
                    )
                )
            except (KeyError, IndexError, TypeError):
                continue
        logger.info("natural_questions: loaded %d real examples for split=%s", len(records), split)
        return records if n is None else records[:n]

    n = n or 100
    kb = SyntheticKB(seed=hash(("nq", split)) % (2**31))
    logger.info("natural_questions: using synthetic fallback (%d examples, split=%s)", n, split)
    return kb.sample_singlehop(n, DATASET_NAME, split)
