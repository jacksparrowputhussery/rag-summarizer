"""
classifier/dataset.py
=======================
Loads the strategy-label JSONL files. Classifier input is STRICTLY the
question text (spec section 22) — dataset name, question_type, and gold
evidence are never exposed to the classifier, even though they exist in the
label file for analysis purposes.
"""
from __future__ import annotations

from collections import Counter

from utils import read_jsonl_list

LABEL_NAMES = ["bm25", "semantic", "hybrid", "multihop"]


def load_split(path: str) -> tuple[list[str], list[int], list[dict]]:
    records = read_jsonl_list(path)
    questions = [r["question"] for r in records]
    labels = [r["label"] for r in records]
    return questions, labels, records


def label_distribution(labels: list[int]) -> dict:
    counts = Counter(labels)
    total = len(labels)
    return {
        LABEL_NAMES[i]: {"count": counts.get(i, 0), "fraction": counts.get(i, 0) / total if total else 0.0}
        for i in range(len(LABEL_NAMES))
    }
