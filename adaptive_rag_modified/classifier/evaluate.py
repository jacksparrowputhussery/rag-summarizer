"""
classifier/evaluate.py
========================
Evaluates a trained classifier on data/labels/strategy_labels_test.jsonl.

Usage:
    python -m classifier.evaluate --config config.yaml
"""
from __future__ import annotations

import argparse
import json

from classifier.dataset import LABEL_NAMES, load_split
from classifier.model import TfidfLogRegClassifier
from utils import get_logger, load_config, resolve_path, write_jsonl

logger = get_logger(__name__)


def evaluate(cfg: dict, model=None) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    labels_dir = resolve_path("data/labels")
    ckpt_dir = resolve_path("checkpoints")
    test_q, test_y, _ = load_split(str(labels_dir / "strategy_labels_test.jsonl"))

    if model is None:
        backend = cfg["classifier"].get("backend", "tfidf_logreg")
        if backend == "tfidf_logreg":
            model = TfidfLogRegClassifier.load(str(ckpt_dir / "classifier_tfidf_logreg.pkl"))
        else:
            from classifier.model import TransformerClassifier

            model = TransformerClassifier.__new__(TransformerClassifier)
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            from utils import get_device

            path = str(ckpt_dir / "transformer")
            model.cfg = cfg
            model.device = get_device(cfg)
            model.tokenizer = AutoTokenizer.from_pretrained(path)
            model.model = AutoModelForSequenceClassification.from_pretrained(path).to(model.device)

    preds = model.predict(test_q)

    accuracy = accuracy_score(test_y, preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(test_y, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(test_y, preds, average="weighted", zero_division=0)
    per_class_p, per_class_r, per_class_f1, per_class_support = precision_recall_fscore_support(
        test_y, preds, average=None, labels=list(range(len(LABEL_NAMES))), zero_division=0
    )
    cm = confusion_matrix(test_y, preds, labels=list(range(len(LABEL_NAMES))))

    results = {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": {
            LABEL_NAMES[i]: {
                "precision": float(per_class_p[i]),
                "recall": float(per_class_r[i]),
                "f1": float(per_class_f1[i]),
                "support": int(per_class_support[i]),
            }
            for i in range(len(LABEL_NAMES))
        },
        "confusion_matrix": {"labels": LABEL_NAMES, "matrix": cm.tolist()},
    }

    results_dir = resolve_path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "classifier_test_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Classifier test accuracy=%.4f macro_f1=%.4f", accuracy, macro_f1)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    evaluate(cfg)


if __name__ == "__main__":
    main()
