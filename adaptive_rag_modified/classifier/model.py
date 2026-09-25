"""
classifier/model.py
=====================
Four-class strategy classifier: question text -> {BM25, Semantic, Hybrid,
Multi-hop}.

Two backends, selected via config `classifier.backend`:
  - "tfidf_logreg" (default): TF-IDF + multinomial logistic regression.
    Fully offline, trains in seconds, and is what actually runs in this
    sandbox. Good enough to validate the whole pipeline end-to-end.
  - "transformer": DeBERTa-v3-base (or any HF encoder) + linear head, trained
    with CrossEntropyLoss/AdamW as specified. Requires internet access to
    download model weights; used when `run.debug=false` and a suitable
    environment (GPU, HF Hub access) is available.

Both backends expose the same three methods: fit(), predict(), predict_proba().
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from utils import get_logger

logger = get_logger(__name__)


class TfidfLogRegClassifier:
    backend = "tfidf_logreg"

    def __init__(self, cfg: dict):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.cfg = cfg
        self.vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
        class_weight = "balanced" if cfg["classifier"].get("class_weighting", True) else None
        self.clf = LogisticRegression(max_iter=2000, class_weight=class_weight)

    def fit(self, questions: list[str], labels: list[int]):
        X = self.vectorizer.fit_transform(questions)
        unique_labels = sorted(set(labels))
        if len(unique_labels) < 2:
            # Degenerate label distribution (e.g. a tiny/synthetic smoke-test
            # corpus where one strategy trivially dominates every question).
            # A real classifier can't be fit on a single class, so fall back
            # to a constant predictor that always outputs that one class,
            # and log a clear warning instead of crashing the pipeline.
            logger.warning(
                "Training labels contain only one class (%s). Falling back to a "
                "constant predictor. This typically means the corpus/questions "
                "are too easy/small to differentiate retrieval strategies -- "
                "expected in a tiny DEBUG/synthetic run, but should NOT happen "
                "on a real, full-scale dataset run.",
                unique_labels[0],
            )
            self._constant_label = unique_labels[0]
            self._is_constant = True
            return self
        self._is_constant = False
        self.clf.fit(X, labels)
        return self

    def predict(self, questions: list[str]) -> np.ndarray:
        if getattr(self, "_is_constant", False):
            return np.full(len(questions), self._constant_label, dtype=int)
        X = self.vectorizer.transform(questions)
        return self.clf.predict(X)

    def predict_proba(self, questions: list[str]) -> np.ndarray:
        if getattr(self, "_is_constant", False):
            num_classes = self.cfg["classifier"]["num_classes"]
            proba = np.zeros((len(questions), num_classes))
            proba[:, self._constant_label] = 1.0
            return proba
        X = self.vectorizer.transform(questions)
        return self.clf.predict_proba(X)

    def save(self, path: str):
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "TfidfLogRegClassifier":
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)


class TransformerClassifier:
    """DeBERTa-v3-base (default) four-class classifier. Only instantiated
    when classifier.backend == 'transformer' AND torch/transformers +
    network access to the model hub are available; see classifier/train.py
    for the graceful fallback logic."""

    backend = "transformer"

    def __init__(self, cfg: dict):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from utils import get_device

        self.cfg = cfg
        model_name = cfg["classifier"]["model_name"]
        num_classes = cfg["classifier"]["num_classes"]
        self.device = get_device(cfg)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes).to(self.device)

    def _encode(self, questions: list[str]):
        return self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=self.cfg["classifier"]["max_length"],
            return_tensors="pt",
        ).to(self.device)

    def predict(self, questions: list[str]) -> np.ndarray:
        import torch

        self.model.eval()
        with torch.no_grad():
            preds = []
            batch_size = 128
            for i in range(0, len(questions), batch_size):
                batch = self._encode(questions[i : i + batch_size])
                logits = self.model(**batch).logits
                preds.append(logits.argmax(dim=-1).cpu().numpy())
            return np.concatenate(preds) if preds else np.array([], dtype=int)

    def predict_proba(self, questions: list[str]) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        self.model.eval()
        with torch.no_grad():
            probas = []
            batch_size = 128
            for i in range(0, len(questions), batch_size):
                batch = self._encode(questions[i : i + batch_size])
                logits = self.model(**batch).logits
                probas.append(F.softmax(logits, dim=-1).cpu().numpy())
            return np.concatenate(probas, axis=0) if probas else np.empty((0, self.cfg["classifier"]["num_classes"]))

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


def build_classifier(cfg: dict):
    backend = cfg["classifier"].get("backend", "tfidf_logreg")
    if backend == "transformer":
        try:
            return TransformerClassifier(cfg)
        except Exception as e:  # noqa: BLE001
            logger.warning("Transformer backend unavailable (%s). Falling back to tfidf_logreg.", e)
            return TfidfLogRegClassifier(cfg)
    return TfidfLogRegClassifier(cfg)
