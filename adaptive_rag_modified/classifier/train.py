"""
classifier/train.py
=====================
Trains the four-class strategy classifier on data/labels/strategy_labels_train.jsonl,
validating on strategy_labels_val.jsonl.

- Reports label distribution BEFORE any class-weighting/balancing is applied
  (spec section 25).
- tfidf_logreg backend: single closed-form-ish fit (LogisticRegression),
  class_weight="balanced" if classifier.class_weighting=true.
- transformer backend: CrossEntropyLoss + AdamW + linear LR schedule +
  early stopping on validation macro-F1, best checkpoint saved.

Usage:
    python -m classifier.train --config config.yaml
"""
from __future__ import annotations

import argparse
import json

from classifier.dataset import label_distribution, load_split
from classifier.model import TfidfLogRegClassifier, TransformerClassifier, build_classifier
from utils import get_logger, load_config, resolve_path, set_seed

logger = get_logger(__name__)


def _train_transformer(model: TransformerClassifier, train_q, train_y, val_q, val_y, cfg: dict, ckpt_dir):
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset
    from sklearn.metrics import f1_score

    class QADataset(Dataset):
        def __init__(self, questions, labels):
            self.questions, self.labels = questions, labels

        def __len__(self):
            return len(self.questions)

        def __getitem__(self, idx):
            return self.questions[idx], self.labels[idx]

    def collate(batch):
        qs, ys = zip(*batch)
        return list(qs), torch.tensor(ys, dtype=torch.long)

    ccfg = cfg["classifier"]
    train_loader = DataLoader(QADataset(train_q, train_y), batch_size=ccfg["batch_size"], shuffle=True, collate_fn=collate)

    class_weights = None
    if ccfg.get("class_weighting", True):
        import numpy as np

        counts = np.bincount(train_y, minlength=ccfg["num_classes"])
        counts = np.maximum(counts, 1)
        weights = counts.sum() / (len(counts) * counts)
        class_weights = torch.tensor(weights, dtype=torch.float32, device=model.device)

    optimizer = AdamW(model.model.parameters(), lr=ccfg["learning_rate"])
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_f1, patience_left = -1.0, ccfg["early_stopping_patience"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, ccfg["epochs"] + 1):
        model.model.train()
        total_loss = 0.0
        for questions, labels in train_loader:
            batch = model._encode(questions)
            labels = labels.to(model.device)
            optimizer.zero_grad()
            logits = model.model(**batch).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_preds = model.predict(val_q)
        val_f1 = f1_score(val_y, val_preds, average="macro", zero_division=0)
        logger.info("Epoch %d: train_loss=%.4f val_macro_f1=%.4f", epoch, total_loss / max(1, len(train_loader)), val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_left = ccfg["early_stopping_patience"]
            model.save(str(ckpt_dir))
            logger.info("New best checkpoint saved (val_macro_f1=%.4f)", best_f1)
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("Early stopping at epoch %d", epoch)
                break

    return best_f1


def train(cfg: dict):
    set_seed(cfg["run"]["seed"])
    labels_dir = resolve_path("data/labels")
    ckpt_dir = resolve_path("checkpoints")

    train_q, train_y, _ = load_split(str(labels_dir / "strategy_labels_train.jsonl"))
    val_q, val_y, _ = load_split(str(labels_dir / "strategy_labels_val.jsonl"))

    logger.info("Train label distribution (before balancing): %s", json.dumps(label_distribution(train_y), indent=2))
    logger.info("Val label distribution: %s", json.dumps(label_distribution(val_y), indent=2))

    model = build_classifier(cfg)

    if isinstance(model, TfidfLogRegClassifier):
        model.fit(train_q, train_y)
        model.save(str(ckpt_dir / "classifier_tfidf_logreg.pkl"))
        logger.info("Saved TF-IDF+LogReg classifier to checkpoints/classifier_tfidf_logreg.pkl")
    else:
        _train_transformer(model, train_q, train_y, val_q, val_y, cfg, ckpt_dir / "transformer")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
