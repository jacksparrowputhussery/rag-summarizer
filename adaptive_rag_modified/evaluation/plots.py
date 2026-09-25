"""
evaluation/plots.py
=====================
Matplotlib visualizations saved under results/plots/:
  - confusion matrix heatmap
  - strategy comparison bar chart (Recall@5 / F1@5 / MRR / Hit@5)
  - per-dataset heatmap (dataset x strategy -> Evidence Recall@5)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import resolve_path


def _plots_dir() -> Path:
    d = resolve_path("results/plots")
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_confusion_matrix(matrix: list[list[int]], labels: list[str], title: str = "Strategy Classifier Confusion Matrix"):
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                     color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = _plots_dir() / "confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_strategy_comparison(method_metrics: dict[str, dict], metric_keys: list[str] = None):
    metric_keys = metric_keys or ["recall@5", "f1@5", "mrr", "hit@5"]
    methods = list(method_metrics.keys())
    x = np.arange(len(methods))
    width = 0.8 / len(metric_keys)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, mk in enumerate(metric_keys):
        vals = [method_metrics[m].get(mk, 0.0) for m in methods]
        ax.bar(x + i * width, vals, width, label=mk)

    ax.set_xticks(x + width * (len(metric_keys) - 1) / 2)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Retrieval Strategy Comparison")
    ax.legend()
    fig.tight_layout()
    out = _plots_dir() / "strategy_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_per_dataset_heatmap(dataset_x_strategy: dict[str, dict[str, float]], metric_name: str = "recall@5"):
    datasets = list(dataset_x_strategy.keys())
    strategies = list(next(iter(dataset_x_strategy.values())).keys())
    mat = np.array([[dataset_x_strategy[d].get(s, 0.0) for s in strategies] for d in datasets])

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(strategies)))
    ax.set_yticks(range(len(datasets)))
    ax.set_xticklabels(strategies, rotation=30, ha="right")
    ax.set_yticklabels(datasets)
    ax.set_title(f"Per-Dataset {metric_name}")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = _plots_dir() / "per_dataset_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
