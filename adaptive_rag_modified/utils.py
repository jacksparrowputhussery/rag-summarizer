"""
utils.py
========
Small, dependency-light helpers shared by every module in the project:
- config loading
- deterministic seeding
- JSONL IO
- text normalization (whitespace / unicode / punctuation)
- lightweight logging

Keeping these in one place avoids each module re-implementing normalization
slightly differently, which would silently break gold-evidence matching.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: str | Path = "config.yaml") -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(cfg_path: str) -> Path:
    """Resolve a path from config relative to the project root."""
    p = Path(cfg_path)
    return p if p.is_absolute() else PROJECT_ROOT / p


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(cfg: dict):
    """Resolves `run.device` ("auto" | "cpu" | "cuda") from config into an
    actual torch.device, honoring an explicit user choice instead of always
    silently auto-detecting. Returns None if torch isn't installed (callers
    that don't need torch, e.g. the tfidf_logreg classifier, never call this)."""
    try:
        import torch
    except ImportError:
        return None

    requested = (cfg.get("run", {}) or {}).get("device", "auto")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            get_logger(__name__).warning("run.device='cuda' requested but CUDA is not available; falling back to cpu.")
            return torch.device("cpu")
        return torch.device("cuda")
    # "auto" (default)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# --------------------------------------------------------------------------- #
# JSONL IO
# --------------------------------------------------------------------------- #
def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl_list(path: str | Path) -> list[dict]:
    return list(read_jsonl(path))


# --------------------------------------------------------------------------- #
# Text normalization
# --------------------------------------------------------------------------- #
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_text(text: str) -> str:
    """Full normalization pipeline used for preprocessing raw dataset text."""
    if text is None:
        return ""
    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    return text


def normalize_for_matching(text: str) -> str:
    """Aggressive normalization used only for evidence-matching comparisons:
    lowercase, strip punctuation, collapse whitespace, normalize unicode."""
    if text is None:
        return ""
    text = normalize_unicode(text)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def simple_word_tokenize(text: str) -> list[str]:
    return normalize_for_matching(text).split()


def simple_sentence_split(text: str) -> list[str]:
    """Lightweight, dependency-free sentence splitter (good enough for
    Wikipedia-style paragraphs used by these datasets). Falls back to NLTK
    punkt if available and already downloaded."""
    text = normalize_text(text)
    if not text:
        return []
    try:
        import nltk

        return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
    except Exception:
        pass
    # Fallback: split on sentence-ending punctuation followed by whitespace+capital
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    return [p.strip() for p in parts if p.strip()]


def deterministic_hash(text: str) -> str:
    import hashlib

    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def any_key(d: dict, keys: list[str], default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default
