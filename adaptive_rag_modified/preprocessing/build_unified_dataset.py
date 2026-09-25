"""
preprocessing/build_unified_dataset.py
========================================
Runs every per-dataset loader, then:
  - drops malformed examples (no question, no gold evidence at all)
  - deduplicates by (dataset, normalized question)
  - preserves official split when available, otherwise assigns one
  - writes data/processed/{dataset}.jsonl and data/processed/unified.jsonl

Usage:
    python -m preprocessing.build_unified_dataset --config config.yaml
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from preprocessing import hotpotqa, musique, natural_questions, squad, triviaqa, wiki2multi
from utils import get_logger, load_config, normalize_for_matching, read_jsonl_list, resolve_path, set_seed, write_jsonl

logger = get_logger(__name__)

LOADERS = {
    "squad": squad,
    "natural_questions": natural_questions,
    "triviaqa": triviaqa,
    "musique": musique,
    "hotpotqa": hotpotqa,
    "2wikimultihopqa": wiki2multi,
}


def _is_well_formed(rec: dict) -> bool:
    if not rec.get("question") or not rec.get("id"):
        return False
    if not rec.get("gold_sentences") and not rec.get("gold_documents"):
        return False  # spec: must have gold evidence explicit in annotations
    return True


def _dedupe(records: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for r in records:
        key = (r["dataset"], normalize_for_matching(r["question"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _assign_split(records: list[dict], split_ratios: dict, seed: int) -> list[dict]:
    """Deterministically partition each dataset into train/validation/test splits."""
    from collections import defaultdict

    by_dataset = defaultdict(list)
    for r in records:
        by_dataset[r["dataset"]].append(r)

    rng = random.Random(seed)
    out = []
    for dataset_name, ds_records in by_dataset.items():
        rng.shuffle(ds_records)
        n = len(ds_records)
        n_train = int(n * split_ratios["train"])
        n_val = int(n * split_ratios["val"])
        for i, r in enumerate(ds_records):
            if i < n_train:
                r["split"] = "train"
            elif i < n_train + n_val:
                r["split"] = "validation"
            else:
                r["split"] = "test"
            out.append(r)
    return out


def build(cfg: dict, per_dataset_n: dict | None = None) -> list[dict]:
    set_seed(cfg["run"]["seed"])
    use_datasets = cfg["datasets"]["use"]
    processed_dir = resolve_path(cfg["datasets"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    debug = cfg["run"].get("debug", True)

    if debug:
        # DEBUG mode: same small count for every dataset, for fast, uniform
        # smoke tests. Explicit `per_dataset_n` argument (if given) still wins.
        default_n = cfg["debug"]["questions_per_dataset"]
        resolved_n = {name: default_n for name in use_datasets}
    else:
        # FULL mode: read per-dataset counts from config.yaml
        # (datasets.per_dataset_n), falling back to
        # full_run.default_questions_per_dataset for any dataset not listed.
        configured = cfg["datasets"].get("per_dataset_n", {}) or {}
        fallback_n = cfg.get("full_run", {}).get("default_questions_per_dataset")
        resolved_n = {name: configured.get(name, fallback_n) for name in use_datasets}

    if per_dataset_n:
        resolved_n.update(per_dataset_n)  # explicit function argument always wins

    all_records = []
    for name in use_datasets:
        cached_file = processed_dir / f"{name}.jsonl"
        if cached_file.exists() and cached_file.stat().st_size > 0:
            existing = read_jsonl_list(cached_file)
            if existing and existing[0].get("metadata", {}).get("source") != "synthetic_fallback":
                logger.info("%s: found existing %d real processed records -> reusing", name, len(existing))
                all_records.extend(existing)
                continue

        loader = LOADERS[name]
        n_per_split = resolved_n.get(name)
        dataset_records = []
        for split in ("train", "validation", "test"):
            n = n_per_split if n_per_split is None else max(1, n_per_split // 3)
            recs = loader.load(cfg, split=split, n=n)
            dataset_records.extend(recs)

        before = len(dataset_records)
        dataset_records = [r for r in dataset_records if _is_well_formed(r)]
        dataset_records = _dedupe(dataset_records)
        after = len(dataset_records)
        logger.info("%s: %d -> %d after cleaning/dedup", name, before, after)

        write_jsonl(processed_dir / f"{name}.jsonl", dataset_records)
        all_records.extend(dataset_records)

    all_records = _assign_split(all_records, cfg["splits"], cfg["run"]["seed"])
    write_jsonl(processed_dir / "unified.jsonl", all_records)
    logger.info("Unified dataset: %d total examples across %d datasets", len(all_records), len(use_datasets))
    return all_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    build(cfg)


if __name__ == "__main__":
    main()
