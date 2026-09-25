"""
labeling/generate_labels.py
==============================
For every question in the unified dataset:
  1. Run all four retrievers using ONLY the question text (no gold leakage).
  2. Evaluate each strategy's retrieved chunks against gold evidence.
  3. Pick the best strategy (labeling/strategy_selector.py).
  4. Write a training example: {id, question, label, label_name, evidence_metrics}.

Outputs:
    data/labels/strategy_labels_{train,val,test}.jsonl
    cache/strategy_labels/*  (per-question full metrics, for later analysis)

Usage:
    python -m labeling.generate_labels --config config.yaml
"""
from __future__ import annotations

import argparse

from tqdm import tqdm

from evaluation.evidence_matching import evaluate_chunks_against_gold
from evaluation.retrieval_metrics import compute_question_metrics
from labeling.strategy_selector import select_best_strategy
from retrieval.factory import STRATEGY_NAME_TO_ID, build_all_retrievers
from utils import get_logger, load_config, read_jsonl_list, resolve_path, write_jsonl

logger = get_logger(__name__)

_SPLIT_FILE_MAP = {"train": "strategy_labels_train.jsonl", "validation": "strategy_labels_val.jsonl", "test": "strategy_labels_test.jsonl"}


def generate_labels_for_split(records: list[dict], retrievers: dict, cfg: dict) -> tuple[list[dict], list[dict[str, dict]]]:
    """Returns (label_records, per_question_full_metrics_for_oracle)."""
    ks = cfg["evaluation"]["ks"]
    top_k = cfg["labeling"]["top_k"]  # shared retrieval depth for label generation
    label_cfg = cfg["labeling"]
    batch_size = cfg.get("performance", {}).get("retrieval_batch_size", 32)

    # Reuse the already-built dense encoder for the semantic-similarity
    # fallback in evidence matching (step 4 of the matching priority), so we
    # don't load/build a second encoder just for this.
    match_encoder = retrievers["semantic"].index.encoder
    semantic_threshold = cfg["evaluation"]["semantic_match_threshold"]
    token_f1_threshold = cfg["evaluation"]["token_f1_match_threshold"]

    label_records = []
    full_metrics_for_oracle = []

    for batch_start in tqdm(range(0, len(records), batch_size), desc="Generating strategy labels (batched)"):
        batch = records[batch_start:batch_start + batch_size]
        batch_questions = [r["question"] for r in batch]  # ONLY question text is passed to retrievers

        # For each strategy, retrieve the WHOLE batch in one call -- this is
        # what keeps the GPU busy instead of one forward-pass per question.
        per_strategy_batch_results: dict[str, list[list[dict]]] = {}
        per_strategy_batch_latency_ms: dict[str, float] = {}
        for strategy_name, retriever in retrievers.items():
            results, elapsed_s = retriever.retrieve_batch_timed(batch_questions, top_k)
            per_strategy_batch_results[strategy_name] = results
            per_strategy_batch_latency_ms[strategy_name] = (elapsed_s * 1000.0) / max(1, len(batch))

        emb_cache = {}
        for idx_in_batch, rec in enumerate(batch):
            gold_sentences = rec.get("gold_sentences", [])
            gold_documents = rec.get("gold_documents", [])

            per_strategy_metrics = {}
            for strategy_name in retrievers:
                chunks = per_strategy_batch_results[strategy_name][idx_in_batch]
                match_result = evaluate_chunks_against_gold(
                    chunks, gold_sentences, gold_documents,
                    encoder=match_encoder, semantic_threshold=semantic_threshold, token_f1_threshold=token_f1_threshold,
                    emb_cache=emb_cache,
                )
                q_metrics = compute_question_metrics(match_result["gold_hits"], num_retrieved=len(chunks), ks=ks)
                q_metrics["latency_ms"] = per_strategy_batch_latency_ms[strategy_name]  # approximate: batch time / batch size
                per_strategy_metrics[strategy_name] = q_metrics

            best_strategy, _ = select_best_strategy(per_strategy_metrics, label_cfg)
            label_id = STRATEGY_NAME_TO_ID[best_strategy]

            label_records.append(
                {
                    "id": rec["id"],
                    "question": rec["question"],
                    "label": label_id,
                    "label_name": best_strategy,
                    "evidence_metrics": {
                        s: {"recall@5": m.get("recall@5", 0.0), "f1@5": m.get("f1@5", 0.0), "mrr": m.get("mrr", 0.0)}
                        for s, m in per_strategy_metrics.items()
                    },
                }
            )
            full_metrics_for_oracle.append(per_strategy_metrics)

    return label_records, full_metrics_for_oracle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true", help="Force regeneration of labels even if existing")
    args = parser.parse_args()
    cfg = load_config(args.config)

    processed_dir = resolve_path(cfg["datasets"]["processed_dir"])
    corpus_dir = resolve_path(cfg["datasets"].get("corpus_dir", "data/corpus"))
    labels_dir = resolve_path("data/labels")
    labels_dir.mkdir(parents=True, exist_ok=True)

    chunks = read_jsonl_list(corpus_dir / "chunks.jsonl")
    unified = read_jsonl_list(processed_dir / "unified.jsonl")
    logger.info("Loaded %d chunks and %d questions", len(chunks), len(unified))

    retrievers = build_all_retrievers(chunks, cfg)

    for split, out_name in _SPLIT_FILE_MAP.items():
        split_records = [r for r in unified if r["split"] == split]
        if not split_records:
            logger.warning("No records for split=%s; skipping", split)
            continue
        out_file = labels_dir / out_name
        if not args.force and out_file.exists() and out_file.stat().st_size > 0:
            existing = read_jsonl_list(out_file)
            existing_ids = {r["id"] for r in existing}
            split_ids = {r["id"] for r in split_records}
            if existing_ids == split_ids:
                logger.info("Found %s with %d/%d matching records -> skipping regeneration", out_name, len(existing), len(split_records))
                continue
            logger.info("Existing %s IDs do not match unified split (%d vs %d records) -> regenerating split", out_name, len(existing), len(split_records))
        label_records, _ = generate_labels_for_split(split_records, retrievers, cfg)
        write_jsonl(out_file, label_records)
        logger.info("Wrote %d labels to %s", len(label_records), out_name)


if __name__ == "__main__":
    main()
