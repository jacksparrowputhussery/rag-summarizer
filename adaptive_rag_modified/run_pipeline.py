"""
run_pipeline.py
=================
Runs the ENTIRE pipeline end-to-end, in the order required to sanity-check
everything before a full-scale run (spec section 38):

    Preprocessing -> Corpus construction -> BM25 -> Semantic -> Hybrid ->
    Multi-hop -> Gold matching -> Metrics -> Label generation ->
    Classifier training -> Adaptive inference

Run in DEBUG mode first (config.yaml: run.debug=true). Only switch
run.debug=false (and point datasets.use at full HF dataset downloads) once
DEBUG mode passes.

Usage:
    python run_pipeline.py --config config.yaml
"""
from __future__ import annotations

import argparse
import time

from utils import get_logger, load_config

logger = get_logger("run_pipeline")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-ablations", action="store_true", help="Skip the (slower) ablation studies")
    parser.add_argument("--from-step", type=int, default=1, help="Start pipeline from step N (1 to 7)")
    parser.add_argument("--step", type=int, default=None, help="Run only step N (1 to 7)")
    parser.add_argument("--force-baselines", action="store_true", help="Force re-running retrieval baselines even if cached")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mode = "DEBUG" if cfg["run"].get("debug", True) else "FULL"
    logger.info("=" * 70)
    logger.info("Running Modified Adaptive-RAG pipeline in %s mode", mode)
    logger.info("=" * 70)

    steps = []

    def step(name, fn, step_idx):
        steps.append((name, fn, step_idx))

    def _preprocessing():
        from preprocessing.build_unified_dataset import build
        from utils import resolve_path

        processed_dir = resolve_path(cfg["datasets"]["processed_dir"])
        unified_path = processed_dir / "unified.jsonl"
        if unified_path.exists() and unified_path.stat().st_size > 0:
            logger.info("Found existing unified dataset at %s -> skipping preprocessing", unified_path)
            return
        build(cfg)

    def _corpus():
        from corpus.build_corpus import build
        from utils import resolve_path

        corpus_dir = resolve_path(cfg["datasets"].get("corpus_dir", "data/corpus"))
        chunks_path = corpus_dir / "chunks.jsonl"
        if chunks_path.exists() and chunks_path.stat().st_size > 0:
            logger.info("Found existing corpus at %s -> skipping corpus construction", chunks_path)
            return
        build(cfg)

    def _labels():
        from labeling.generate_labels import main as gen_labels

        gen_labels()

    def _baselines():
        from utils import resolve_path

        results_dir = resolve_path("results")
        cache_dir = resolve_path("cache/retrieval_results")
        baseline_file = results_dir / "retrieval_baselines_test.json"
        metrics_file = cache_dir / "baseline_full_metrics_test.pkl"
        if (
            not args.force_baselines
            and baseline_file.exists()
            and baseline_file.stat().st_size > 0
            and metrics_file.exists()
            and metrics_file.stat().st_size > 0
        ):
            logger.info("Found existing retrieval baselines at %s -> skipping", baseline_file)
            return
        from experiments.run_retrieval_baselines import run as run_baselines

        run_baselines(cfg, split="test")

    def _classifier():
        from experiments.run_classifier import run as run_classifier

        run_classifier(cfg)

    def _adaptive():
        from experiments.run_adaptive import run as run_adaptive

        run_adaptive(cfg, split="test")

    def _ablations():
        from experiments.run_ablations import run as run_ablations

        run_ablations(cfg)

    step("1/7 Preprocessing (six datasets -> unified.jsonl)", _preprocessing, 1)
    step("2/7 Corpus construction (documents + chunks)", _corpus, 2)
    step("3/7 Strategy label generation (BM25/Semantic/Hybrid/Multihop vs gold)", _labels, 3)
    step("4/7 Independent retrieval baselines", _baselines, 4)
    step("5/7 Classifier training + evaluation", _classifier, 5)
    step("6/7 Adaptive pipeline vs fixed strategies vs oracle", _adaptive, 6)
    if not args.skip_ablations:
        step("7/7 Ablation studies", _ablations, 7)

    for name, fn, step_idx in steps:
        if args.step is not None and step_idx != args.step:
            continue
        if step_idx < args.from_step:
            continue
        logger.info("-" * 70)
        logger.info("STEP: %s", name)
        logger.info("-" * 70)
        start = time.time()
        fn()
        logger.info("Completed '%s' in %.1fs", name, time.time() - start)

    logger.info("=" * 70)
    logger.info("Pipeline complete. See results/ for all metrics, tables, and plots.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
