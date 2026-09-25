"""
corpus/build_corpus.py
========================
Builds a single, consistent document + chunk corpus from the union of every
`gold_documents` entry across the unified dataset (this guarantees every
question's gold evidence is actually retrievable from the corpus, while
retrieval strategies still have to find it among distractor chunks from
other documents/questions).

Writes:
    data/corpus/documents.jsonl   {doc_id, title, text}
    data/corpus/chunks.jsonl      {chunk_id, doc_id, title, text, sentence_ids}

Usage:
    python -m corpus.build_corpus --config config.yaml
"""
from __future__ import annotations

import argparse

from corpus.chunker import chunk_documents
from utils import get_logger, load_config, read_jsonl_list, resolve_path, write_jsonl

logger = get_logger(__name__)


def collect_documents(unified_records: list[dict]) -> list[dict]:
    seen = {}
    for r in unified_records:
        for d in r.get("gold_documents", []):
            if d["doc_id"] not in seen:
                seen[d["doc_id"]] = {"doc_id": d["doc_id"], "title": d.get("title", ""), "text": d["text"]}
    return list(seen.values())


def build(cfg: dict) -> tuple[list[dict], list[dict]]:
    processed_dir = resolve_path(cfg["datasets"]["processed_dir"])
    corpus_dir = resolve_path(cfg["datasets"].get("corpus_dir", "data/corpus"))
    corpus_dir.mkdir(parents=True, exist_ok=True)

    unified_path = processed_dir / "unified.jsonl"
    unified_records = read_jsonl_list(unified_path)
    logger.info("Loaded %d unified examples for corpus construction", len(unified_records))

    documents = collect_documents(unified_records)
    logger.info("Collected %d unique documents", len(documents))

    chunk_cfg = cfg["chunking"]
    chunks = chunk_documents(
        documents,
        chunk_size=chunk_cfg["chunk_size"],
        chunk_overlap=chunk_cfg["chunk_overlap"],
        unit=chunk_cfg.get("unit", "words"),
    )
    logger.info("Produced %d chunks (chunk_size=%d, overlap=%d)", len(chunks), chunk_cfg["chunk_size"], chunk_cfg["chunk_overlap"])

    write_jsonl(corpus_dir / "documents.jsonl", documents)
    write_jsonl(corpus_dir / "chunks.jsonl", [c.to_dict() for c in chunks])
    return documents, [c.to_dict() for c in chunks]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    build(cfg)


if __name__ == "__main__":
    main()
