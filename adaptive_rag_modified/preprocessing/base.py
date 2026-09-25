"""
preprocessing/base.py
======================
Shared plumbing for every per-dataset loader:

  1. `make_unified_record(...)`   -> the common schema every loader must emit
  2. `try_load_hf_dataset(...)`   -> best-effort HuggingFace `datasets` loader
  3. `SyntheticKB`                -> a small offline knowledge base used as a
                                      fallback whenever the real dataset can't
                                      be downloaded (no network / no cached
                                      copy). This keeps the ENTIRE pipeline
                                      runnable end-to-end (DEBUG mode) without
                                      internet access, while the real loaders
                                      are fully implemented for when a machine
                                      with dataset access runs this code.

IMPORTANT: gold evidence is NEVER invented from the answer for real datasets.
The synthetic fallback is clearly separated and only used for offline
sanity-checking / DEBUG mode, never silently substituted for real data when
real data is available.
"""
from __future__ import annotations

import random
from typing import Any, Optional

from utils import normalize_text, simple_sentence_split, deterministic_hash, get_logger

logger = get_logger(__name__)

UNIFIED_FIELDS = [
    "id",
    "question",
    "answer",
    "dataset",
    "question_type",   # "single-hop" | "multi-hop"
    "split",
    "gold_documents",  # list[{"doc_id","title","text"}]
    "gold_sentences",  # list[{"doc_id","sentence_id","text"}]
    "num_hops",
    "metadata",
]


def make_unified_record(
    id: str,
    question: str,
    answer: str,
    dataset: str,
    question_type: str,
    split: str,
    gold_documents: list[dict],
    gold_sentences: list[dict],
    num_hops: Optional[int] = None,
    metadata: Optional[dict] = None,
) -> dict:
    return {
        "id": id,
        "question": normalize_text(question),
        "answer": normalize_text(answer) if answer else "",
        "dataset": dataset,
        "question_type": question_type,
        "split": split,
        "gold_documents": gold_documents or [],
        "gold_sentences": gold_sentences or [],
        "num_hops": num_hops if num_hops is not None else (1 if question_type == "single-hop" else 2),
        "metadata": metadata or {},
    }


def try_load_hf_dataset(path: str, name: Optional[str] = None, split: str = "train", trust_remote_code: bool = False):
    """Attempt to load a dataset via the HuggingFace `datasets` library.

    Returns the loaded dataset object, or None if it could not be loaded
    (library missing, no network, gated dataset, etc). Callers should fall
    back to `SyntheticKB` in that case.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("`datasets` library not installed; using synthetic fallback for %s", path)
        return None

    try:
        ds = load_dataset(path, name, split=split, trust_remote_code=trust_remote_code)
        return ds
    except Exception as e:  # noqa: BLE001 - any failure -> offline fallback
        logger.warning("Could not load HF dataset '%s' (%s): %s. Using synthetic fallback.", path, name, e)
        return None


def doc_to_sentences(doc_id: str, title: str, text: str) -> list[dict]:
    """Split a document's text into sentences with stable sentence_ids."""
    sents = simple_sentence_split(text)
    return [{"doc_id": doc_id, "sentence_id": i, "text": s} for i, s in enumerate(sents)]


# --------------------------------------------------------------------------- #
# Synthetic offline knowledge base (fallback only)
# --------------------------------------------------------------------------- #
_SEED_TOPICS = [
    ("Marie Curie", "physicist", "Poland", "radioactivity", 1867),
    ("Alan Turing", "mathematician", "United Kingdom", "computer science", 1912),
    ("Ada Lovelace", "mathematician", "United Kingdom", "computer programming", 1815),
    ("Nikola Tesla", "inventor", "Serbia", "alternating current", 1856),
    ("Rosalind Franklin", "chemist", "United Kingdom", "DNA structure", 1920),
    ("Albert Einstein", "physicist", "Germany", "relativity", 1879),
    ("Katherine Johnson", "mathematician", "United States", "orbital mechanics", 1918),
    ("Charles Darwin", "naturalist", "United Kingdom", "evolution", 1809),
    ("Grace Hopper", "computer scientist", "United States", "compilers", 1906),
    ("Isaac Newton", "physicist", "United Kingdom", "classical mechanics", 1642),
    ("John Cabot", "navigator", "Italy", "exploration of North America", 1450),
    ("Sebastian Cabot", "navigator", "England", "exploration of South America", 1474),
    ("Maria Mitchell", "astronomer", "United States", "comet discovery", 1818),
    ("James Clerk Maxwell", "physicist", "Scotland", "electromagnetism", 1831),
    ("Hedy Lamarr", "inventor", "Austria", "frequency hopping", 1914),
    ("Chien-Shiung Wu", "physicist", "China", "beta decay experiments", 1912),
    ("Alexander Fleming", "biologist", "Scotland", "penicillin", 1881),
    ("Barbara McClintock", "geneticist", "United States", "genetic transposition", 1902),
    ("Werner Heisenberg", "physicist", "Germany", "quantum mechanics", 1901),
    ("Emmy Noether", "mathematician", "Germany", "abstract algebra", 1882),
]

_FIRST_NAMES = [
    "Elena", "Marcus", "Priya", "Kenji", "Fatima", "Diego", "Astrid", "Ravi",
    "Ingrid", "Tomas", "Yuki", "Amara", "Lukas", "Noor", "Sven", "Leila",
    "Mateo", "Zara", "Oleg", "Naledi", "Iris", "Dmitri", "Hana", "Felix",
]
_LAST_NAMES = [
    "Volkov", "Reyes", "Nakamura", "Adeyemi", "Costa", "Lindqvist", "Haddad",
    "Kowalski", "Okafor", "Bergstrom", "Moreau", "Petrova", "Sato", "Diallo",
    "Andersen", "Rossi", "Bakker", "Nilsson", "Farrell", "Ibrahim",
]
_PROFESSIONS = [
    "physicist", "mathematician", "chemist", "biologist", "astronomer",
    "computer scientist", "geneticist", "engineer", "naturalist", "inventor",
]
_FIELDS = [
    "quantum computing", "renewable energy systems", "protein folding",
    "climate modeling", "cryptographic protocols", "neural network theory",
    "plate tectonics", "vaccine development", "robotics", "materials science",
    "radio astronomy", "behavioral genetics", "fluid dynamics", "algebraic topology",
]
_COUNTRIES = [
    "Brazil", "Kenya", "Japan", "Sweden", "Egypt", "Canada", "India",
    "Argentina", "South Korea", "Nigeria", "Norway", "Vietnam",
]


def _generate_topics(n: int, seed: int = 7) -> list[tuple]:
    """Generates `n` unique (name, profession, country, field, year) topics.
    Starts from a small pool of real historical figures for flavor, then
    extends combinatorially so the offline KB can scale beyond a handful of
    documents without duplicate names/questions colliding during dedup."""
    rng = random.Random(seed)
    topics = list(_SEED_TOPICS)
    used_names = {t[0] for t in topics}
    while len(topics) < n:
        name = f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"
        if name in used_names:
            continue
        used_names.add(name)
        profession = rng.choice(_PROFESSIONS)
        country = rng.choice(_COUNTRIES)
        field = rng.choice(_FIELDS)
        year = rng.randint(1850, 1985)
        topics.append((name, profession, country, field, year))
    return topics[:n]


class SyntheticKB:
    """A tiny, deterministic offline knowledge base used only when real
    dataset access is unavailable. Produces both single-hop and multi-hop
    style QA pairs with exact gold document/sentence provenance so the rest
    of the pipeline (chunking, retrieval, evidence matching) can be fully
    exercised offline."""

    def __init__(self, seed: int = 42, num_documents: int = 90):
        self.rng = random.Random(seed)
        self.documents: list[dict] = []
        self._topics = _generate_topics(num_documents, seed=1234)  # shared, fixed topic pool across all datasets
        self._build_documents()

    def _build_documents(self):
        n = len(self._topics)
        for i, (name, profession, country, field, year) in enumerate(self._topics):
            doc_id = f"doc_{i:04d}"
            mentee_name = self._topics[(i + 1) % n][0]  # deterministic "mentored" link to another topic
            sentences = [
                f"{name} was a {profession} born in {country} in {year}.",
                f"{name} is best known for contributions to {field}.",
                f"Throughout their career, {name} published influential work related to {field}.",
                f"{name} mentored {mentee_name} early in their career.",
                f"Institutions around the world have honored {name} for advances in {field}.",
            ]
            text = " ".join(sentences)
            self.documents.append(
                {
                    "doc_id": doc_id,
                    "title": name,
                    "text": text,
                    "sentences": sentences,
                    "field": field,
                    "mentee_name": mentee_name,
                }
            )
        self._title_to_doc = {d["title"]: d for d in self.documents}
        self._split_pools = self._partition_by_split()

    def _partition_by_split(self) -> dict:
        """Deterministically partitions the (fixed, shared) topic pool into
        disjoint train/validation/test document sets -- mirroring how real
        datasets guarantee non-overlapping official splits -- so synthetic
        question text never collides across splits during deduplication."""
        order = list(range(len(self.documents)))
        random.Random(555).shuffle(order)  # fixed seed: same partition for every loader/dataset
        n = len(order)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        return {
            "train": [self.documents[i] for i in order[:n_train]],
            "validation": [self.documents[i] for i in order[n_train:n_train + n_val]],
            "test": [self.documents[i] for i in order[n_train + n_val:]],
        }

    def sample_singlehop(self, n: int, dataset: str, split: str) -> list[dict]:
        out = []
        pool = self._split_pools.get(split, self.documents)
        if not pool:
            return out
        docs = self.rng.sample(pool, k=min(n, len(pool))) if n <= len(pool) else [
            self.rng.choice(pool) for _ in range(n)
        ]
        for i in range(n):
            doc = docs[i % len(docs)]
            sent_idx = self.rng.randrange(len(doc["sentences"]))
            qid = f"{dataset}_{split}_sh_{i:05d}_{deterministic_hash(doc['doc_id'] + str(sent_idx))}"
            phrasing = self.rng.choice(
                [
                    f"What is {doc['title']} known for?",
                    f"What field did {doc['title']} contribute to?",
                    f"Which area of study is {doc['title']} associated with?",
                    f"What was {doc['title']}'s main area of expertise?",
                ]
            )
            question = phrasing
            answer = doc["title"]
            out.append(
                make_unified_record(
                    id=qid,
                    question=question,
                    answer=answer,
                    dataset=dataset,
                    question_type="single-hop",
                    split=split,
                    gold_documents=[{"doc_id": doc["doc_id"], "title": doc["title"], "text": doc["text"]}],
                    gold_sentences=[
                        {"doc_id": doc["doc_id"], "sentence_id": sent_idx, "text": doc["sentences"][sent_idx]}
                    ],
                    num_hops=1,
                    metadata={"source": "synthetic_fallback"},
                )
            )
        return out

    def sample_multihop(self, n: int, dataset: str, split: str, hops: int = 2) -> list[dict]:
        """Builds genuine bridging (2-hop) questions: the question names only
        the first-hop entity; the second-hop entity/document must be
        *discovered* from the first hop's evidence (its "mentored X" link)
        before its gold sentence can be retrieved. A single-shot query
        (BM25/semantic/hybrid) will typically only recover the first-hop
        gold sentence, while genuine multi-hop query refinement can recover
        both -- which is exactly the behavior this offline KB is meant to
        exercise end-to-end."""
        out = []
        pool = self._split_pools.get(split, self.documents)
        if not pool:
            return out
        start_docs = self.rng.sample(pool, k=min(n, len(pool))) if n <= len(pool) else [
            self.rng.choice(pool) for _ in range(n)
        ]
        for i in range(n):
            doc_a = start_docs[i % len(start_docs)]
            doc_b = self._title_to_doc[doc_a["mentee_name"]]

            gold_documents = [
                {"doc_id": doc_a["doc_id"], "title": doc_a["title"], "text": doc_a["text"]},
                {"doc_id": doc_b["doc_id"], "title": doc_b["title"], "text": doc_b["text"]},
            ]
            gold_sentences = [
                {"doc_id": doc_a["doc_id"], "sentence_id": 3, "text": doc_a["sentences"][3]},  # "mentored X" sentence
                {"doc_id": doc_b["doc_id"], "sentence_id": 1, "text": doc_b["sentences"][1]},  # doc_b's field sentence
            ]
            qid = f"{dataset}_{split}_mh_{i:05d}_{deterministic_hash(doc_a['doc_id'] + doc_b['doc_id'])}"
            question = f"What field did the person mentored by {doc_a['title']} go on to contribute to?"
            answer = doc_b["field"]
            out.append(
                make_unified_record(
                    id=qid,
                    question=question,
                    answer=answer,
                    dataset=dataset,
                    question_type="multi-hop",
                    split=split,
                    gold_documents=gold_documents,
                    gold_sentences=gold_sentences,
                    num_hops=2,
                    metadata={"source": "synthetic_fallback"},
                )
            )
        return out
