"""
corpus/chunker.py
===================
Splits documents into overlapping, retrievable chunks while preserving the
document id, title, and the exact sentence_ids each chunk covers. This
sentence-id bookkeeping is what lets `evaluation/evidence_matching.py` later
determine whether a retrieved chunk contains the gold sentence.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from utils import simple_sentence_split, simple_word_tokenize


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    sentence_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "text": self.text,
            "sentence_ids": self.sentence_ids,
        }


def chunk_document(doc: dict, chunk_size: int = 128, chunk_overlap: int = 20, unit: str = "words") -> list[Chunk]:
    """Chunk a single document `{doc_id, title, text}` into overlapping,
    sentence-aligned chunks of ~chunk_size words (unit="words") or, if
    unit="tokens", using the same whitespace-word tokenizer as a stand-in for
    a subword tokenizer (kept dependency-free; swap in a real tokenizer by
    passing pre-tokenized text if needed).
    """
    doc_id = doc["doc_id"]
    title = doc.get("title", "")
    sentences = simple_sentence_split(doc["text"])
    if not sentences:
        return []

    # Precompute word counts per sentence so we can greedily pack sentences
    # into chunks of ~chunk_size words with chunk_overlap words of overlap.
    sent_word_counts = [len(simple_word_tokenize(s)) for s in sentences]

    chunks: list[Chunk] = []
    start_sent = 0
    n_sents = len(sentences)
    chunk_idx = 0

    while start_sent < n_sents:
        word_count = 0
        end_sent = start_sent
        while end_sent < n_sents and (word_count == 0 or word_count + sent_word_counts[end_sent] <= chunk_size):
            word_count += sent_word_counts[end_sent]
            end_sent += 1
        end_sent = max(end_sent, start_sent + 1)  # always include at least one sentence

        sent_ids = list(range(start_sent, end_sent))
        text = " ".join(sentences[start_sent:end_sent])
        chunk_id = f"{doc_id}_chunk_{chunk_idx}"
        chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc_id, title=title, text=text, sentence_ids=sent_ids))
        chunk_idx += 1

        if end_sent >= n_sents:
            break

        # step back by `chunk_overlap` words worth of sentences for the next window
        overlap_words = 0
        back = end_sent
        while back > start_sent and overlap_words < chunk_overlap:
            back -= 1
            overlap_words += sent_word_counts[back]
        next_start = back if back > start_sent else end_sent
        start_sent = next_start

    return chunks


def chunk_documents(documents: list[dict], chunk_size: int = 128, chunk_overlap: int = 20, unit: str = "words") -> list[Chunk]:
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap, unit=unit))
    return all_chunks
