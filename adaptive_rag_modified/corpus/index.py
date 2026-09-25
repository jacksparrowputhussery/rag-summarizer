"""
corpus/index.py
=================
Builds (and caches) the indices shared by the retrievers:
  - BM25 index over chunk text (rank_bm25.BM25Okapi)
  - Dense embedding matrix + FAISS index over chunk text

Both are expensive to build, so results are cached to disk and only rebuilt
when the underlying chunk corpus or config changes (tracked via a hash of
chunk_ids + config signature).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from utils import deterministic_hash, get_logger, resolve_path

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #
class BM25Index:
    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        from rank_bm25 import BM25Okapi
        from utils import simple_word_tokenize

        self.chunks = chunks
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        tokenized = [simple_word_tokenize(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(tokenized, k1=k1, b=b)

    def get_scores(self, query: str) -> np.ndarray:
        from utils import simple_word_tokenize

        tokens = simple_word_tokenize(query)
        return np.asarray(self._bm25.get_scores(tokens))


def _cache_key(chunks: list[dict], extra: str) -> str:
    sig = "|".join(c["chunk_id"] for c in chunks[:1000]) + f"|n={len(chunks)}|{extra}"
    return deterministic_hash(sig)


def build_or_load_bm25(chunks: list[dict], cfg: dict) -> BM25Index:
    cache_dir = resolve_path(cfg["bm25"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(chunks, f"k1={cfg['bm25']['k1']},b={cfg['bm25']['b']}")
    cache_path = cache_dir / f"bm25_{key}.pkl"

    if cache_path.exists():
        logger.info("Loading cached BM25 index from %s", cache_path)
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    logger.info("Building BM25 index over %d chunks", len(chunks))
    index = BM25Index(chunks, k1=cfg["bm25"]["k1"], b=cfg["bm25"]["b"])
    with open(cache_path, "wb") as f:
        pickle.dump(index, f)
    return index


# --------------------------------------------------------------------------- #
# Dense / semantic
# --------------------------------------------------------------------------- #
class TfidfEncoder:
    """Offline-safe stand-in for a sentence-embedding model. Used when
    run.debug=true or when sentence-transformers / model download is
    unavailable. Implements the same `.encode()` interface."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(max_features=20000)
        self._fitted = False

    def fit(self, texts: list[str]):
        self._vectorizer.fit(texts)
        self._fitted = True

    def encode(self, texts: list[str], batch_size: int = 32, show_progress_bar: bool = False, normalize_embeddings: bool = True) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        mat = self._vectorizer.transform(texts).toarray().astype("float32")
        if normalize_embeddings:
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat = mat / norms
        return mat


def load_encoder(cfg: dict):
    """Returns an object exposing `.encode(list[str]) -> np.ndarray`.

    Uses sentence-transformers when run.debug=false AND the library / model
    weights are reachable; otherwise falls back to a TF-IDF encoder so the
    whole pipeline runs fully offline.
    """
    debug = cfg["run"].get("debug", True)
    model_name = cfg["semantic"]["model_name"] if not debug else cfg["semantic"].get("debug_model_name", "tfidf")

    if model_name != "tfidf":
        try:
            from sentence_transformers import SentenceTransformer
            from utils import get_device

            device = get_device(cfg)
            logger.info("Loading dense encoder '%s' on device=%s", model_name, device)
            return SentenceTransformer(model_name, device=str(device) if device is not None else None)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not load '%s' (%s); falling back to TF-IDF encoder.", model_name, e)

    logger.info("Using offline TF-IDF encoder as semantic stand-in")
    return TfidfEncoder()


class DenseIndex:
    def __init__(self, chunks: list[dict], embeddings: np.ndarray, encoder):
        self.chunks = chunks
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.embeddings = embeddings  # (N, D), L2-normalized
        self.encoder = encoder
        self._faiss_index = None
        self._try_build_faiss()

    def _try_build_faiss(self):
        try:
            import faiss

            dim = self.embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)  # cosine sim via inner product on normalized vecs
            index.add(self.embeddings)
            self._faiss_index = index
        except Exception as e:  # noqa: BLE001
            logger.warning("FAISS unavailable (%s); falling back to numpy brute-force search.", e)
            self._faiss_index = None

    def search(self, query_vec: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (scores, indices) into self.chunks, top_k each."""
        query_vec = query_vec.reshape(1, -1).astype("float32")
        if self._faiss_index is not None:
            scores, idxs = self._faiss_index.search(query_vec, min(top_k, len(self.chunks)))
            return scores[0], idxs[0]
        sims = (self.embeddings @ query_vec.T).ravel()
        top_k = min(top_k, len(sims))
        idxs = np.argpartition(-sims, top_k - 1)[:top_k]
        idxs = idxs[np.argsort(-sims[idxs])]
        return sims[idxs], idxs

    def search_batch(self, query_vecs: np.ndarray, top_k: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Batched version of `search`: query_vecs is (num_queries, dim).
        Returns a list of (scores, indices) pairs, one per query. This is
        the main GPU-utilization win during large-scale runs -- FAISS
        natively batches multi-row search in a single call, and even the
        numpy fallback does one matrix multiply for the whole batch instead
        of `num_queries` separate ones."""
        query_vecs = np.asarray(query_vecs, dtype="float32")
        if query_vecs.ndim == 1:
            query_vecs = query_vecs.reshape(1, -1)
        k = min(top_k, len(self.chunks))

        if self._faiss_index is not None:
            scores, idxs = self._faiss_index.search(query_vecs, k)  # (n, k) each, native batch
            return [(scores[i], idxs[i]) for i in range(len(query_vecs))]

        # numpy fallback: one (n_queries, n_chunks) matrix multiply for the whole batch
        sims = query_vecs @ self.embeddings.T  # (n_queries, n_chunks)
        results = []
        for row in sims:
            idxs = np.argpartition(-row, k - 1)[:k]
            idxs = idxs[np.argsort(-row[idxs])]
            results.append((row[idxs], idxs))
        return results


def build_or_load_dense(chunks: list[dict], cfg: dict) -> DenseIndex:
    encoder = load_encoder(cfg)
    debug = cfg["run"].get("debug", True)
    model_name = cfg["semantic"]["model_name"] if not debug else cfg["semantic"].get("debug_model_name", "tfidf")

    cache_dir = resolve_path(cfg["semantic"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(chunks, f"model={model_name}")
    emb_cache_path = cache_dir / f"emb_{key}.npy"

    texts = [c["text"] for c in chunks]

    if isinstance(encoder, TfidfEncoder):
        # TF-IDF must be fit on this exact corpus; embeddings aren't portable
        # across corpora so we don't reuse cache across different chunk sets
        # beyond the hash key already encodes chunk_ids.
        if emb_cache_path.exists():
            logger.info("Loading cached TF-IDF embeddings from %s", emb_cache_path)
            embeddings = np.load(emb_cache_path)
            encoder.fit(texts)  # still need a fitted vectorizer for query-time encode()
        else:
            logger.info("Encoding %d chunks with TF-IDF stand-in encoder", len(chunks))
            embeddings = encoder.encode(texts, normalize_embeddings=True)
            np.save(emb_cache_path, embeddings)
    else:
        if emb_cache_path.exists():
            logger.info("Loading cached dense embeddings from %s", emb_cache_path)
            embeddings = np.load(emb_cache_path)
        else:
            logger.info("Encoding %d chunks with '%s'", len(chunks), model_name)
            embeddings = np.asarray(
                encoder.encode(texts, batch_size=cfg["semantic"]["batch_size"], show_progress_bar=False, normalize_embeddings=True)
            ).astype("float32")
            np.save(emb_cache_path, embeddings)

    return DenseIndex(chunks, embeddings.astype("float32"), encoder)


def encode_query(encoder, query: str) -> np.ndarray:
    vec = encoder.encode([query], normalize_embeddings=True)
    return np.asarray(vec[0], dtype="float32")


def encode_queries(encoder, queries: list[str], batch_size: int = 32) -> np.ndarray:
    """Batched query encoding -- the main GPU-utilization win. Encoding N
    queries one at a time means N separate forward passes (mostly idle GPU
    between calls); encoding them together lets the encoder actually batch
    the matrix ops. `batch_size` here is the encoder's internal batch size
    (relevant for sentence-transformers on GPU); the TF-IDF stand-in ignores
    it since it's a single vectorized transform regardless."""
    if not queries:
        return np.zeros((0, 0), dtype="float32")
    vecs = encoder.encode(queries, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")
