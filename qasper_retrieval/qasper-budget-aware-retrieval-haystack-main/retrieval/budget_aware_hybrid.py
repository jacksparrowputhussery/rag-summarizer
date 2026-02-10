'''
from typing import List, Optional, Dict
import torch

from haystack.nodes.retriever import BaseRetriever
from haystack.nodes import BM25Retriever
from haystack.document_stores import InMemoryDocumentStore
from haystack.schema import Document, FilterType

from sentence_transformers import SentenceTransformer
from retrieval.scoring import score_documents


class BudgetAwareHybridRetriever(BaseRetriever):
    def __init__(
        self,
        *,
        document_store,
        embedding_model: str,
        token_budget: int,
        signal_weights: Dict[str, float],
        section_type_weights: Dict[str, float],
        use_gpu: bool = False,
        **kwargs,  # REQUIRED for Haystack 1.x
    ):
        # 🔑 Haystack 1.x: BaseRetriever takes NO args
        super().__init__(**kwargs)

        # 🔑 document_store is just an attribute in 1.x
        self.document_store = document_store
        self.store = document_store

        self.token_budget = token_budget
        self.signal_weights = signal_weights
        self.section_type_weights = section_type_weights

        # Dense encoder
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer(embedding_model, device=device)

        # -------------------------
        # Separate BM25 store
        # -------------------------
        self.keyword_store = InMemoryDocumentStore(use_bm25=True)
        self.keyword_store.write_documents(
            document_store.get_all_documents()
        )

        self.bm25 = BM25Retriever(document_store=self.keyword_store)

    def retrieve(
        self,
        query: str,
        filters: Optional[FilterType] = None,
        top_k: int = 10,
        **kwargs,
    ) -> List[Document]:

        # -------------------------
        # Dense retrieval (FAISS)
        # -------------------------
        q_emb = self.encoder.encode([query])[0]
        dense_docs = self.store.query_by_embedding(
            query_emb=q_emb,
            top_k=50,
        )

        # -------------------------
        # Sparse retrieval (BM25)
        # -------------------------
        sparse_docs = self.bm25.retrieve(
            query=query,
            top_k=50,
        )

        # -------------------------
        # Merge dense + sparse
        # -------------------------
        merged = {}

        for d in dense_docs:
            merged[d.id] = {
                "doc": d,
                "semantic": d.score or 0.0,
                "lexical": 0.0,
            }

        for d in sparse_docs:
            merged.setdefault(
                d.id,
                {"doc": d, "semantic": 0.0, "lexical": 0.0}
            )
            merged[d.id]["lexical"] = d.score or 0.0

        # -------------------------
        # Multi-signal scoring
        # -------------------------
        scored_docs = score_documents(
            list(merged.values()),
            self.signal_weights,
            self.section_type_weights,
        )

        # -------------------------
        # Budget-aware selection
        # -------------------------
        return self._budget_select(scored_docs, top_k)

    def _budget_select(
        self,
        docs: List[Document],
        top_k: int,
    ) -> List[Document]:

        selected = []
        used_tokens = 0

        docs = sorted(
            docs,
            key=lambda d: d.score / max(d.meta.get("word_count", 1), 1),
            reverse=True,
        )

        for d in docs:
            wc = d.meta.get("word_count", 0)

            if used_tokens + wc <= self.token_budget:
                selected.append(d)
                used_tokens += wc

            if len(selected) >= top_k:
                break

        return sorted(selected, key=lambda d: d.score, reverse=True)

    def retrieve_batch(
        self,
        queries: List[str],
        filters: Optional[List[Optional[FilterType]]] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> List[List[Document]]:

        if top_k is None:
            top_k = 10

        if filters is None:
            filters = [None] * len(queries)

        return [
            self.retrieve(query=q, filters=f, top_k=top_k)
            for q, f in zip(queries, filters)
        ]
'''


''''
from typing import List, Optional, Dict
import torch

from haystack.nodes.retriever import BaseRetriever
from haystack.nodes import BM25Retriever
from haystack.document_stores import InMemoryDocumentStore
from haystack.schema import Document, FilterType

from sentence_transformers import SentenceTransformer
from retrieval.scoring import score_documents


class BudgetAwareHybridRetriever(BaseRetriever):
    def __init__(
        self,
        *,
        document_store,
        embedding_model: str,
        token_budget: int,
        signal_weights: Dict[str, float],
        section_type_weights: Dict[str, float],
        use_gpu: bool = False,
        **kwargs,  # REQUIRED for Haystack 1.x
    ):
        # 🔑 Haystack 1.x: BaseRetriever takes NO args
        super().__init__(**kwargs)

        # 🔑 document_store is just an attribute in 1.x
        self.document_store = document_store
        self.store = document_store

        self.token_budget = token_budget
        self.signal_weights = signal_weights
        self.section_type_weights = section_type_weights

        # Dense encoder
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer(embedding_model, device=device)

        # -------------------------
        # Separate BM25 store
        # -------------------------
        self.keyword_store = InMemoryDocumentStore(use_bm25=True)
        self.keyword_store.write_documents(
            document_store.get_all_documents()
        )

        self.bm25 = BM25Retriever(document_store=self.keyword_store)

    def retrieve(
        self,
        query: str,
        filters: Optional[FilterType] = None,
        top_k: int = 10,
        **kwargs,
    ) -> List[Document]:

        # -------------------------
        # Dense retrieval (FAISS)
        # -------------------------
        q_emb = self.encoder.encode([query])[0]
        
        # 🔧 FIX: Pass filters to FAISS query
        dense_docs = self.store.query_by_embedding(
            query_emb=q_emb,
            top_k=50,
            filters=filters,  # Now properly filtered
        )

        # -------------------------
        # Sparse retrieval (BM25)
        # -------------------------
        # 🔧 FIX: Pass filters to BM25 retrieval
        sparse_docs = self.bm25.retrieve(
            query=query,
            top_k=50,
            filters=filters,  # Now properly filtered
        )

        # -------------------------
        # Merge dense + sparse
        # -------------------------
        merged = {}

        for d in dense_docs:
            merged[d.id] = {
                "doc": d,
                "semantic": d.score or 0.0,
                "lexical": 0.0,
            }

        for d in sparse_docs:
            merged.setdefault(
                d.id,
                {"doc": d, "semantic": 0.0, "lexical": 0.0}
            )
            merged[d.id]["lexical"] = d.score or 0.0

        # -------------------------
        # Multi-signal scoring
        # -------------------------
        scored_docs = score_documents(
            list(merged.values()),
            self.signal_weights,
            self.section_type_weights,
        )

        # -------------------------
        # Budget-aware selection
        # -------------------------
        return self._budget_select(scored_docs, top_k)

    def _budget_select(
        self,
        docs: List[Document],
        top_k: int,
    ) -> List[Document]:
        """
        Select documents that fit within token budget,
        prioritizing score-to-token ratio.
        """
        selected = []
        used_tokens = 0

        # 🔧 FIX: Added safety check for word_count
        docs = sorted(
            docs,
            key=lambda d: d.score / max(d.meta.get("word_count", 1), 1),
            reverse=True,
        )

        for d in docs:
            wc = d.meta.get("word_count", 0)

            if used_tokens + wc <= self.token_budget:
                selected.append(d)
                used_tokens += wc

            if len(selected) >= top_k:
                break

        # Return in score order (highest first)
        return sorted(selected, key=lambda d: d.score, reverse=True)

    def retrieve_batch(
        self,
        queries: List[str],
        filters: Optional[List[Optional[FilterType]]] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> List[List[Document]]:
        """
        Batch retrieval for multiple queries.
        """
        if top_k is None:
            top_k = 10

        if filters is None:
            filters = [None] * len(queries)

        return [
            self.retrieve(query=q, filters=f, top_k=top_k)
            for q, f in zip(queries, filters)
        ]
'''


from typing import List, Optional, Dict
import torch

from haystack.nodes.retriever import BaseRetriever
from haystack.nodes import BM25Retriever
from haystack.document_stores import InMemoryDocumentStore
from haystack.schema import Document, FilterType

from sentence_transformers import SentenceTransformer
from retrieval.scoring import score_documents


class BudgetAwareHybridRetriever(BaseRetriever):
    def __init__(
        self,
        *,
        document_store,
        embedding_model: str,
        token_budget: int,
        signal_weights: Dict[str, float],
        section_type_weights: Dict[str, float],
        use_gpu: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.document_store = document_store
        self.store = document_store

        self.token_budget = token_budget
        self.signal_weights = signal_weights
        self.section_type_weights = section_type_weights

        # Dense encoder
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer(embedding_model, device=device)

        # Separate BM25 store
        self.keyword_store = InMemoryDocumentStore(use_bm25=True)
        self.keyword_store.write_documents(
            document_store.get_all_documents()
        )

        self.bm25 = BM25Retriever(document_store=self.keyword_store)
        
        # 🔧 FIX: Build paper_id index for filtering
        self._build_paper_index()

    def _build_paper_index(self):
        """Build an index of document IDs by paper_id for fast filtering."""
        self.paper_index = {}
        
        for doc in self.store.get_all_documents():
            paper_id = doc.meta.get("paper_id")
            if paper_id:
                if paper_id not in self.paper_index:
                    self.paper_index[paper_id] = []
                self.paper_index[paper_id].append(doc.id)
        
        print(f"📊 Indexed {len(self.paper_index)} papers")

    def retrieve(
        self,
        query: str,
        filters: Optional[FilterType] = None,
        top_k: int = 10,
        **kwargs,
    ) -> List[Document]:

        # 🔧 FIX: Get allowed document IDs from filters
        allowed_ids = self._get_allowed_ids(filters)

        # Dense retrieval (FAISS)
        q_emb = self.encoder.encode([query])[0]
        
        dense_docs = self.store.query_by_embedding(
            query_emb=q_emb,
            top_k=100,  # Get more candidates for filtering
        )

        # Sparse retrieval (BM25)
        sparse_docs = self.bm25.retrieve(
            query=query,
            top_k=100,  # Get more candidates for filtering
        )

        # 🔧 FIX: Apply manual filtering
        if allowed_ids:
            dense_docs = [d for d in dense_docs if d.id in allowed_ids]
            sparse_docs = [d for d in sparse_docs if d.id in allowed_ids]

        # Merge dense + sparse
        merged = {}

        for d in dense_docs:
            merged[d.id] = {
                "doc": d,
                "semantic": d.score or 0.0,
                "lexical": 0.0,
            }

        for d in sparse_docs:
            merged.setdefault(
                d.id,
                {"doc": d, "semantic": 0.0, "lexical": 0.0}
            )
            merged[d.id]["lexical"] = d.score or 0.0

        # Multi-signal scoring
        scored_docs = score_documents(
            list(merged.values()),
            self.signal_weights,
            self.section_type_weights,
        )

        # Budget-aware selection
        return self._budget_select(scored_docs, top_k)

    def _get_allowed_ids(self, filters: Optional[FilterType]) -> Optional[set]:
        """
        Extract allowed document IDs from filters.
        Returns None if no filters, otherwise returns set of allowed doc IDs.
        """
        if not filters:
            return None
        
        paper_ids = filters.get("paper_id", [])
        if not paper_ids:
            return None
        
        allowed_ids = set()
        for pid in paper_ids:
            allowed_ids.update(self.paper_index.get(pid, []))
        
        return allowed_ids if allowed_ids else None

    def _budget_select(
        self,
        docs: List[Document],
        top_k: int,
    ) -> List[Document]:
        """
        Select documents that fit within token budget.
        
        🔧 FIX: Changed strategy to prioritize high-scoring documents
        instead of score-to-token ratio, which favored short headers.
        """
        selected = []
        used_tokens = 0

        # 🔧 FIX: Sort by SCORE ONLY (not score/word_count)
        # This prevents short headers from dominating results
        docs = sorted(docs, key=lambda d: d.score, reverse=True)

        for d in docs:
            wc = d.meta.get("word_count", 0)

            # Add document if it fits in budget
            if used_tokens + wc <= self.token_budget:
                selected.append(d)
                used_tokens += wc

                # Stop if we have enough documents
                if len(selected) >= top_k:
                    break

        return selected

    def retrieve_batch(
        self,
        queries: List[str],
        filters: Optional[List[Optional[FilterType]]] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> List[List[Document]]:
        """
        Batch retrieval for multiple queries.
        """
        if top_k is None:
            top_k = 10

        if filters is None:
            filters = [None] * len(queries)

        return [
            self.retrieve(query=q, filters=f, top_k=top_k)
            for q, f in zip(queries, filters)
        ]