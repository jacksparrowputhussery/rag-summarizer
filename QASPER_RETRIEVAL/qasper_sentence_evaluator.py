import numpy as np
import re
from typing import List, Dict


class SentenceLevelEvaluator:
    """
    Full KILT-style sentence-level evaluator for QASPER.
    
    Computes:
        - Recall@k
        - Precision@k
        - HitRatio@k
        - NDCG@k
        - MRR
        - MAP
        - Answer Coverage (optional)
        
    Works on top of your existing:
        - loader (QASPERLoader)
        - indexer (SimpleIndexer)
        - retriever (SimpleTier1Retriever)
    """

    def __init__(self, loader, indexer, retriever, max_sentences_per_q=50):
        self.loader = loader
        self.indexer = indexer
        self.retriever = retriever
        self.max_sentences_per_q = max_sentences_per_q
        self.encoder = indexer.encoder  # same embedding model as chunks

    # ------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------

    def split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if len(s) > 5]

    def substantial_overlap(self, a: str, b: str, threshold=0.25) -> bool:
        w1 = set(a.lower().split())
        w2 = set(b.lower().split())
        if not w1 or not w2:
            return False
        overlap = len(w1 & w2) / len(w1)
        return overlap >= threshold

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    def compute_precision_at_k(self, ranked, gold, k):
        hits = sum(
            any(self.substantial_overlap(s, g) for g in gold)
            for s in ranked[:k]
        )
        return hits / k

    def compute_recall_at_k(self, ranked, gold, k):
        if len(gold) == 0:
            return 1.0
        hits = 0
        for g in gold:
            if any(self.substantial_overlap(g, s) for s in ranked[:k]):
                hits += 1
        return hits / len(gold)

    def compute_hit_ratio(self, ranked, gold, k):
        for s in ranked[:k]:
            if any(self.substantial_overlap(s, g) for g in gold):
                return 1.0
        return 0.0

    def compute_mrr(self, ranked, gold):
        for i, s in enumerate(ranked, 1):
            if any(self.substantial_overlap(s, g) for g in gold):
                return 1.0 / i
        return 0.0

    def compute_map(self, ranked, gold):
        if len(gold) == 0:
            return 1.0

        hits = 0
        ap_scores = []

        for i, s in enumerate(ranked, 1):
            if any(self.substantial_overlap(s, g) for g in gold):
                hits += 1
                ap_scores.append(hits / i)

        return np.mean(ap_scores) if ap_scores else 0.0

    def compute_ndcg(self, ranked, gold, k):
        def dcg(scores):
            return sum(s / np.log2(i + 2) for i, s in enumerate(scores))

        rel = []
        for s in ranked[:k]:
            is_rel = any(self.substantial_overlap(s, g) for g in gold)
            rel.append(1 if is_rel else 0)

        ideal = sorted(rel, reverse=True)

        dcg_val = dcg(rel)
        idcg_val = dcg(ideal)

        if idcg_val == 0:
            return 0.0

        return dcg_val / idcg_val

    # ------------------------------------------------------------
    # Evaluate one question
    # ------------------------------------------------------------

    def evaluate_question(self, query: str, evidence_sents: List[str]):
        """Retrieve top paragraphs, split to sentences, rank sentences, compute metrics."""

        # Retrieve paragraphs first
        para_ids = self.retriever.retrieve(query, top_k=10)

        # Extract sentences
        all_sents = []
        for cid in para_ids:
            text = self.indexer.chunks[cid].content
            all_sents.extend(self.split_into_sentences(text))

        # Sentence budget
        all_sents = all_sents[:self.max_sentences_per_q]
        if len(all_sents) == 0:
            return None

        # Rank sentences semantically
        q_emb = self.encoder.encode([query])[0]
        sent_embs = self.encoder.encode(all_sents)

        sims = np.dot(sent_embs, q_emb) / (
            np.linalg.norm(sent_embs, axis=1) * np.linalg.norm(q_emb)
        )

        ranked = [s for _, s in sorted(zip(sims, all_sents), reverse=True)]

        # Compute metrics
        metrics = {}

        for k in [5, 10]:
            metrics[f"precision@{k}"] = self.compute_precision_at_k(ranked, evidence_sents, k)
            metrics[f"recall@{k}"] = self.compute_recall_at_k(ranked, evidence_sents, k)
            metrics[f"hit_ratio@{k}"] = self.compute_hit_ratio(ranked, evidence_sents, k)
            metrics[f"ndcg@{k}"] = self.compute_ndcg(ranked, evidence_sents, k)

        metrics["mrr"] = self.compute_mrr(ranked, evidence_sents)
        metrics["map"] = self.compute_map(ranked, evidence_sents)

        return metrics

    # ------------------------------------------------------------
    # Evaluate an entire paper
    # ------------------------------------------------------------

    def evaluate_paper(self, paper_id, k_values=[5, 10]):
        questions = self.loader.get_questions(paper_id)

        results = {
            "paper_id": paper_id,
            "num_questions": len(questions),
            "per_question": [],
            "aggregated": {}
        }

        # collect metrics
        metric_lists = {}
        for k in k_values:
            metric_lists[f"precision@{k}"] = []
            metric_lists[f"recall@{k}"] = []
            metric_lists[f"hit_ratio@{k}"] = []
            metric_lists[f"ndcg@{k}"] = []

        metric_lists["mrr"] = []
        metric_lists["map"] = []

        # evaluate each question
        for q in questions:
            query = q["question"]
            gold_evidence = q["evidence"]   # already gold sentences in QASPER

            q_metrics = self.evaluate_question(query, gold_evidence)
            if q_metrics is None:
                continue

            results["per_question"].append({
                "question": query,
                **q_metrics
            })

            # add to aggregated lists
            for m, v in q_metrics.items():
                metric_lists[m].append(v)

        # aggregate
        for m, values in metric_lists.items():
            if len(values) == 0:
                results["aggregated"][m] = 0.0
            else:
                results["aggregated"][m] = float(np.mean(values))

        return results
