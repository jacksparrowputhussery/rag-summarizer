# test_kilt_implementation.py
"""
Fixed QASPER sentence-level evaluation runner.
- Converts QASPER evidence (section + paragraph indices) into gold sentences
- Uses SentenceLevelEvaluator.evaluate_question(...) which expects gold sentence strings
- Computes KILT-style retrieval metrics and aggregates them
"""

import json
import numpy as np
from typing import List, Dict, Set
from tqdm import tqdm
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from qasper_sentence_evaluator import SentenceLevelEvaluator


# -------------------------
# QASPER Loader (same shape as your dataset)
# -------------------------
class QASPERLoader:
    def __init__(self, qasper_path: str):
        self.qasper_path = qasper_path
        self.data = self._load_qasper()

    def _load_qasper(self) -> Dict:
        with open(self.qasper_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_paper_ids(self) -> List[str]:
        return list(self.data.keys())

    def get_paper_sections(self, paper_id: str) -> List[Dict]:
        paper = self.data[paper_id]
        sections = []
        sections.append({
            "title": "Abstract",
            "content": paper.get("abstract", ""),
            "section_type": "abstract"
        })

        full_text_dict = paper.get("full_text", {})
        section_names = full_text_dict.get("section_name", [])
        paragraphs_lists = full_text_dict.get("paragraphs", [])

        if not isinstance(section_names, list):
            section_names = [section_names]
        if not isinstance(paragraphs_lists, list):
            paragraphs_lists = [paragraphs_lists]

        for idx, sec_name in enumerate(section_names):
            if idx < len(paragraphs_lists):
                paras = paragraphs_lists[idx]
                if isinstance(paras, list):
                    content = "\n\n".join(str(p) for p in paras if p)
                else:
                    content = str(paras)
                sections.append({
                    "title": sec_name,
                    "content": content,
                    "section_type": self._classify(sec_name)
                })

        return sections

    def _classify(self, section_name: str) -> str:
        name = section_name.lower()
        if "abstract" in name: return "abstract"
        if any(k in name for k in ["method", "approach", "model", "experiment"]): return "methodology"
        if any(k in name for k in ["result", "evaluation", "performance"]): return "results"
        if any(k in name for k in ["conclusion", "discussion", "future"]): return "conclusion"
        if any(k in name for k in ["introduction", "background", "related"]): return "introduction"
        return "body"

    def get_questions(self, paper_id: str) -> List[Dict]:
        """
        Returns questions in a normalized form:
          - question: str
          - evidence: raw evidence items (as stored in QASPER)
          - extractive_spans: list[str] (may be empty)
        """
        paper = self.data[paper_id]
        qas = paper.get("qas", {})

        questions = []
        question_texts = qas.get("question", [])
        answers_list = qas.get("answers", [])

        # ensure lengths
        n = len(question_texts)
        if len(answers_list) < n:
            answers_list.extend([[] for _ in range(len(answers_list), n)])

        for idx, qtext in enumerate(question_texts):
            ans_container = answers_list[idx] if idx < len(answers_list) else []
            evidence = []
            extractive_spans = []

            # fallback handling of different formats
            if isinstance(ans_container, dict) and "answer" in ans_container:
                anns = ans_container["answer"]
            elif isinstance(ans_container, list) and len(ans_container) > 0 and isinstance(ans_container[0], dict) and "answer" in ans_container[0]:
                anns = ans_container[0]["answer"]
            else:
                anns = ans_container  # maybe it's already list of annotation dicts

            if isinstance(anns, list) and len(anns) > 0:
                ann = anns[0]
                if isinstance(ann, dict):
                    if ann.get("unanswerable", False):
                        evidence = []
                        extractive_spans = []
                    else:
                        evidence = ann.get("evidence", []) or []
                        extractive_spans = ann.get("extractive_spans", []) or []
                else:
                    # unexpected: ann might be a primitive
                    evidence = []
                    extractive_spans = []
            else:
                evidence = []
                extractive_spans = []

            questions.append({
                "question": qtext,
                "evidence": evidence,
                "extractive_spans": extractive_spans
            })

        return questions


# -------------------------
# Simple indexer / retriever (keeps short paragraphs)
# -------------------------
@dataclass
class Chunk:
    id: str
    content: str
    doc_id: str
    section_name: str
    section_type: str
    position: float
    token_count: int
    paragraph_idx: int


class SimpleIndexer:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.chunks: Dict[str, Chunk] = {}
        self.chunk_embeddings = np.array([])  # will hold numpy array
        self.chunk_id_to_idx: Dict[str, int] = {}

    def index_paper(self, paper_id: str, sections: List[Dict]):
        all_chunks = []
        para_counter = len(self.chunks)  # continue counter across papers

        for s_idx, sec in enumerate(sections):
            paras = sec.get("content", "").split("\n\n")
            for p_idx, p in enumerate(paras):
                p = p.strip()
                if len(p) == 0:
                    continue

                cid = f"{paper_id}_s{s_idx}_p{p_idx}"
                chunk = Chunk(
                    id=cid,
                    content=p,
                    doc_id=paper_id,
                    section_name=sec.get("title", ""),
                    section_type=sec.get("section_type", "body"),
                    position=s_idx / max(len(sections), 1),
                    token_count=len(p.split()),
                    paragraph_idx=para_counter
                )
                self.chunks[cid] = chunk
                all_chunks.append(chunk)
                para_counter += 1

        # encode new chunk texts and append to existing embeddings
        if len(all_chunks) > 0:
            texts = [c.content for c in all_chunks]
            embs = self.encoder.encode(texts, batch_size=32)
            if self.chunk_embeddings.size == 0:
                self.chunk_embeddings = embs
            else:
                self.chunk_embeddings = np.vstack([self.chunk_embeddings, embs])
            start = len(self.chunk_id_to_idx)
            for i, c in enumerate(all_chunks):
                self.chunk_id_to_idx[c.id] = start + i


class SimpleScorer:
    def __init__(self, indexer: SimpleIndexer):
        self.indexer = indexer
        corpus = [c.content for c in indexer.chunks.values()]
        tokenized = [doc.lower().split() for doc in corpus]
        # If corpus empty, BM25 will raise — caller must ensure scorer constructed after indexing
        self.bm25 = BM25Okapi(tokenized) if len(tokenized) > 0 else None

    def score(self, query: str, q_emb, ids: List[str]):
        bm25_scores = self.bm25.get_scores(query.lower().split()) if self.bm25 is not None else np.zeros(len(ids))
        out = {}
        for cid in ids:
            idx = self.indexer.chunk_id_to_idx[cid]
            chunk_emb = self.indexer.chunk_embeddings[idx]
            semantic = np.dot(q_emb, chunk_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(chunk_emb))
            lex = bm25_scores[idx] if self.bm25 is not None else 0.0
            out[cid] = 0.7 * semantic + 0.3 * lex
        return out


class SimpleTier1Retriever:
    def __init__(self, indexer: SimpleIndexer, scorer: SimpleScorer, token_budget=4000):
        self.indexer = indexer
        self.scorer = scorer
        self.token_budget = token_budget

    def retrieve(self, query: str, top_k: int = 10):
        # compute semantic sims
        if self.indexer.chunk_embeddings.size == 0:
            return []
        q_emb = self.indexer.encoder.encode([query])[0]
        sims = np.dot(self.indexer.chunk_embeddings, q_emb) / (np.linalg.norm(self.indexer.chunk_embeddings, axis=1) * np.linalg.norm(q_emb))
        top_idxs = np.argsort(sims)[::-1][:50]
        cids = list(self.indexer.chunks.keys())
        candidates = [cids[i] for i in top_idxs if i < len(cids)]
        scores = self.scorer.score(query, q_emb, candidates)
        sorted_by_score = sorted(candidates, key=lambda x: scores[x], reverse=True)
        selected = []
        tokens = 0
        for cid in sorted_by_score:
            chunk = self.indexer.chunks[cid]
            if tokens + chunk.token_count <= self.token_budget:
                selected.append(cid)
                tokens += chunk.token_count
                if len(selected) >= top_k:
                    break
        return selected


# -------------------------
# Utilities: convert QASPER evidence -> gold sentences
# -------------------------
def evidence_to_sentences(loader: QASPERLoader, paper_id: str, evidence_list, extractive_spans) -> List[str]:
    """
    Convert QASPER evidence entries to actual sentence strings.
    - evidence_list: list of evidence items (often [section_name, paragraph_idx] or similar)
    - extractive_spans: list of exact strings extracted by annotator (may be used directly)
    Returns deduplicated list of gold sentences (strings).
    """
    gold: List[str] = []
    seen: Set[str] = set()

    # Add extractive spans directly (these are textual)
    if extractive_spans:
        for s in extractive_spans:
            if not s:
                continue
            ss = s.strip()
            if ss and ss not in seen:
                gold.append(ss)
                seen.add(ss)

    paper = loader.data[paper_id]
    full_text = paper.get("full_text", {})
    section_names = full_text.get("section_name", [])
    paragraphs_lists = full_text.get("paragraphs", [])

    # normalize lists
    if not isinstance(section_names, list):
        section_names = [section_names]
    if not isinstance(paragraphs_lists, list):
        paragraphs_lists = [paragraphs_lists]

    # helper: find section index by name (case-insensitive)
    def find_section_idx(name: str):
        name_lower = name.lower()
        for i, sn in enumerate(section_names):
            if isinstance(sn, str) and sn.lower() == name_lower:
                return i
        # fallback: try substring match
        for i, sn in enumerate(section_names):
            if isinstance(sn, str) and name_lower in sn.lower():
                return i
        return None

    # evidence_list could be empty or contain various formats
    if evidence_list:
        for ev in evidence_list:
            # format 1: [section_name, paragraph_idx]
            if isinstance(ev, (list, tuple)) and len(ev) >= 2:
                sec = ev[0]
                pidx = ev[1]
                try:
                    # pidx may be string or int; try cast to int
                    pidx = int(pidx)
                except Exception:
                    # can't parse index -> skip
                    continue

                sec_idx = find_section_idx(sec) if isinstance(sec, str) else None
                if sec_idx is None or sec_idx >= len(paragraphs_lists):
                    # fallback: try to treat the first section (abstract) if available
                    continue

                sec_paras = paragraphs_lists[sec_idx]
                # ensure sec_paras is a list
                if not isinstance(sec_paras, list) or pidx < 0 or pidx >= len(sec_paras):
                    continue

                paragraph_text = str(sec_paras[pidx]).strip()
                # split paragraph into sentences
                sents = re.split(r'(?<=[.!?])\s+', paragraph_text)
                for s in sents:
                    s_strip = s.strip()
                    if len(s_strip) > 3 and s_strip not in seen:
                        gold.append(s_strip)
                        seen.add(s_strip)

            # format 2: dict-like with keys
            elif isinstance(ev, dict):
                # try to extract 'section' and 'paragraph' keys
                sec = ev.get("section") or ev.get("section_name") or ev.get("sec")
                pidx = ev.get("paragraph") or ev.get("para_idx") or ev.get("p")
                if sec is None or pidx is None:
                    continue
                try:
                    pidx = int(pidx)
                except Exception:
                    continue
                sec_idx = find_section_idx(sec) if isinstance(sec, str) else None
                if sec_idx is None:
                    continue
                sec_paras = paragraphs_lists[sec_idx]
                if not isinstance(sec_paras, list) or pidx < 0 or pidx >= len(sec_paras):
                    continue
                paragraph_text = str(sec_paras[pidx]).strip()
                sents = re.split(r'(?<=[.!?])\s+', paragraph_text)
                for s in sents:
                    s_strip = s.strip()
                    if len(s_strip) > 3 and s_strip not in seen:
                        gold.append(s_strip)
                        seen.add(s_strip)

            # format 3: a raw string (maybe a sentence already)
            elif isinstance(ev, str) and len(ev.strip()) > 3:
                ss = ev.strip()
                if ss not in seen:
                    gold.append(ss)
                    seen.add(ss)

            # else: ignore unknown format

    return gold


# -------------------------
# Main evaluation runner
# -------------------------
def run_qasper_evaluation(qasper_path: str, num_papers=10, output_path="results.json"):
    print("=============================================")
    print("   QASPER SENTENCE-LEVEL EVALUATION PIPELINE ")
    print("=============================================")

    loader = QASPERLoader(qasper_path)
    paper_ids = loader.get_paper_ids()[:num_papers]
    print(f"[1/5] Loaded {len(paper_ids)} papers")

    # 1) create indexer and index all papers first
    indexer = SimpleIndexer()
    print("[2/5] Indexing papers...")
    for pid in tqdm(paper_ids):
        secs = loader.get_paper_sections(pid)
        indexer.index_paper(pid, secs)
    print(f"Indexed {len(indexer.chunks)} paragraph chunks")

    # 2) Now create scorer + retriever (after indexing)
    scorer = SimpleScorer(indexer)
    retriever = SimpleTier1Retriever(indexer, scorer)

    # 3) Sentence-level evaluator
    sent_eval = SentenceLevelEvaluator(loader, indexer, retriever, max_sentences_per_q=50)

    # Aggregation containers
    aggregated = {
        "recall@5": [], "recall@10": [],
        "precision@5": [], "precision@10": [],
        "hit_ratio@5": [], "hit_ratio@10": [],
        "ndcg@5": [], "ndcg@10": [],
        "mrr": [], "map": []
    }

    all_results = []

    print("[3/5] Running sentence-level evaluation...")
    # Evaluate paper-by-paper but call evaluate_question to pass converted gold sentences
    for pid in tqdm(paper_ids):
        questions = loader.get_questions(pid)
        per_q_results = []
        for q in questions:
            query = q.get("question", "")
            evidence_raw = q.get("evidence", [])
            extractive_spans = q.get("extractive_spans", [])

            gold_sents = evidence_to_sentences(loader, pid, evidence_raw, extractive_spans)
            # If no gold sentences found, try to fall back to extractive_spans only
            if len(gold_sents) == 0 and extractive_spans:
                for s in extractive_spans:
                    if s and s.strip():
                        gold_sents.append(s.strip())

            if len(gold_sents) == 0:
                # If still empty, skip this question (unanswerable or badly formatted)
                continue

            q_metrics = sent_eval.evaluate_question(query, gold_sents)
            if q_metrics is None:
                continue

            per_q_results.append({
                "question": query,
                "gold_count": len(gold_sents),
                **q_metrics
            })

            # accumulate aggregated lists
            for m in aggregated:
                aggregated[m].append(q_metrics.get(m, 0.0))

        all_results.append({
            "paper_id": pid,
            "num_questions": len(per_q_results),
            "per_question": per_q_results
        })

    print("[4/5] Computing final statistics...")
    metrics = {}
    for m, vals in aggregated.items():
        if len(vals) == 0:
            metrics[m] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        else:
            arr = np.array(vals)
            metrics[m] = {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "min": float(np.min(arr)), "max": float(np.max(arr))}

    final = {
        "overall": {
            "num_papers": num_papers,
            "num_chunks": len(indexer.chunks),
            "metrics": metrics
        },
        "per_paper": all_results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    print("[5/5] Done. Results saved to", output_path)
    # Print brief summary
    print("\nFINAL KILT METRICS SUMMARY")
    for m, s in metrics.items():
        print(f"{m:12} mean={s['mean']:.4f}  std={s['std']:.4f}")

    return final


if __name__ == "__main__":
    run_qasper_evaluation("qasper-dev-v0.3.json", num_papers=10)
