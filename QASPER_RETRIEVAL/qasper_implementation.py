"""
Complete Implementation with QASPER Dataset
Includes: Data loading, evaluation metrics, full pipeline
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import re
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import time
from tqdm import tqdm

# ============================================================================
# QASPER DATA LOADER
# ============================================================================

class QASPERLoader:
    """
    Loads and preprocesses QASPER dataset
    ACTUAL QASPER format (from Hugging Face):
    {
        "paper_id": {
            "title": str,
            "abstract": str,
            "full_text": {
                "section_name": ["para1", "para2", ...],
                ...
            },
            "qas": {
                "question_id": [{
                    "question": str,
                    "answers": [{"answer": {...}}]
                }],
                ...
            }
        }
    }
    """
    
    def __init__(self, qasper_path: str):
        self.qasper_path = qasper_path
        self.data = self._load_qasper()
    
    def _load_qasper(self) -> Dict:
        """Load QASPER JSON file"""
        with open(self.qasper_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_paper_ids(self) -> List[str]:
        """Get list of all paper IDs"""
        return list(self.data.keys())
    
    def get_paper_text(self, paper_id: str) -> str:
        """Get full paper text with section markers"""
        paper = self.data[paper_id]
        
        # Build full text with section headers
        full_text = f"# {paper['title']}\n\n"
        full_text += f"## Abstract\n{paper['abstract']}\n\n"
        
        # full_text is a dict: {section_name: [paragraphs]}
        for section_name, paragraphs in paper['full_text'].items():
            full_text += f"## {section_name}\n"
            if isinstance(paragraphs, list):
                for para in paragraphs:
                    full_text += f"{para}\n\n"
            else:
                full_text += f"{paragraphs}\n\n"
        
        return full_text
    
    def get_paper_sections(self, paper_id: str) -> List[Dict]:
        """Get structured sections for hierarchical parsing"""
        paper = self.data[paper_id]
        sections = []
        
        # Abstract
        sections.append({
            'title': 'Abstract',
            'content': paper['abstract'],
            'section_type': 'abstract'
        })
        
        # Full text - QASPER uses parallel lists: section_name and paragraphs
        full_text_dict = paper['full_text']
        
        section_names = full_text_dict.get('section_name', [])
        paragraphs_lists = full_text_dict.get('paragraphs', [])
        
        # Ensure both are lists
        if not isinstance(section_names, list):
            section_names = [section_names]
        if not isinstance(paragraphs_lists, list):
            paragraphs_lists = [paragraphs_lists]
        
        # Match sections to their paragraphs (parallel structure)
        for idx, section_name in enumerate(section_names):
            if idx < len(paragraphs_lists):
                section_paragraphs = paragraphs_lists[idx]
                
                # section_paragraphs is a list of paragraphs for this section
                if isinstance(section_paragraphs, list):
                    content = '\n\n'.join(str(p) for p in section_paragraphs if p)
                else:
                    content = str(section_paragraphs)
                
                sections.append({
                    'title': section_name,
                    'content': content,
                    'section_type': self._classify_section_type(section_name)
                })
        
        return sections
    
    def get_questions(self, paper_id: str) -> List[Dict]:
        """Get all questions for a paper with ground truth"""
        paper = self.data[paper_id]
        questions = []
        
        # ACTUAL QASPER structure: qas is a dict with parallel lists
        # {'question': [q1, q2, ...], 'question_id': [id1, id2, ...], 'answers': [[ans1], [ans2], ...]}
        qas_dict = paper['qas']
        
        # Get lists
        question_texts = qas_dict.get('question', [])
        question_ids = qas_dict.get('question_id', [])
        answers_list = qas_dict.get('answers', [])
        
        # Ensure we have matching lengths
        num_questions = len(question_texts)
        if len(question_ids) < num_questions:
            question_ids.extend([f'q{i}' for i in range(len(question_ids), num_questions)])
        if len(answers_list) < num_questions:
            answers_list.extend([[] for _ in range(len(answers_list), num_questions)])
        
        # Process each question
        for idx in range(num_questions):
            question_text = question_texts[idx]
            question_id = question_ids[idx] if idx < len(question_ids) else f'q{idx}'
            answers = answers_list[idx] if idx < len(answers_list) else []
            
            # Get answer details
            evidence = []
            extractive_spans = []
            free_form = ''
            
            # Check if answers is a dict (QASPER format)
            if isinstance(answers, dict) and 'answer' in answers:
                answer_annotations = answers['answer']
                
                # answer_annotations is a list of annotation dicts
                if isinstance(answer_annotations, list) and len(answer_annotations) > 0:
                    # Take first annotation
                    answer = answer_annotations[0]
                    
                    # Skip unanswerable
                    if isinstance(answer, dict) and answer.get('unanswerable', False):
                        continue
                    
                    # Extract evidence and spans - they're directly in answer dict
                    if isinstance(answer, dict):
                        evidence = answer.get('evidence', [])
                        extractive_spans = answer.get('extractive_spans', [])
                        free_form = answer.get('free_form_answer', '')
            elif isinstance(answers, list) and len(answers) > 0:
                # Fallback: answers is directly a list of dicts
                first_answer_dict = answers[0]
                
                if isinstance(first_answer_dict, dict) and 'answer' in first_answer_dict:
                    answer_annotations = first_answer_dict['answer']
                    
                    if isinstance(answer_annotations, list) and len(answer_annotations) > 0:
                        answer = answer_annotations[0]
                        
                        if isinstance(answer, dict) and answer.get('unanswerable', False):
                            continue
                        
                        if isinstance(answer, dict):
                            evidence = answer.get('evidence', [])
                            extractive_spans = answer.get('extractive_spans', [])
                            free_form = answer.get('free_form_answer', '')
            
            # Ensure lists
            if not isinstance(evidence, list):
                evidence = [evidence] if evidence else []
            if not isinstance(extractive_spans, list):
                extractive_spans = [extractive_spans] if extractive_spans else []
            
            questions.append({
                'question': question_text,
                'evidence': evidence,
                'extractive_spans': extractive_spans,
                'free_form_answer': free_form,
                'question_id': question_id
            })
        
        return questions
    
    def _classify_section_type(self, section_name: str) -> str:
        """Classify section type from name"""
        name_lower = section_name.lower()
        
        if 'abstract' in name_lower:
            return 'abstract'
        elif any(kw in name_lower for kw in ['method', 'approach', 'model', 'experiment']):
            return 'methodology'
        elif any(kw in name_lower for kw in ['result', 'finding', 'evaluation', 'performance']):
            return 'results'
        elif any(kw in name_lower for kw in ['conclusion', 'discussion', 'future']):
            return 'conclusion'
        elif any(kw in name_lower for kw in ['introduction', 'background', 'related']):
            return 'introduction'
        else:
            return 'body'


# ============================================================================
# SIMPLIFIED COMPONENTS (adapted from previous artifacts)
# ============================================================================

@dataclass
class Chunk:
    id: str
    content: str
    doc_id: str
    section_name: str
    section_type: str
    position: float
    token_count: int
    paragraph_idx: int  # For tracking evidence matching

class SimpleIndexer:
    """Simplified indexer for QASPER (no LLM dependency for summaries)"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.chunks: Dict[str, Chunk] = {}
        self.documents: Dict[str, Dict] = {}
        self.chunk_embeddings = []
        self.chunk_id_to_idx = {}
    
    def index_paper(self, paper_id: str, sections: List[Dict], 
                    loader: QASPERLoader) -> None:
        """Index a QASPER paper"""
        
        all_chunks = []
        para_counter = 0
        
        for section_idx, section in enumerate(sections):
            section_title = section['title']
            section_content = section['content']
            
            # Split section into paragraphs
            paragraphs = section_content.split('\n\n')
            
            for para_idx, para in enumerate(paragraphs):
                para = para.strip()
                if len(para) < 50:  # Skip tiny paragraphs
                    continue
                
                chunk_id = f"{paper_id}_s{section_idx}_p{para_idx}"
                
                chunk = Chunk(
                    id=chunk_id,
                    content=para,
                    doc_id=paper_id,
                    section_name=section_title,  # Use actual section title
                    section_type=section['section_type'],
                    position=section_idx / len(sections),
                    token_count=len(para.split()),
                    paragraph_idx=para_counter
                )
                
                all_chunks.append(chunk)
                self.chunks[chunk_id] = chunk
                para_counter += 1
        
        # Store document metadata
        paper_data = loader.data[paper_id]
        self.documents[paper_id] = {
            'title': paper_data['title'],
            'abstract': paper_data['abstract'],
            'total_chunks': len(all_chunks)
        }
        
        # Encode chunks
        chunk_texts = [c.content for c in all_chunks]
        embeddings = self.encoder.encode(chunk_texts, 
                                        show_progress_bar=False,
                                        batch_size=32)
        
        # Store embeddings with mapping
        start_idx = len(self.chunk_embeddings)
        for i, chunk in enumerate(all_chunks):
            self.chunk_id_to_idx[chunk.id] = start_idx + i
        
        if len(self.chunk_embeddings) == 0:
            self.chunk_embeddings = embeddings
        else:
            self.chunk_embeddings = np.vstack([self.chunk_embeddings, embeddings])


class SimpleScorer:
    """Simplified multi-signal scorer"""
    
    def __init__(self, indexer: SimpleIndexer):
        self.indexer = indexer
        
        # Initialize BM25
        corpus = [chunk.content for chunk in indexer.chunks.values()]
        tokenized = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)
        
        # Weights (tuned for QASPER)
        self.weights = {
            'semantic': 0.30,
            'lexical': 0.20,
            'structural': 0.25,
            'position': 0.15,
            'section_type': 0.10
        }
        
        self.section_weights = {
            'abstract': 1.4,
            'introduction': 1.1,
            'methodology': 1.2,
            'results': 1.3,
            'conclusion': 1.3,
            'body': 1.0
        }
    
    def score_chunks(self, query: str, query_embedding: np.ndarray,
                    chunk_ids: List[str]) -> Dict[str, float]:
        """Compute multi-signal scores"""
        
        # Semantic scores
        semantic_scores = {}
        for cid in chunk_ids:
            idx = self.indexer.chunk_id_to_idx[cid]
            chunk_emb = self.indexer.chunk_embeddings[idx]
            sim = np.dot(query_embedding, chunk_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb)
            )
            semantic_scores[cid] = float(sim)
        
        # Lexical scores (BM25)
        tokenized_query = query.lower().split()
        all_bm25_scores = self.bm25.get_scores(tokenized_query)
        
        lexical_scores = {}
        for cid in chunk_ids:
            idx = self.indexer.chunk_id_to_idx[cid]
            lexical_scores[cid] = all_bm25_scores[idx]
        
        # Normalize lexical
        max_lex = max(lexical_scores.values()) if lexical_scores else 1.0
        lexical_scores = {k: v/max_lex for k, v in lexical_scores.items()}
        
        # Structural scores
        structural_scores = {}
        for cid in chunk_ids:
            chunk = self.indexer.chunks[cid]
            section_weight = self.section_weights.get(chunk.section_type, 1.0)
            structural_scores[cid] = section_weight
        
        # Position scores (U-shaped)
        position_scores = {}
        for cid in chunk_ids:
            chunk = self.indexer.chunks[cid]
            pos = chunk.position
            distance_from_edge = min(pos, 1.0 - pos)
            position_scores[cid] = 1.0 - 0.4 * (2 * distance_from_edge) ** 2
        
        # Section type scores
        section_type_scores = {}
        for cid in chunk_ids:
            chunk = self.indexer.chunks[cid]
            section_type_scores[cid] = self.section_weights.get(chunk.section_type, 1.0)
        
        # Combine
        final_scores = {}
        for cid in chunk_ids:
            score = (
                self.weights['semantic'] * semantic_scores[cid] +
                self.weights['lexical'] * lexical_scores[cid] +
                self.weights['structural'] * structural_scores[cid] +
                self.weights['position'] * position_scores[cid] +
                self.weights['section_type'] * section_type_scores[cid]
            )
            final_scores[cid] = score
        
        return final_scores


class SimpleTier1Retriever:
    """Fast budget-aware retrieval"""
    
    def __init__(self, indexer: SimpleIndexer, scorer: SimpleScorer,
                 token_budget: int = 4000):
        self.indexer = indexer
        self.scorer = scorer
        self.token_budget = token_budget
    
    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        """Retrieve top-k chunks within budget"""
        
        # Encode query
        query_embedding = self.indexer.encoder.encode([query])[0]
        
        # Get initial candidates (top-50 by semantic similarity)
        similarities = np.dot(
            self.indexer.chunk_embeddings,
            query_embedding
        ) / (
            np.linalg.norm(self.indexer.chunk_embeddings, axis=1) *
            np.linalg.norm(query_embedding)
        )
        
        top_50_indices = np.argsort(similarities)[::-1][:50]
        all_chunk_ids = list(self.indexer.chunks.keys())
        candidates = [all_chunk_ids[idx] for idx in top_50_indices]
        
        # Multi-signal scoring
        scores = self.scorer.score_chunks(query, query_embedding, candidates)
        
        # Budget-aware selection (greedy by efficiency)
        efficiency = {
            cid: scores[cid] / max(self.indexer.chunks[cid].token_count, 1)
            for cid in candidates
        }
        
        sorted_chunks = sorted(efficiency.keys(), 
                              key=lambda x: efficiency[x],
                              reverse=True)
        
        selected = []
        total_tokens = 0
        
        for chunk_id in sorted_chunks:
            chunk = self.indexer.chunks[chunk_id]
            if total_tokens + chunk.token_count <= self.token_budget:
                selected.append(chunk_id)
                total_tokens += chunk.token_count
                
                if len(selected) >= top_k:
                    break
        
        # Sort by original score
        selected.sort(key=lambda x: scores[x], reverse=True)
        
        return selected


# ============================================================================
# EVALUATION METRICS
# ============================================================================

class QASPEREvaluator:
    """
    Evaluates retrieval quality on QASPER
    """
    
    def __init__(self, indexer: SimpleIndexer, loader: QASPERLoader):
        self.indexer = indexer
        self.loader = loader
    
    def compute_recall_at_k(self, retrieved_chunks: List[str],
                           evidence: List[str], k: int = 5) -> float:
        """
        Recall@k: What fraction of evidence chunks were retrieved in top-k?
        """
        if not evidence or len(evidence) == 0:
            # No evidence needed - return 1.0 (vacuously true)
            return 1.0
        
        retrieved_texts = [
            self.indexer.chunks[cid].content 
            for cid in retrieved_chunks[:k]
        ]
        
        # Count how many evidence spans are found
        found = 0
        for ev_text in evidence:
            # Fuzzy match: check if evidence is substantially in retrieved chunks
            for ret_text in retrieved_texts:
                if self._substantial_overlap(ev_text, ret_text):
                    found += 1
                    break  # Found this evidence, move to next
        
        return found / len(evidence)
    
    def compute_precision_at_k(self, retrieved_chunks: List[str],
                              evidence: List[str], k: int = 5) -> float:
        """
        Precision@k: What fraction of top-k chunks contain evidence?
        """
        if not evidence or len(evidence) == 0:
            # If no evidence expected, precision is undefined - return 1.0 for fairness
            return 1.0
        
        retrieved_texts = [
            self.indexer.chunks[cid].content 
            for cid in retrieved_chunks[:k]
        ]
        
        relevant = 0
        for ret_text in retrieved_texts:
            for ev_text in evidence:
                if self._substantial_overlap(ev_text, ret_text):
                    relevant += 1
                    break  # Count chunk only once
        
        return relevant / k if k > 0 else 0.0
    
    def compute_mrr(self, retrieved_chunks: List[str],
                   evidence: List[str]) -> float:
        """
        Mean Reciprocal Rank: 1/rank of first relevant chunk
        """
        if not evidence:
            return 0.0
        
        for rank, chunk_id in enumerate(retrieved_chunks, 1):
            chunk_text = self.indexer.chunks[chunk_id].content
            for ev_text in evidence:
                if self._substantial_overlap(ev_text, chunk_text):
                    return 1.0 / rank
        
        return 0.0  # No relevant chunk found
    
    def compute_answer_coverage(self, retrieved_chunks: List[str],
                                answer_spans: List[str]) -> float:
        """
        What fraction of answer spans appear in retrieved chunks?
        """
        if not answer_spans:
            return 1.0
        
        retrieved_texts = [
            self.indexer.chunks[cid].content 
            for cid in retrieved_chunks
        ]
        combined_text = ' '.join(retrieved_texts).lower()
        
        found = 0
        for span in answer_spans:
            if span.lower() in combined_text:
                found += 1
        
        return found / len(answer_spans)
    
    def _substantial_overlap(self, text1: str, text2: str, 
                           threshold: float = 0.3) -> bool:
        """
        Check if two texts have substantial word overlap
        Uses bidirectional overlap - checks both directions
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return False
        
        # Bidirectional overlap - check both ways
        overlap1 = len(words1 & words2) / len(words1)  # Evidence → Chunk
        overlap2 = len(words1 & words2) / len(words2)  # Chunk → Evidence
        
        # Match if either direction exceeds threshold
        return overlap1 >= threshold or overlap2 >= threshold
    
    def evaluate_paper(self, paper_id: str, retriever,
                      k_values: List[int] = [5, 10]) -> Dict:
        """Evaluate retrieval for all questions in a paper"""
        
        questions = self.loader.get_questions(paper_id)
        
        results = {
            'paper_id': paper_id,
            'num_questions': len(questions),
            'per_question': [],
            'aggregated': {}
        }
        
        all_recalls = {k: [] for k in k_values}
        all_precisions = {k: [] for k in k_values}
        all_mrrs = []
        all_coverages = []
        
        for q_data in questions:
            query = q_data['question']
            evidence = q_data['evidence']
            answer_spans = q_data['extractive_spans']
            
            # Retrieve
            retrieved = retriever.retrieve(query, top_k=max(k_values))
            
            # Compute metrics
            q_results = {
                'question': query,
                'retrieved_count': len(retrieved)
            }
            
            for k in k_values:
                recall = self.compute_recall_at_k(retrieved, evidence, k)
                precision = self.compute_precision_at_k(retrieved, evidence, k)
                
                q_results[f'recall@{k}'] = recall
                q_results[f'precision@{k}'] = precision
                
                all_recalls[k].append(recall)
                all_precisions[k].append(precision)
            
            mrr = self.compute_mrr(retrieved, evidence)
            coverage = self.compute_answer_coverage(retrieved, answer_spans)
            
            q_results['mrr'] = mrr
            q_results['answer_coverage'] = coverage
            
            all_mrrs.append(mrr)
            all_coverages.append(coverage)
            
            results['per_question'].append(q_results)
        
        # Aggregate
        for k in k_values:
            results['aggregated'][f'recall@{k}'] = np.mean(all_recalls[k])
            results['aggregated'][f'precision@{k}'] = np.mean(all_precisions[k])
        
        results['aggregated']['mrr'] = np.mean(all_mrrs)
        results['aggregated']['answer_coverage'] = np.mean(all_coverages)
        
        return results


# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

def run_qasper_evaluation(qasper_path: str, 
                         num_papers: int = 10,
                         output_path: str = 'results.json'):
    """
    Complete evaluation pipeline
    """
    
    print("="*60)
    print("QASPER EVALUATION PIPELINE")
    print("="*60)
    
    # Load data
    print("\n[1/5] Loading QASPER dataset...")
    loader = QASPERLoader(qasper_path)
    paper_ids = loader.get_paper_ids()[:num_papers]
    print(f"Loaded {len(paper_ids)} papers")
    
    # Initialize components
    print("\n[2/5] Initializing indexer and scorer...")
    indexer = SimpleIndexer()
    
    # Index papers
    print("\n[3/5] Indexing papers...")
    for paper_id in tqdm(paper_ids, desc="Indexing"):
        sections = loader.get_paper_sections(paper_id)
        indexer.index_paper(paper_id, sections, loader)
    
    print(f"Indexed {len(indexer.chunks)} chunks total")
    
    scorer = SimpleScorer(indexer)
    retriever = SimpleTier1Retriever(indexer, scorer, token_budget=4000)
    
    # Initialize evaluator
    print("\n[4/5] Running evaluation...")
    evaluator = QASPEREvaluator(indexer, loader)
    
    # Evaluate each paper
    all_results = []
    aggregate_metrics = {
        'recall@5': [],
        'recall@10': [],
        'precision@5': [],
        'precision@10': [],
        'mrr': [],
        'answer_coverage': []
    }
    
    for paper_id in tqdm(paper_ids, desc="Evaluating"):
        paper_results = evaluator.evaluate_paper(paper_id, retriever, k_values=[5, 10])
        all_results.append(paper_results)
        
        # Collect for overall aggregation
        agg = paper_results['aggregated']
        for metric in aggregate_metrics.keys():
            aggregate_metrics[metric].append(agg[metric])
    
    # Overall statistics
    print("\n[5/5] Computing final statistics...")
    
    overall = {
        'num_papers': num_papers,
        'num_chunks_total': len(indexer.chunks),
        'metrics': {
            metric: {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
            for metric, values in aggregate_metrics.items()
        }
    }
    
    # Save results
    final_results = {
        'overall': overall,
        'per_paper': all_results,
        'config': {
            'token_budget': 4000,
            'top_k': 10,
            'model': 'all-MiniLM-L6-v2'
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    for metric, stats in overall['metrics'].items():
        print(f"\n{metric.upper()}")
        print(f"  Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
        print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    
    print(f"\nResults saved to: {output_path}")
    print("="*60)
    
    return final_results


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Download QASPER from: https://github.com/allenai/qasper
    # Expected format: qasper-train-v0.3.json or qasper-dev-v0.3.json
    
    QASPER_PATH = "qasper-dev-v0.3.json"  # Update this path
    
    results = run_qasper_evaluation(
        qasper_path=QASPER_PATH,
        num_papers=10,  # Start with 10 for quick testing
        output_path='qasper_tier1_results.json'
    )