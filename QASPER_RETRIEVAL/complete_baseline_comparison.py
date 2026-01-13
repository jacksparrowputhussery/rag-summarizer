"""
Complete comparison with all standard baselines for research paper
Includes: BM25, Dense Retrieval, Standard RAG, and advanced methods
"""

import numpy as np
from typing import List, Dict
import json
from tqdm import tqdm
from scipy import stats
import pandas as pd

# Import existing implementations
from qasper_implementation import (
    QASPERLoader, SimpleIndexer, SimpleScorer, 
    QASPEREvaluator, SimpleTier1Retriever
)
from baseline_comparison import BaselineRAG, SemanticPlusLexicalRetriever


# ============================================================================
# BASELINE 1: BM25 ONLY (Traditional IR)
# ============================================================================

class BM25OnlyRetriever:
    """Pure lexical retrieval using BM25"""
    
    def __init__(self, indexer, scorer):
        self.indexer = indexer
        self.scorer = scorer
    
    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        """BM25-only retrieval"""
        tokenized_query = query.lower().split()
        all_bm25_scores = self.scorer.bm25.get_scores(tokenized_query)
        
        # Get top-k by BM25 score
        all_chunk_ids = list(self.indexer.chunks.keys())
        top_k_indices = np.argsort(all_bm25_scores)[::-1][:top_k]
        
        return [all_chunk_ids[idx] for idx in top_k_indices]


# ============================================================================
# BASELINE 2: Dense Retrieval (Simulates DPR)
# ============================================================================

class DenseRetrievalBaseline:
    """
    Dense retrieval baseline (similar to DPR)
    Uses same embeddings as baseline but with different normalization
    """
    
    def __init__(self, indexer):
        self.indexer = indexer
    
    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        """Dense retrieval with L2 normalization"""
        query_embedding = self.indexer.encoder.encode([query])[0]
        
        # L2 normalize (common in DPR)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        normalized_embeddings = self.indexer.chunk_embeddings / np.linalg.norm(
            self.indexer.chunk_embeddings, axis=1, keepdims=True
        )
        
        # Dot product (equivalent to cosine for normalized vectors)
        similarities = np.dot(normalized_embeddings, query_embedding)
        
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        all_chunk_ids = list(self.indexer.chunks.keys())
        
        return [all_chunk_ids[idx] for idx in top_k_indices]


# ============================================================================
# BASELINE 3: Hybrid (BM25 + Dense with learned weights)
# ============================================================================

class HybridRetriever:
    """
    Hybrid retrieval (BM25 + Dense)
    Common in modern systems (Elasticsearch, Weaviate)
    """
    
    def __init__(self, indexer, scorer, alpha=0.5):
        self.indexer = indexer
        self.scorer = scorer
        self.alpha = alpha  # Weight for semantic vs lexical
    
    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        """Hybrid retrieval"""
        query_embedding = self.indexer.encoder.encode([query])[0]
        
        # Semantic scores
        similarities = np.dot(
            self.indexer.chunk_embeddings, query_embedding
        ) / (
            np.linalg.norm(self.indexer.chunk_embeddings, axis=1) *
            np.linalg.norm(query_embedding)
        )
        
        # Lexical scores
        tokenized_query = query.lower().split()
        bm25_scores = self.scorer.bm25.get_scores(tokenized_query)
        
        # Normalize both to [0, 1]
        sem_norm = (similarities - similarities.min()) / (similarities.max() - similarities.min() + 1e-10)
        bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-10)
        
        # Combine
        hybrid_scores = self.alpha * sem_norm + (1 - self.alpha) * bm25_norm
        
        top_k_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        all_chunk_ids = list(self.indexer.chunks.keys())
        
        return [all_chunk_ids[idx] for idx in top_k_indices]


    #Test if multi-signal actually helps
class SemanticOnlyBudgetAware:
    """
    Budget-aware retrieval using ONLY semantic similarity
    Tests if multi-signal scoring actually helps or if budget is doing all the work
    """
    
    def __init__(self, indexer, token_budget=4000):
        self.indexer = indexer
        self.token_budget = token_budget
        print(f"[SemanticOnlyBudgetAware] Initialized with budget={token_budget}")
    
    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        """
        Retrieve chunks using:
        1. Pure semantic similarity (cosine)
        2. Budget-aware selection (knapsack optimization)
        """
        
        # Step 1: Encode query
        query_embedding = self.indexer.encoder.encode([query])[0]
        
        # Step 2: Compute semantic similarity with ALL chunks
        similarities = np.dot(
            self.indexer.chunk_embeddings,
            query_embedding
        ) / (
            np.linalg.norm(self.indexer.chunk_embeddings, axis=1) *
            np.linalg.norm(query_embedding)
        )
        
        # Step 3: Get top-50 candidates (initial filtering)
        top_50_indices = np.argsort(similarities)[::-1][:50]
        all_chunk_ids = list(self.indexer.chunks.keys())
        candidates = [all_chunk_ids[idx] for idx in top_50_indices]
        
        # Step 4: Create score dictionary
        scores = {}
        for i, chunk_id in enumerate(candidates):
            scores[chunk_id] = similarities[top_50_indices[i]]
        
        # Step 5: Compute efficiency ratio (score per token)
        efficiency = {}
        for chunk_id in candidates:
            chunk = self.indexer.chunks[chunk_id]
            token_count = chunk.token_count
            
            # Avoid division by zero
            if token_count == 0:
                token_count = 1
            
            efficiency[chunk_id] = scores[chunk_id] / token_count
        
        # Step 6: Sort by efficiency (greedy knapsack)
        sorted_chunks = sorted(
            candidates,
            key=lambda x: efficiency[x],
            reverse=True
        )
        
        # Step 7: Budget-aware selection
        selected = []
        total_tokens = 0
        
        for chunk_id in sorted_chunks:
            chunk = self.indexer.chunks[chunk_id]
            
            # Check if adding this chunk would exceed budget
            if total_tokens + chunk.token_count <= self.token_budget:
                selected.append(chunk_id)
                total_tokens += chunk.token_count
                
                # Optional: early stopping if we have enough high-quality chunks
                if len(selected) >= top_k and scores[chunk_id] < 0.5:
                    break
        
        # Step 8: Rerank selected chunks by original semantic score (not efficiency)
        selected.sort(key=lambda x: scores[x], reverse=True)
        
        # Debug output (optional - comment out for production)
        # print(f"  Selected {len(selected)} chunks, {total_tokens}/{self.token_budget} tokens")
        
        return selected


# ============================================================================
# COMPREHENSIVE EVALUATION
# ============================================================================

def run_complete_comparison(qasper_path: str, num_papers: int = 10):
    """
    Complete comparison with all baselines
    """
    
    print("="*80)
    print("COMPREHENSIVE BASELINE COMPARISON FOR RESEARCH PAPER")
    print("="*80)
    
    # Setup
    loader = QASPERLoader(qasper_path)
    paper_ids = loader.get_paper_ids()[:num_papers]
    
    print(f"\n[Setup] Indexing {num_papers} papers...")
    indexer = SimpleIndexer()
    for paper_id in tqdm(paper_ids, desc="Indexing"):
        sections = loader.get_paper_sections(paper_id)
        indexer.index_paper(paper_id, sections, loader)
    
    scorer = SimpleScorer(indexer)
    evaluator = QASPEREvaluator(indexer, loader)
    
    # Initialize all retrievers
    retrievers = {
        '1. BM25 (Lexical Only)': BM25OnlyRetriever(indexer, scorer),
        '2. Dense Retrieval (DPR-style)': DenseRetrievalBaseline(indexer),
        '3. Hybrid (α=0.5)': HybridRetriever(indexer, scorer, alpha=0.5),
        '4. Standard RAG (Semantic Only)': BaselineRAG(indexer),
        '5. Semantic + Lexical (0.6/0.4)': SemanticPlusLexicalRetriever(indexer, scorer),
        '6. Multi-Signal (No Budget)': FullMultiSignalRetriever(indexer, scorer),
        '7. Budget-Aware (Ours)': SimpleTier1Retriever(indexer, scorer, token_budget=4000),
        '8. Semantic + Budget (Test)': SemanticOnlyBudgetAware(indexer, token_budget=4000),  # NEW!
    }
    
    # Store all raw results for statistical tests
    all_raw_results = {name: {
        'recall@5': [], 'recall@10': [],
        'precision@5': [], 'precision@10': [],
        'mrr': [], 'answer_coverage': [], 'tokens': []
    } for name in retrievers.keys()}
    
    # Evaluate each retriever
    print(f"\n[Evaluation] Testing {len(retrievers)} methods...")
    
    for name, retriever in retrievers.items():
        print(f"\n  Evaluating: {name}")
        
        for paper_id in tqdm(paper_ids, desc=f"  {name[:20]}", leave=False):
            questions = loader.get_questions(paper_id)
            
            for q_data in questions:
                query = q_data['question']
                evidence = q_data['evidence']
                answer_spans = q_data['extractive_spans']
                
                # Retrieve
                retrieved = retriever.retrieve(query, top_k=10)
                
                # Compute metrics
                all_raw_results[name]['recall@5'].append(
                    evaluator.compute_recall_at_k(retrieved, evidence, k=5)
                )
                all_raw_results[name]['recall@10'].append(
                    evaluator.compute_recall_at_k(retrieved, evidence, k=10)
                )
                all_raw_results[name]['precision@5'].append(
                    evaluator.compute_precision_at_k(retrieved, evidence, k=5)
                )
                all_raw_results[name]['precision@10'].append(
                    evaluator.compute_precision_at_k(retrieved, evidence, k=10)
                )
                all_raw_results[name]['mrr'].append(
                    evaluator.compute_mrr(retrieved, evidence)
                )
                all_raw_results[name]['answer_coverage'].append(
                    evaluator.compute_answer_coverage(retrieved, answer_spans)
                )
                
                # Token count
                tokens = sum(indexer.chunks[cid].token_count for cid in retrieved)
                all_raw_results[name]['tokens'].append(tokens)
    
    # Aggregate results
    aggregated = {}
    for name in retrievers.keys():
        aggregated[name] = {
            metric: {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
            for metric, values in all_raw_results[name].items()
        }
    
    # Statistical significance tests (compare each to baseline)
    baseline_name = '4. Standard RAG (Semantic Only)'
    significance = {}
    
    print("\n[Statistical Tests] Computing p-values...")
    for name in retrievers.keys():
        if name == baseline_name:
            continue
        
        significance[name] = {}
        for metric in ['recall@10', 'precision@5', 'mrr']:
            baseline_scores = all_raw_results[baseline_name][metric]
            method_scores = all_raw_results[name][metric]
            
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(method_scores, baseline_scores)
            significance[name][metric] = {
                't_stat': float(t_stat),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05)
            }
    
    # Print results table
    print_results_table(aggregated, baseline_name, significance)
    
    # Save results
    output = {
        'aggregated': aggregated,
        'significance': significance,
        'raw_results': all_raw_results,
        'num_papers': num_papers,
        'num_questions': sum(len(loader.get_questions(pid)) for pid in paper_ids),
        'baseline': baseline_name
    }
    
    with open('complete_comparison_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    # Create publication-ready tables
    create_latex_tables(aggregated, baseline_name, significance, output)
    
    return output


def print_results_table(aggregated, baseline_name, significance):
    """Print formatted results table"""
    
    print("\n" + "="*100)
    print("MAIN RESULTS TABLE (for paper)")
    print("="*100)
    
    # Header
    print(f"\n{'Method':<35} | {'R@5':<8} | {'R@10':<8} | {'P@5':<8} | {'P@10':<8} | {'MRR':<8} | {'Tokens':<8} | {'Eff.':<6}")
    print("-" * 100)
    
    # Baseline first
    baseline_r10 = aggregated[baseline_name]['recall@10']['mean']
    baseline_tokens = aggregated[baseline_name]['tokens']['mean']
    
    for name in sorted(aggregated.keys()):
        r5 = aggregated[name]['recall@5']['mean']
        r10 = aggregated[name]['recall@10']['mean']
        p5 = aggregated[name]['precision@5']['mean']
        p10 = aggregated[name]['precision@10']['mean']
        mrr = aggregated[name]['mrr']['mean']
        tokens = aggregated[name]['tokens']['mean']
        efficiency = (r10 / tokens) * 1000  # Recall per 1K tokens
        
        # Mark significance
        sig_markers = ""
        if name != baseline_name and name in significance:
            if significance[name]['recall@10']['significant']:
                sig_markers += "*" if significance[name]['recall@10']['p_value'] < 0.05 else ""
                sig_markers += "*" if significance[name]['recall@10']['p_value'] < 0.01 else ""
        
        # Bold if best
        bold_r10 = "**" if r10 == max(agg['recall@10']['mean'] for agg in aggregated.values()) else ""
        bold_eff = "**" if efficiency == max((agg['recall@10']['mean']/agg['tokens']['mean'])*1000 for agg in aggregated.values()) else ""
        
        print(f"{name:<35} | {r5:.3f}    | {bold_r10}{r10:.3f}{bold_r10}{sig_markers:<2} | {p5:.3f}    | {p10:.3f}    | {mrr:.3f}    | {tokens:>6.0f}  | {bold_eff}{efficiency:.2f}{bold_eff}")
    
    print("-" * 100)
    print("\n* p < 0.05, ** p < 0.01 (vs Standard RAG baseline)")
    print("Eff. = Recall@10 per 1000 tokens (higher is better)")
    print("Bold indicates best performance")


def create_latex_tables(aggregated, baseline_name, significance, full_output):
    """Generate LaTeX tables for paper"""
    
    latex = []
    
    # Main results table
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Main Results on QASPER Dataset}")
    latex.append("\\label{tab:main_results}")
    latex.append("\\begin{tabular}{l|ccc|cc|c}")
    latex.append("\\toprule")
    latex.append("Method & R@5 & R@10 & P@5 & P@10 & MRR & Tokens \\\\")
    latex.append("\\midrule")
    
    for name in sorted(aggregated.keys()):
        clean_name = name.split('. ', 1)[1] if '. ' in name else name
        r5 = aggregated[name]['recall@5']['mean']
        r10 = aggregated[name]['recall@10']['mean']
        p5 = aggregated[name]['precision@5']['mean']
        p10 = aggregated[name]['precision@10']['mean']
        mrr = aggregated[name]['mrr']['mean']
        tokens = aggregated[name]['tokens']['mean']
        
        # Add significance markers
        sig = ""
        if name != baseline_name and name in significance:
            if significance[name]['recall@10']['p_value'] < 0.01:
                sig = "$^{**}$"
            elif significance[name]['recall@10']['p_value'] < 0.05:
                sig = "$^{*}$"
        
        latex.append(f"{clean_name} & {r5:.3f} & {r10:.3f}{sig} & {p5:.3f} & {p10:.3f} & {mrr:.3f} & {tokens:.0f} \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    # Save
    with open('latex_main_results.tex', 'w') as f:
        f.write('\n'.join(latex))
    
    print("\n✓ LaTeX table saved to: latex_main_results.tex")


# ============================================================================
# Missing imports
# ============================================================================

class FullMultiSignalRetriever:
    """All signals but no budget awareness (from baseline_comparison.py)"""
    
    def __init__(self, indexer, scorer):
        self.indexer = indexer
        self.scorer = scorer
    
    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        query_embedding = self.indexer.encoder.encode([query])[0]
        
        # Get candidates
        similarities = np.dot(self.indexer.chunk_embeddings, query_embedding) / (
            np.linalg.norm(self.indexer.chunk_embeddings, axis=1) *
            np.linalg.norm(query_embedding)
        )
        top_50_indices = np.argsort(similarities)[::-1][:50]
        all_chunk_ids = list(self.indexer.chunks.keys())
        candidates = [all_chunk_ids[idx] for idx in top_50_indices]
        
        # Full multi-signal scoring
        scores = self.scorer.score_chunks(query, query_embedding, candidates)
        
        # Return top-k (no budget awareness)
        sorted_chunks = sorted(scores.keys(),
                              key=lambda x: scores[x],
                              reverse=True)
        return sorted_chunks[:top_k]


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    QASPER_PATH = "qasper-dev-v0.3.json"
    
    results = run_complete_comparison(QASPER_PATH, num_papers=200)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print("\nFiles generated:")
    print("  - complete_comparison_results.json (all raw data)")
    print("  - latex_main_results.tex (table for paper)")
    print("\nUse these results in your research paper!")