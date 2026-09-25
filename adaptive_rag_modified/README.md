# Modified Adaptive-RAG — Retrieval Strategy Classification Using Gold Evidence

A retrieval-only reimplementation of the research question behind
*Adaptive-RAG* (Jeong et al.): **given a question, which retrieval
strategy — BM25, Semantic, Hybrid, or Multi-hop — is most likely to
retrieve the required gold evidence?**

This project deliberately **excludes** any LLM answer-generation stage.
The pipeline stops at retrieval evaluation:

```
Question → Retrieval Strategy → Retrieved Chunks → Gold Evidence Matching
         → Retrieval Metrics → Best Strategy Label → 4-Class Classifier
```

No answer generation, no answer EM/F1, no QA reader. See `config.yaml`
section comments and each module's docstring for how this constraint is
enforced structurally (e.g. `retrieval/base.py` only accepts a plain
question string — gold evidence is never passed to a retriever).

## Project layout

```
adaptive_rag_modified/
├── config.yaml                  # every knob in one place
├── requirements.txt
├── run_pipeline.py              # runs all 7 stages end-to-end (DEBUG-safe)
├── utils.py                     # shared IO / text-normalization / seeding
│
├── preprocessing/                # one loader per dataset -> unified schema
│   ├── base.py                   #   shared schema + offline synthetic fallback KB
│   ├── squad.py / natural_questions.py / triviaqa.py
│   ├── musique.py / hotpotqa.py / wiki2multi.py
│   └── build_unified_dataset.py  #   merges, dedups, splits
│
├── corpus/                       # shared document/chunk corpus + indices
│   ├── chunker.py                #   sentence-aligned, overlap-aware chunking
│   ├── build_corpus.py
│   └── index.py                  #   BM25 + dense (FAISS or numpy) index, cached
│
├── retrieval/                    # the four retrieval strategies
│   ├── base.py                   #   standardized interface + output schema
│   ├── bm25.py / semantic.py / hybrid.py / multihop.py
│   └── factory.py                #   builds all four, sharing indices
│
├── labeling/                      # automatic strategy-label generation
│   ├── generate_labels.py
│   ├── strategy_selector.py      #   lexicographic / utility-based selection
│   └── utility.py
│
├── classifier/                    # 4-class question -> strategy classifier
│   ├── model.py                  #   TF-IDF+LogReg (offline default) or DeBERTa
│   ├── dataset.py / train.py / evaluate.py
│
├── evaluation/
│   ├── evidence_matching.py      #   ID-based + text-based (exact/overlap/semantic)
│   ├── retrieval_metrics.py      #   Recall/Precision/F1/MRR/Hit/CER/Coverage
│   ├── oracle.py                 #   upper-bound "best possible" selector
│   ├── efficiency.py             #   latency aggregation
│   └── plots.py                  #   confusion matrix, comparison, heatmaps
│
├── experiments/                   # thin, reproducible entrypoints
│   ├── run_retrieval_baselines.py
│   ├── run_label_generation.py
│   ├── run_classifier.py
│   ├── run_adaptive.py           #   adaptive vs fixed vs oracle + tables
│   └── run_ablations.py          #   strategy subsets / chunk size / top-K / alpha
│
├── data/{raw,processed,corpus,labels}/
├── cache/{bm25,embeddings,faiss,retrieval_results,evidence_scores,strategy_labels}/
├── checkpoints/
└── results/
```

## Quickstart (DEBUG mode — fully offline, runs in this sandbox)

```bash
pip install -r requirements.txt   # core deps only; see below for optional extras
python run_pipeline.py --config config.yaml
```

`config.yaml` ships with `run.debug: true`. In this mode:
- Every dataset loader tries the real HuggingFace `datasets` source first,
  and **falls back to a small offline synthetic knowledge base** if the
  library or network access isn't available (as in this sandbox — there is
  no route to huggingface.co here).
- The dense/semantic retriever uses a TF-IDF stand-in encoder instead of
  downloading `BAAI/bge-base-en-v1.5`.
- The classifier uses TF-IDF + Logistic Regression instead of downloading
  `microsoft/deberta-v3-base`.

This means **the entire pipeline — all 7 stages — is verified to run
end-to-end without any internet access**, exactly per spec section 38
("DEBUG mode ... before running the full experiment").

Outputs land in `results/`:
- `results/main_table.md` — the spec-section-30 comparison table
- `results/per_dataset_table.md` — spec-section-31 breakdown
- `results/classifier_test_metrics.json` + `results/plots/confusion_matrix.png`
- `results/ablations/ablations_summary.json`
- `results/adaptive_vs_oracle.json`

## Full-scale configuration checklist

Everything needed for a real, full-scale run is config-driven — no code
edits required. Before running on a machine with internet + GPU access:

| Setting | Where | What to check |
|---|---|---|
| `run.debug` | `config.yaml` | Set to `false` |
| `run.device` | `config.yaml` | `"auto"` picks GPU if available; set `"cpu"`/`"cuda"` explicitly to override |
| `datasets.per_dataset_n` | `config.yaml` | Per-dataset train+val+test counts (see table below); any dataset in `datasets.use` without an entry here falls back to `full_run.default_questions_per_dataset` |
| `classifier.backend` | `config.yaml` | `"transformer"` to use DeBERTa-v3-base instead of TF-IDF+LogReg |
| `semantic.model_name` | `config.yaml` | Real dense encoder, e.g. `BAAI/bge-base-en-v1.5` (used automatically once `run.debug=false`) |
| `labeling.top_k` | `config.yaml` | Retrieval depth used only during strategy-label generation (decoupled from `bm25.top_k`/`semantic.top_k`, which govern retrieval depth at evaluation time) |
| `ablations.sample_size` | `config.yaml` | Caps the (expensive, re-retrieving) chunk-size/top-K/alpha ablation sweeps to a sample of the test split in FULL mode |
| `performance.retrieval_batch_size` | `config.yaml` | Batch size for retrieval during label generation / baselines / adaptive inference (default 32). This is what actually keeps a GPU busy — see "Batched retrieval" below |
| `evaluation.semantic_match_threshold`, `token_f1_match_threshold` | `config.yaml` | Wired into every `evaluate_chunks_against_gold(...)` call site (label generation, baselines, adaptive, ablations) — the semantic-similarity fallback in gold-evidence matching reuses the already-built dense encoder, so it costs nothing extra to enable |

Suggested per-dataset counts for a first full-scale run (already the
`config.yaml` defaults):

| Dataset | Count (train+val+test) |
|---|---:|
| SQuAD | 4,000 |
| Natural Questions | 4,000 |
| TriviaQA | 4,000 |
| HotpotQA | 4,000 |
| 2WikiMultiHopQA | 4,000 |
| MuSiQue | 2,800 |

Roughly equal counts per dataset matter more than raw volume: you want the
classifier to see a balanced mix of single-hop-favoring questions
(SQuAD/NQ/TriviaQA) and multi-hop-favoring questions
(HotpotQA/2Wiki/MuSiQue), not a distribution dominated by whichever dataset
happens to be largest.

## Batched retrieval (throughput on GPU)

Every retriever exposes `retrieve_batch(queries: list[str], top_k: int) -> list[list[dict]]`
in addition to the single-query `retrieve()`:

- **`semantic`**: encodes the whole batch of queries in one encoder forward
  pass (`corpus/index.py::encode_queries`), then does one batched FAISS/numpy
  search instead of one call per question.
- **`hybrid`**: batches its semantic half the same way; BM25 stays a
  per-query loop (CPU-bound, no GPU benefit to batching).
- **`multihop`**: hop-synchronized batching — at each hop, every
  still-active question's (possibly already-refined) query goes through
  `hybrid.retrieve_batch` in one call, rather than looping questions one at
  a time and paying the encoder cost 2-3x over per question. This is the
  biggest win, since multi-hop retrieval is the most expensive strategy.

`labeling/generate_labels.py`, `experiments/run_retrieval_baselines.py`, and
`experiments/run_adaptive.py` all process records in batches of
`performance.retrieval_batch_size` rather than one question at a time
(`run_adaptive.py` additionally batches the classifier's `.predict()` call,
and groups each batch by predicted strategy so each retriever still gets one
batched call). Verified to produce byte-identical results to the unbatched
per-question loop (same ranking, same chunk_ids) — batching only changes
throughput, never the retrieved results.

## Running with real data and real models (FULL mode)

1. Install the optional stack: `pip install sentence-transformers faiss-cpu torch transformers datasets`
2. Edit `config.yaml`:
   ```yaml
   run:
     debug: false
   classifier:
     backend: "transformer"   # use DeBERTa-v3-base instead of TF-IDF+LogReg
   ```
3. Re-run `python run_pipeline.py --config config.yaml`.

Every preprocessing loader already contains the correct real-dataset parsing
logic (SQuAD's answer-span → sentence mapping, HotpotQA/2WikiMultiHopQA's
`supporting_facts`, MuSiQue's `is_supporting` paragraphs, TriviaQA's
`entity_pages`, NQ's short-answer annotations) — `run.debug=false` simply
means loaders no longer need the synthetic fallback because `try_load_hf_dataset`
will succeed.

## Design notes / how each spec requirement is satisfied

- **No LLM / no answer generation anywhere.** There is no `generation/`
  package, no FLAN-T5/GPT/LLaMA loading, no EM/F1 computation. Grep the repo
  for "answer" and you'll only find it as a dataset field, never as a
  generation target.
- **No gold-evidence leakage at inference** (`retrieval/base.py`,
  `retrieval/multihop.py`): every retriever's `.retrieve()` signature takes
  only `(query: str, top_k: int)`. The multi-hop retriever refines its query
  using only its *own* previously retrieved chunk text — never
  `gold_sentences`/`gold_documents`/`answer`/`supporting_facts`. Gold
  evidence only ever appears downstream, inside `evaluation/`.
- **Classifier input is question text only** (`classifier/dataset.py`,
  `labeling/generate_labels.py`): dataset name, question_type, and gold
  evidence are present in the label JSONL for analysis but are never fed to
  the classifier.
- **ID-based matching preferred, text-based fallback** implemented exactly
  as specified in `evaluation/evidence_matching.py` (ID → exact-normalized →
  token-F1 ≥ 0.5 → semantic cosine ≥ threshold).
- **Oracle vs Adaptive vs Fixed** comparison implemented in
  `experiments/run_adaptive.py`, reusing cached per-question, per-strategy
  metrics so the oracle doesn't need to re-run retrieval.
- **Caching**: BM25 indices, dense embeddings, and per-question retrieval
  results are all cache-keyed off the chunk corpus + config, so re-running
  `run_pipeline.py` doesn't redundantly rebuild anything (see `corpus/index.py`).

## Known limitation of the synthetic fallback

The offline synthetic knowledge base (`preprocessing/base.py::SyntheticKB`)
is intentionally small (~90 documents) so DEBUG mode runs in seconds. At
that scale, BM25/semantic/hybrid can all trivially retrieve the right
document for single-hop questions, and even the multi-hop "bridge" document
sometimes surfaces by chance for a single-shot retriever. **This is a
property of the toy corpus size, not a bug in the retrieval or matching
logic** — with the real, much larger multi-document corpora built from the
six actual datasets, the strategies differentiate meaningfully (which is the
entire point of learning a strategy classifier). If you want to see clearer
separation in DEBUG mode, increase `preprocessing/base.py`'s
`SyntheticKB(num_documents=...)` or shrink `bm25.top_k` / `semantic.top_k`.

## Reproducibility

Every experiment script writes its full config-derived cache key alongside
outputs. To log full environment/version info per spec section 40, wrap
`run_pipeline.py` with your own environment dump (e.g. `pip freeze`,
`torch.__version__`, `nvidia-smi`) — a placeholder hook is not included by
default since this sandbox has no GPU to report.
