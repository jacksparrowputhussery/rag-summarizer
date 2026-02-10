import os
import torch
from data.qasper_loader import QASPERLoader
from indexing.document_store import init_document_store
from retrieval.budget_aware_hybrid import BudgetAwareHybridRetriever
from config.defaults import *

# Check if index exists
db_file = "test.db"
index_file = "faiss.index"

print("=" * 60)
print("Checking for existing index...")
print("=" * 60 + "\n")

if not os.path.exists(db_file):
    print(f"❌ Database not found: {db_file}")
    print("\n💡 Please run test_index.py first to create the index!\n")
    exit(1)

if not os.path.exists(index_file):
    print(f"❌ FAISS index not found: {index_file}")
    print("\n💡 Please run test_index.py first to create the index!\n")
    exit(1)

print(f"✅ Found database: {db_file}")
print(f"✅ Found FAISS index: {index_file}\n")

# Load questions
loader = QASPERLoader("qasper-dev-v0.3.json")
paper_ids = loader.get_paper_ids()[0:1]
questions = loader.get_questions(paper_ids)

if not questions:
    print("❌ No questions found!")
    exit(1)

print(f"📋 Loaded {len(questions)} questions\n")

# Load existing index
print("=" * 60)
print("Loading FAISS index...")
print("=" * 60 + "\n")

store = init_document_store(
    sql_url=f"sqlite:///{db_file}",
    faiss_index_path=index_file,
    create=False,  # Loading existing index
)

print(f"✅ Loaded {store.get_document_count()} documents from index\n")

# Create retriever
print("=" * 60)
print("Initializing hybrid retriever...")
print("=" * 60 + "\n")

retriever = BudgetAwareHybridRetriever(
    document_store=store,
    embedding_model=EMBEDDING_MODEL,
    token_budget=DEFAULT_TOKEN_BUDGET,
    signal_weights=DEFAULT_SIGNAL_WEIGHTS,
    section_type_weights=SECTION_TYPE_WEIGHTS,
    use_gpu=torch.cuda.is_available(),
)

print(f"✅ Retriever ready")
print(f"   Token budget: {DEFAULT_TOKEN_BUDGET}")
print(f"   Using GPU: {torch.cuda.is_available()}\n")

# Test retrieval
q = questions[0]

print("=" * 60)
print("RETRIEVAL TEST")
print("=" * 60)
print(f"Question: {q['question']}")
print(f"Paper ID: {q['paper_id']}")
print("=" * 60 + "\n")

results = retriever.retrieve(
    query=q["question"],
    filters={"paper_id": [q["paper_id"]]},
    top_k=10,
)

print(f"✅ Retrieved {len(results)} documents\n")

# Display results
for i, doc in enumerate(results, 1):
    print("=" * 60)
    print(f"📄 Rank {i} | Score: {doc.score:.4f}")
    print(f"   Section: {doc.meta['section_type']}")
    print(f"   Word count: {doc.meta['word_count']}")
    print(f"   Position: {doc.meta.get('position', 0):.2f}")
    print("-" * 60)
    preview = doc.content[:300] + "..." if len(doc.content) > 300 else doc.content
    print(preview)
    print()

# Calculate total tokens used
total_tokens = sum(d.meta.get('word_count', 0) for d in results)
print("=" * 60)
print(f"📊 Total tokens used: {total_tokens} / {DEFAULT_TOKEN_BUDGET}")
print(f"📊 Budget utilization: {100 * total_tokens / DEFAULT_TOKEN_BUDGET:.1f}%")
print("=" * 60)