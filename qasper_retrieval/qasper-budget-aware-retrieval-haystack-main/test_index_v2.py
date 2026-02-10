import os
from data.qasper_loader import QASPERLoader
from indexing.document_store import init_document_store, index_documents

# ⚠️ Clean up old files first
db_file = "test.db"
index_files = ["faiss.index", "faiss.index.faiss", "faiss.index.json"]

print("🧹 Checking for existing files...\n")

# Remove old database
if os.path.exists(db_file):
    os.remove(db_file)
    print(f"✅ Removed old database: {db_file}")

# Remove old index files
for f in index_files:
    if os.path.exists(f):
        os.remove(f)
        print(f"✅ Removed old index: {f}")

if not any(os.path.exists(f) for f in [db_file] + index_files):
    print("✅ No old files found\n")

print("=" * 60)
print("Creating fresh index...")
print("=" * 60 + "\n")

# Load documents
loader = QASPERLoader("qasper-dev-v0.3.json")
paper_ids = loader.get_paper_ids()[:2]
docs = loader.get_documents_for_indexing(paper_ids)

print(f"📄 Loaded {len(docs)} document chunks from {len(paper_ids)} paper(s)\n")

# Create new store
store = init_document_store(
    sql_url=f"sqlite:///{db_file}",
    faiss_index_path="faiss.index",
    create=True,  # Creating new index
)

print("🔨 Building embeddings and FAISS index...\n")

# Index documents
index_documents(
    store, 
    docs,
    use_gpu=False,
    faiss_index_path="faiss.index",
)

print("\n" + "=" * 60)
print("✅ SUCCESS!")
print("=" * 60)
print(f"📊 Indexed documents: {store.get_document_count()}")
print(f"💾 Database: {db_file}")
print(f"🔍 FAISS index: faiss.index")
print("=" * 60)