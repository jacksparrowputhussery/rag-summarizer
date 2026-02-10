'''
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever

from config.defaults import EMBEDDING_MODEL
from config.defaults import EMBEDDING_DIM



def init_document_store(
    sql_url: str,
    faiss_index_path: str = "faiss.index",
    create: bool = False,
):
    """
    create=True  -> create NEW FAISS index
    create=False -> load EXISTING FAISS index
    """

    if create:
        # CREATE MODE
        return FAISSDocumentStore(
            sql_url=sql_url,
            faiss_index_factory_str="Flat",
            embedding_dim=EMBEDDING_DIM,
        )
    else:
        # LOAD MODE (CORRECT)
        return FAISSDocumentStore.load(
            faiss_index_path
        )


def index_documents(
    store,
    documents,
    use_gpu: bool = False,
    faiss_index_path: str = "faiss.index",
):
    """
    Writes documents, builds embeddings, and saves FAISS index.
    """

    # 1. Write documents to SQL
    store.write_documents(documents)

    # 2. Create embedding retriever
    retriever = EmbeddingRetriever(
        document_store=store,
        embedding_model=EMBEDDING_MODEL,
        use_gpu=use_gpu,
    )

    # 3. Generate embeddings + build FAISS index
    store.update_embeddings(retriever)

    # 4. Persist FAISS index
    store.save(index_path=faiss_index_path)

    print(f"FAISS index saved to: {faiss_index_path}")
'''

from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever

from config.defaults import EMBEDDING_MODEL
from config.defaults import EMBEDDING_DIM


def init_document_store(
    sql_url: str,
    faiss_index_path: str = "faiss.index",
    create: bool = False,
):
    """
    create=True  -> create NEW FAISS index
    create=False -> load EXISTING FAISS index
    
    Note: When loading, the sql_url is stored in the faiss.index.json config file,
    so we only need to provide the index_path.
    """

    if create:
        # CREATE MODE - new document store
        return FAISSDocumentStore(
            sql_url=sql_url,
            faiss_index_factory_str="Flat",
            embedding_dim=EMBEDDING_DIM,
        )
    else:
        # LOAD MODE - the saved config contains the sql_url
        # 🔧 FIX: FAISSDocumentStore.load() only takes index_path
        return FAISSDocumentStore.load(
            index_path=faiss_index_path,
        )


def index_documents(
    store,
    documents,
    use_gpu: bool = False,
    faiss_index_path: str = "faiss.index",
):
    """
    Writes documents, builds embeddings, and saves FAISS index.
    """

    # 1. Write documents to SQL
    store.write_documents(documents)

    # 2. Create embedding retriever
    retriever = EmbeddingRetriever(
        document_store=store,
        embedding_model=EMBEDDING_MODEL,
        use_gpu=use_gpu,
    )

    # 3. Generate embeddings + build FAISS index
    store.update_embeddings(retriever)

    # 4. Persist FAISS index
    store.save(index_path=faiss_index_path)

    print(f"✅ FAISS index saved to: {faiss_index_path}")
    print(f"✅ Document count: {store.get_document_count()}")