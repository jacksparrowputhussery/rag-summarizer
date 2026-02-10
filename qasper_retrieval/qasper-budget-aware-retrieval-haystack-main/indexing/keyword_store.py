from haystack.document_stores import SQLDocumentStore

def init_keyword_store(sql_url: str):
    """
    Persistent BM25 keyword store
    """
    return SQLDocumentStore(
        url=sql_url,
        use_bm25=True
    )
