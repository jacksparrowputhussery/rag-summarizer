# config/defaults.py

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

DEFAULT_SIGNAL_WEIGHTS = {
    "semantic": 0.30,
    "lexical": 0.20,
    "structural": 0.25,
    "position": 0.15,
    "section_type": 0.10,
}

SECTION_TYPE_WEIGHTS = {
    "abstract": 1.4,
    "introduction": 1.1,
    "methodology": 1.2,
    "results": 1.3,
    "conclusion": 1.3,
    "body": 1.0,
}

DEFAULT_TOKEN_BUDGET = 4000
