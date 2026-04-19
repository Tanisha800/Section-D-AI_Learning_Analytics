import os

# Lazy-loaded globals — only initialized when vector_db actually exists
_embedding = None
_embedding_loaded = False


def _get_embedding():
    """Load embedding model only when actually needed (lazy init)."""
    global _embedding, _embedding_loaded

    if _embedding_loaded:
        return _embedding

    _embedding_loaded = True

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("Embedding model loaded")
    except Exception as e:
        print("Embedding load failed:", e)
        _embedding = None

    return _embedding


def load_db():
    try:
        if not os.path.exists("vector_db"):
            print("vector_db folder not found")
            return None

        embedding = _get_embedding()

        if embedding is None:
            print("Embedding not initialized")
            return None

        from langchain_community.vectorstores import FAISS

        db = FAISS.load_local(
            "vector_db",
            embedding,
            allow_dangerous_deserialization=True  # IMPORTANT
        )

        print("FAISS DB loaded")
        return db

    except Exception as e:
        print("FAISS load error:", e)
        return None


def retrieve(query):
    try:
        db = load_db()

        if db is None:
            return ["No vector database available"]

        docs = db.similarity_search(query, k=3)

        return [d.page_content for d in docs]

    except Exception as e:
        print("Retrieval error:", e)
        return [f"RAG Error: {str(e)}"]