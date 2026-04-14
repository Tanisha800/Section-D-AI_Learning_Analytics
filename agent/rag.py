from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_db():
    try:
        return FAISS.load_local("vector_db", embedding)
    except:
        return None

def retrieve(query):
    db = load_db()
    if db is None:
        return []

    docs = db.similarity_search(query, k=3)
    return [d.page_content for d in docs]