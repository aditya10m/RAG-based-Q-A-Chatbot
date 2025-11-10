from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_retriever(index_path: str, embedding_model: str, k: int = 4):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": k})
    return retriever