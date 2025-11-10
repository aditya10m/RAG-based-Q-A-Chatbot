from fastapi import FastAPI
from pydantic import BaseModel
import yaml
from src.rag.retriever import load_retriever
from src.rag.pipeline import make_rag_chain

app = FastAPI(title="RAG Q&A Chatbot")

with open("src/configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

retriever = load_retriever(index_path="indexes/my_index", embedding_model=cfg["embedding_model"], k=cfg["k"])
chain = make_rag_chain(retriever, model_name=cfg["llm_model"], device=cfg.get("device", "auto"))

class Ask(BaseModel):
    question: str

@app.post("/ask")
def ask(item: Ask):
    result = chain.invoke(item.question)
    return {"answer": str(result)}