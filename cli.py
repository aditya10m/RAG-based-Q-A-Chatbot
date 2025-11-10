import typer, yaml
from rich import print
from src.rag.ingest import build_index
from src.rag.retriever import load_retriever
from src.rag.pipeline import make_rag_chain

app = typer.Typer(help="RAG Chatbot CLI")

@app.command()
def ingest(docs_dir: str = "data/sample_docs", index_path: str = "indexes/my_index", config: str = "src/configs/config.yaml"):
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)
    build_index(docs_dir, index_path, cfg["embedding_model"])

@app.command()
def ask(question: str, index_path: str = "indexes/my_index", config: str = "src/configs/config.yaml"):
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)
    retriever = load_retriever(index_path, cfg["embedding_model"], cfg["k"])
    chain = make_rag_chain(retriever, cfg["llm_model"], cfg.get("device", "auto"))
    ans = chain.invoke(question)
    print({"answer": str(ans)})

if __name__ == "__main__":
    app()