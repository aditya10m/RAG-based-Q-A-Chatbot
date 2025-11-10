import os
import pathlib
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docloaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_texts(docs_dir: str) -> List[str]:
    texts = []
    p = pathlib.Path(docs_dir)
    for path in p.rglob("*"):
        if path.suffix.lower() in {".txt", ".md"}:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                texts.append(f.read())
    return texts

def build_index(docs_dir: str, index_path: str, embedding_model: str):
    texts = load_texts(docs_dir)
    if not texts:
        raise RuntimeError(f"No .txt/.md files found in {docs_dir}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    db = FAISS.from_texts(chunks, embedding=embeddings)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    db.save_local(index_path)
    print(f"Saved FAISS index to {index_path}")

if __name__ == "__main__":
    import argparse, yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_dir", type=str, required=True)
    parser.add_argument("--index_path", type=str, default="indexes/default")
    parser.add_argument("--config", type=str, default="src/configs/config.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    build_index(args.docs_dir, args.index_path, cfg["embedding_model"])