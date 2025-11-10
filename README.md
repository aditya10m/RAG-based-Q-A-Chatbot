# RAG-based Q&A Chatbot

A production-ready starter that combines Retrieval-Augmented Generation (RAG) with optional LoRA / QLoRA fine-tuning of LLaMA-family models.
Built with Hugging Face Transformers, LangChain, FAISS, PEFT, bitsandbytes.

## Features
- RAG pipeline: chunking, embeddings, FAISS vector store
- LLM: any HF causal LM (e.g., meta-llama/Meta-Llama-3-8B-Instruct)
- Fine-tuning: LoRA / QLoRA with PEFT
- Evaluation: accuracy (EM/F1) + perplexity
- API (FastAPI) and CLI (Typer)

## Structure
See repository tree in this README's header.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# Build index
python -m src.cli ingest --docs_dir data/sample_docs --index_path indexes/my_index

# Run API
uvicorn src.app:app --reload --port 8000
# Ask
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d '{"question":"What is RAG?"}'

# CLI ask
python -m src.cli ask --index_path indexes/my_index --question "What is RAG?"
```

## LoRA / QLoRA
```bash
python -m src.finetune.train_lora   --base_model meta-llama/Meta-Llama-3-8B-Instruct   --train_file data/sample_data/qa.jsonl   --output_dir models/llama3-8b-lora   --use_qlora   --batch_size 2 --grad_accum 8 --epochs 1
```

## Evaluation
```bash
python -m src.eval.evaluate   --model meta-llama/Meta-Llama-3-8B-Instruct   --eval_file data/sample_data/qa.jsonl
```