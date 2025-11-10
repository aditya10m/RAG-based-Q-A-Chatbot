from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline
import torch

def build_llm(model_name: str, device: str = "auto"):
    device_map = "auto" if device == "auto" else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map=device_map
    )
    tok = AutoTokenizer.from_pretrained(model_name)
    pipe = hf_pipeline("text-generation", model=model, tokenizer=tok, max_new_tokens=256)
    return HuggingFacePipeline(pipeline=pipe)

def make_rag_chain(retriever, model_name: str, device: str = "auto"):
    template = \"\"\"You are a helpful assistant. Use the provided context to answer the question.
If the answer is not in the context, say you do not know.

Context:
{context}

Question: {question}

Answer:\"\"\"
    prompt = ChatPromptTemplate.from_template(template)

    llm = build_llm(model_name=model_name, device=device)

    def format_docs(docs: List):
        return "\\n\\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return chain