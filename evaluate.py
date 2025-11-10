import json, argparse, math, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

def normalize(text):
    import re, string
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\\s+", " ", text).strip()
    return text

def exact_f1(pred, gold):
    p = normalize(pred)
    g = normalize(gold)
    if not p and not g:
        return 1.0, 1.0
    em = float(p == g)
    p_toks, g_toks = p.split(), g.split()
    common = {}
    for t in p_toks:
        if t in g_toks:
            common[t] = common.get(t, 0) + 1
    num_same = sum(min(p_toks.count(t), g_toks.count(t)) for t in common)
    if len(p_toks) == 0 or len(g_toks) == 0:
        return em, 0.0
    precision = num_same / len(p_toks)
    recall = num_same / len(g_toks)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return em, f1

def ppl(model_name, eval_texts):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    nlls = []
    for text in eval_texts:
        enc = tok(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            nll = outputs.loss * input_ids.shape[1]
        nlls.append(nll.item())
    total_nll = sum(nlls)
    total_tokens = sum(len(tok(t)["input_ids"]) for t in eval_texts)
    return math.exp(total_nll / total_tokens)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=str)
    ap.add_argument("--eval_file", required=True, type=str, help="jsonl with question, answer")
    args = ap.parse_args()
    qa = [json.loads(l) for l in open(args.eval_file, "r", encoding="utf-8") if l.strip()]
    pipe = pipeline("text-generation", model=args.model)
    preds = []
    for ex in qa:
        prompt = f"Question: {ex['question']}\\nAnswer:"
        out = pipe(prompt, max_new_tokens=64)[0]["generated_text"]
        pred = out.split("Answer:", 1)[-1].strip()
        preds.append(pred)
    ems, f1s = [], []
    for pred, ex in zip(preds, qa):
        em, f1 = exact_f1(pred, ex["answer"])
        ems.append(em); f1s.append(f1)
    perp = ppl(args.model, [f"Question: {x['question']}\\nAnswer: {x['answer']}" for x in qa])
    print(json.dumps({"exact_match": float(np.mean(ems)), "f1": float(np.mean(f1s)), "perplexity": float(perp)}, indent=2))

if __name__ == "__main__":
    main()