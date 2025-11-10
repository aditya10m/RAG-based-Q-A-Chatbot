import os, json, argparse
from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig, Trainer)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True, type=str, help="HF model id or local path")
    p.add_argument("--train_file", required=True, type=str, help="jsonl with fields: question, answer")
    p.add_argument("--output_dir", default="models/lora-out", type=str)
    p.add_argument("--epochs", default=1, type=int)
    p.add_argument("--batch_size", default=2, type=int)
    p.add_argument("--lr", default=2e-4, type=float)
    p.add_argument("--use_qlora", action="store_true")
    return p.parse_args()

def load_qa_dataset(path):
    # Expect jsonl
    def gen():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                yield {"text": f"Question: {obj['question']}\\nAnswer: {obj['answer']}"}
    from datasets import Dataset
    return Dataset.from_generator(gen)

def main():
    args = parse_args()

    bnb_config = None
    if args.use_qlora:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="bfloat16")

    model = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=bnb_config, device_map="auto" if args.use_qlora else None)
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tok.pad_token = tok.eos_token

    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","v_proj","k_proj","o_proj"])
    model = get_peft_model(model, lora_config)

    ds = load_qa_dataset(args.train_file)

    def tokenize(example):
        return tok(example["text"], truncation=True, max_length=1024)
    ds_tok = ds.map(tokenize, batched=False, remove_columns=ds.column_names)

    data_collator = DataCollatorForLanguageModeling(tok, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        fp16=True,
        save_strategy="epoch",
        report_to="none"
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=ds_tok, data_collator=data_collator)
    trainer.train()

    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"Saved LoRA-adapted model to {args.output_dir}")

if __name__ == "__main__":
    main()