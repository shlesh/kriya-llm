import argparse, math, torch, json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--eval_json", required=True)
    ap.add_argument("--seq_len", type=int, default=4096)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("json", data_files=args.eval_json, split="train")
    ds = ds.map(lambda ex: tokenizer(ex["text"]), batched=True, remove_columns=ds.column_names)
    def group(ex):
        ids = []
        for t in ex["input_ids"]:
            ids.extend(t + [tokenizer.eos_token_id])
        L = (len(ids)//args.seq_len)*args.seq_len
        return {"input_ids":[ids[i:i+args.seq_len] for i in range(0,L,args.seq_len)]}
    ds = ds.map(group, batched=True)
    ds = ds.map(lambda ex: {"labels": ex["input_ids"]}, batched=True)

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    args_tr = TrainingArguments(output_dir="tmp-eval", per_device_eval_batch_size=1)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    trainer = Trainer(model=model, args=args_tr, eval_dataset=ds, data_collator=collator)
    metrics = trainer.evaluate()
    ppl = math.exp(metrics["eval_loss"])
    print(f"Perplexity: {ppl:.3f}")

if __name__ == "__main__":
    main()
