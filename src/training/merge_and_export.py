import argparse, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print("Merged model saved to", args.out_dir)

if __name__ == "__main__":
    main()
