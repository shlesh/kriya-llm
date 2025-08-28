import argparse, torch, sys
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from peft import PeftModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True, help="Base or merged model path/ID")
    ap.add_argument("--adapter", default=None, help="Optional LoRA adapter dir (PEFT)")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")

    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print(">>> Enter prompt (CTRL+C to quit)")
    while True:
        try:
            prompt = input("\nYou: ")
        except KeyboardInterrupt:
            print("\nBye!"); break
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.8, top_p=0.95, streamer=streamer)
        _ = gen  # streamed

if __name__ == "__main__":
    main()
