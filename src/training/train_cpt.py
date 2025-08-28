import argparse, os, yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_info()


# ---------------------------- config helpers --------------------------------
def load_config(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

def get_bnb_config(cfg: dict):
    if cfg.get("load_in_4bit", True):
        compute = cfg.get("bnb_compute_dtype", "bfloat16")
        compute_dtype = getattr(torch, compute)
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    return None


# ------------------------------ data utils ----------------------------------
def tokenize_function(examples, tokenizer):
    # We pack ourselves; no attention_mask here
    return tokenizer(examples["text"], add_special_tokens=False, return_attention_mask=False)

def group_texts(examples, block_size: int, eos_id: int):
    # concatenate then split into blocks of block_size
    concatenated = []
    for t in examples["input_ids"]:
        concatenated.extend(t + [eos_id])
    total_len = (len(concatenated) // block_size) * block_size
    chunks = [concatenated[i:i+block_size] for i in range(0, total_len, block_size)]
    return {"input_ids": chunks, "labels": [x[:] for x in chunks]}


# ------------------------- safe k-bit preparation ----------------------------
def prepare_model_for_kbit_training_safe(model, enable_gc: bool = True):
    """
    Minimal, VRAM-safe prep:
    - disable use_cache
    - enable gradient checkpointing (non-reentrant if available)
    - cast *only* norms to float32
    DO NOT cast lm_head to fp32 (big vocab => OOM).
    """
    model.config.use_cache = False
    if enable_gc:
        try:
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                model.gradient_checkpointing_enable()
            try:
                model.enable_input_require_grads()
            except Exception:
                pass
            print("[train] gradient checkpointing enabled.")
        except Exception as e:
            print(f"[train] could not enable gradient checkpointing: {e}")

    norm_names = ("norm", "layer_norm", "ln_f")
    for name, module in model.named_modules():
        if any(n in name for n in norm_names):
            try:
                module.to(torch.float32)
            except Exception:
                pass
    return model


# ---------------------------------- main ------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--eval_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--config", default="configs/training.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_id = args.model_id or cfg["model_id"]
    seq_len = int(cfg.get("seq_length", 512))
    attn_impl = cfg.get("misc", {}).get("attn_implementation", "sdpa")

    os.makedirs(args.out_dir, exist_ok=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset -> tokenize -> pack
    train_ds = load_dataset("json", data_files=args.train_json, split="train")
    eval_ds  = load_dataset("json", data_files=args.eval_json,  split="train")

    tokenized_train = train_ds.map(
        lambda x: tokenize_function(x, tokenizer), batched=True,
        remove_columns=train_ds.column_names, num_proc=1
    )
    tokenized_eval  = eval_ds.map(
        lambda x: tokenize_function(x, tokenizer), batched=True,
        remove_columns=eval_ds.column_names, num_proc=1
    )

    # Keep only input_ids before grouping (avoid attention_mask length issues)
    tokenized_train = tokenized_train.remove_columns([c for c in tokenized_train.column_names if c != "input_ids"])
    tokenized_eval  = tokenized_eval.remove_columns([c for c in tokenized_eval.column_names  if c != "input_ids"])

    def _group(examples):
        return group_texts(examples, block_size=seq_len, eos_id=tokenizer.eos_token_id)

    lm_train = tokenized_train.map(_group, batched=True, num_proc=1)
    lm_eval  = tokenized_eval.map(_group,  batched=True, num_proc=1)

    # Model (4-bit QLoRA) with SDPA; allow light CPU offload if needed
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.get("misc",{}).get("tf32", True))
    qconfig = get_bnb_config(cfg)

    # Optional max_memory to prevent init OOM and force some offload
    max_mem = None
    gpu_gib = cfg.get("misc", {}).get("max_memory_gpu_gib", None)
    cpu_gib = cfg.get("misc", {}).get("cpu_ram_gib", None)
    if gpu_gib:
        max_mem = { "cuda:0": f"{float(gpu_gib)}GiB" }
        if cpu_gib:
            max_mem["cpu"] = f"{int(cpu_gib)}GiB"

    common_kwargs = dict(
        trust_remote_code=True,
        quantization_config=qconfig,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if max_mem is not None:
        common_kwargs["max_memory"] = max_mem

    model = AutoModelForCausalLM.from_pretrained(model_id, **common_kwargs)

    # Safe prep (no fp32 cast of lm_head)
    model = prepare_model_for_kbit_training_safe(model, enable_gc=bool(cfg.get("gradient_checkpointing", True)))

    # LoRA
    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 8)),
        lora_alpha=int(lora_cfg.get("alpha", 16)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(lora_cfg.get("target_modules", ["q_proj","v_proj"])),
        bias=lora_cfg.get("bias","none"),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # Training args
    t = cfg["train"]
    max_steps_val = t.get("max_steps")
    max_steps_val = -1 if max_steps_val is None else int(max_steps_val)

    optim_name = cfg.get("optim", {}).get("name", "adamw_torch")
    ds_cfg = cfg.get("misc", {}).get("deepspeed_config", None)

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        learning_rate=float(t["lr"]),
        lr_scheduler_type=t.get("lr_scheduler_type","cosine"),
        warmup_ratio=float(t.get("warmup_ratio", 0.03)),
        per_device_train_batch_size=int(t["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(t.get("per_device_eval_batch_size",1)),
        gradient_accumulation_steps=int(t["gradient_accumulation_steps"]),
        num_train_epochs=float(t.get("num_epochs", 1)),
        max_steps=max_steps_val,
        logging_steps=int(t.get("logging_steps", 25)),
        eval_strategy="steps",
        eval_steps=int(t.get("eval_steps", 250)),
        save_steps=int(t.get("save_steps", 500)),
        save_total_limit=int(t.get("save_total_limit", 3)),
        bf16=bool(cfg.get("misc",{}).get("bf16", True)),
        fp16=bool(cfg.get("misc",{}).get("fp16", False)),
        optim=optim_name,  # paged_adamw_8bit by default
        report_to=["wandb"] if os.environ.get("WANDB_DISABLED","false").lower()!="true" else [],
        logging_dir=os.path.join(args.out_dir, "logs"),
        deepspeed=ds_cfg,  # optional Zero-2 CPU offload
    )

    # Collator builds attention_mask dynamically at batch-time
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_train,
        eval_dataset=lm_eval,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.out_dir)
