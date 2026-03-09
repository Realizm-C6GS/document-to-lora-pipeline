import os
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model

# ----- paths -----
MODEL_DIR  = Path(r"D:\ai\retro\tinychat")           # TinyChat base model
DATA_PATH  = Path(r"D:\ai\retro\training_data.jsonl")
OUTPUT_DIR = Path(r"D:\ai\retro\tinychat_lora")

# ----- dataset -----
ds = load_dataset("json", data_files=str(DATA_PATH), split="train")

def build_prompt(ex):
    instr = ex.get("instruction", "").strip()
    inp   = ex.get("input", "").strip()
    out   = ex.get("output", "").strip()
    prompt = f"Instruction:\n{instr}\n"
    if inp:
        prompt += f"Input:\n{inp}\n"
    prompt += "Response:"
    ex["text"] = prompt + out
    return ex

ds = ds.map(build_prompt, remove_columns=ds.column_names)

# ----- tokenizer -----
tok = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    use_fast=False,
    local_files_only=True,
    trust_remote_code=True
)
tok.pad_token = tok.eos_token

def tokenize(batch):
    enc = tok(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=1024,
    )
    enc["labels"] = enc["input_ids"].copy()
    return enc

ds_tok = ds.map(tokenize, batched=True)

# ----- model (4‑bit quantization) -----
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    quantization_config=bnb,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True
)

# ----- LoRA configuration -----
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)

# ----- training setup -----
args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=200,
    fp16=True,
    optim="paged_adamw_8bit",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_tok,
)

trainer.train()
model.save_pretrained(OUTPUT_DIR / "adapter")

print("LoRA adapter saved to", OUTPUT_DIR / "adapter")
