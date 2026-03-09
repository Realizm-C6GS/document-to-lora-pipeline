import os, glob
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model

# --- directories ---
BASE_DIR  = r"D:\ai\retro"
QA_DIR    = os.path.join(BASE_DIR, "qa")
LORA_DIR  = os.path.join(BASE_DIR, "lora")
LOGS_DIR  = os.path.join(BASE_DIR, "logs")
MODEL_DIR = os.path.join(BASE_DIR, "qwen3_8b")

os.makedirs(LORA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# --- dataset ---
DATA_FILES = glob.glob(os.path.join(QA_DIR, "**", "*.jsonl"), recursive=True)
if not DATA_FILES:
    raise SystemExit("No Q/A files found under qa/. Run generate_qa_all.py first.")

print(f"Found {len(DATA_FILES)} Q/A files")

data = load_dataset("json", data_files=DATA_FILES, split="train")

def to_text(example):
    instr = example.get("instruction", "").strip()
    inp   = example.get("input", "").strip()
    out   = example.get("output", "").strip()
    prompt = f"Instruction:\n{instr}\n"
    if inp:
        prompt += f"Input:\n{inp}\n"
    prompt += "Response:"
    example["text"] = prompt + out
    return example

data = data.map(to_text, remove_columns=data.column_names)

# --- tokenizer ---
tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=True)
tok.pad_token = tok.eos_token

def tokenize(batch):
    return tok(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=4096,   # 8B model supports large context
    ) | {"labels": tok(batch["text"], truncation=True, padding="max_length", max_length=4096)["input_ids"]}

data_tok = data.map(tokenize, batched=True)

# --- quantization / 4-bit QLoRA setup ---
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",   # use bf16 math for stability if hardware supports it
)

# --- base model load ---
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)

# --- LoRA configuration ---
lora_conf = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_conf)

# --- training arguments ---
train_out = os.path.join(LORA_DIR, "qwen3_8b_lora")
args = TrainingArguments(
    output_dir=train_out,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,        # effective batch size 16
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=200,
    fp16=True,                             # fallback for non-bf16 GPUs
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
)

trainer = Trainer(model=model, args=args, train_dataset=data_tok)
trainer.train()

adapter_dir = os.path.join(train_out, "adapter")
model.save_pretrained(adapter_dir)
print("LoRA adapter saved to", adapter_dir)
