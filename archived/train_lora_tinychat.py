import os, glob
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

BASE_DIR = r"D:\ai\retro"
QA_DIR = os.path.join(BASE_DIR, "qa")
LORA_DIR = os.path.join(BASE_DIR, "lora")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODEL_DIR = os.path.join(BASE_DIR, "tinychat")

os.makedirs(LORA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

DATA_FILES = glob.glob(os.path.join(QA_DIR, "**", "*.jsonl"), recursive=True)
if not DATA_FILES:
    raise SystemExit("No Q/A files found under qa/; run generate_qa_all.py first.")

print(f"Found {len(DATA_FILES)} Q/A files")

data = load_dataset("json", data_files=DATA_FILES, split="train")

def to_text(example):
    instr = example.get("instruction","").strip()
    inp   = example.get("input","").strip()
    out   = example.get("output","").strip()
    prompt = f"Instruction:\n{instr}\n"
    if inp:
        prompt += f"Input:\n{inp}\n"
    prompt += "Response:"
    example["text"] = prompt + out
    return example

data = data.map(to_text, remove_columns=data.column_names)

tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=True)
tok.pad_token = tok.eos_token

def tokenize(batch):
    enc = tok(batch["text"], truncation=True, padding="max_length", max_length=1024)
    enc["labels"] = enc["input_ids"].copy()
    return enc

data_tok = data.map(tokenize, batched=True)

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)

lora_conf = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_conf)

args = TrainingArguments(
    output_dir=os.path.join(LORA_DIR,"tinychat_lora"),
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=200,
    fp16=True,
    optim="paged_adamw_8bit"
)

trainer = Trainer(model=model, args=args, train_dataset=data_tok)
trainer.train()

model.save_pretrained(os.path.join(LORA_DIR,"tinychat_lora","adapter"))
print("✅ LoRA adapter saved to", os.path.join(LORA_DIR,"tinychat_lora","adapter"))
