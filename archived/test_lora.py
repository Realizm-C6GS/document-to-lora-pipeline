from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_dir    = Path(r"D:\ai\retro\tinychat")
adapter_dir = Path(r"D:\ai\retro\tinychat_lora\adapter")   # ← new adapter

tok = AutoTokenizer.from_pretrained(
    str(base_dir), local_files_only=True, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    str(base_dir), local_files_only=True,
    trust_remote_code=True, device_map="auto"
)

print("Attaching LoRA adapter…")
model = PeftModel.from_pretrained(model, str(adapter_dir), local_files_only=True)
print("Adapter loaded.")

prompt = "Explain how the Pentium 4 trace cache improves performance compared to a traditional instruction cache."

chat = tok.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tok(chat, return_tensors="pt").to(model.device)

print("Generating response…\n")
out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
response = tok.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
print("Model + LoRA response:\n", response)
