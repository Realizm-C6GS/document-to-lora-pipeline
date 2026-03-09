from transformers import AutoModelForCausalLM
from peft import PeftModel

base   = r"D:\ai\retro\tinychat"
lora   = r"D:\ai\retro\tinychat_lora\adapter"
merged = r"D:\ai\retro\tinychat_merged"

model = AutoModelForCausalLM.from_pretrained(base, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, lora)
model = model.merge_and_unload()
model.save_pretrained(merged)
print("Merged model saved to:", merged)
