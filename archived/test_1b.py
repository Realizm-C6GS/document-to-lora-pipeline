from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

base_dir = Path(r"D:\ai\retro\tinychat")

tok = AutoTokenizer.from_pretrained(str(base_dir), local_files_only=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(str(base_dir), local_files_only=True,
                                             trust_remote_code=True, device_map="auto")

# simple prompt
prompt = "Why did Intel design the NetBurst pipeline to be so long?"

# apply the chat template that the model expects
chat = tok.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True
)

inputs = tok(chat, return_tensors="pt").to(model.device)

out = model.generate(**inputs, max_new_tokens=150, do_sample=False)
response = tok.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
print("Model response:\n", response)
