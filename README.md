# Document to LoRA Pipeline

## Current Workflow
These scripts now resolve paths from the repo root by default, so they can be run from this repository without editing hardcoded drive letters.

Install dependencies first:
python -m pip install -r requirements.txt

Initialize the llama.cpp submodule after cloning:
git submodule update --init --recursive

1. Put source PDFs in:
   pdfs\

2. Split PDFs into text sections:
   python dump_split.py

   Output:
   sections\<document-name>\*.txt

3. Generate Q/A training data from those text files:
   python generate_qa_all_dyn.py

   Output:
   qa\<document-name>\*.jsonl

   Logs:
   logs\qa_progress.json
   logs\qa_failures.log

4. Train the Qwen3-8B LoRA:
   python train_lora.py

   Output:
   lora_fpu64\qwen3_8b_lora\adapter\

5. Merge the adapter and convert it to GGUF:
   python merge_and_convert.py

6. Optional: run the interactive menu instead of invoking the scripts directly:
   python menu.py

   This is an extra convenience layer over the scripts above. It has not been fully validated end-to-end yet, so treat it as optional.


## Main Files to Use
- dump_split.py
  Current PDF ingestion and chunking script. Optional flags include:
  --pdf-dir, --sections-dir, --min-section-words, --fallback-part-size

- generate_qa_all_dyn.py
  Current Q/A generation script. This is the preferred generator because it detects model context and sizes chunks dynamically.
  Optional flags include:
  --api-url, --model-id, --sections-dir, --qa-dir, --safe-context-ratio

- train_lora.py
  Current local-model LoRA training script.
  Optional flags include:
  --model-dir, --output-root, --run-name, --max-len, --target-modules
  If --run-name is omitted, it defaults to <model-name>_lora.

- merge_and_convert.py
  Merges a LoRA adapter into the base model, runs llama.cpp conversion, and optionally quantizes the GGUF.
  Optional flags include:
  --output-root, --run-name, --adapter-dir, --merged-dir, --gguf-out, --quantize-type, --quantized-out

- menu.py
  Optional ASCII workflow menu for running the pipeline interactively. Useful for experimenting with paths and settings without typing full commands.

- requirements.txt
  Top-level Python dependencies for the active workflow.

- get-qwen3-8b.txt
  Example command for downloading Qwen3-8B. The workflow is no longer tied to that exact model directory if you pass --model-dir.


## Archived Files
The archived\ folder contains older or superseded scripts kept for reference only. These include:

- one-off PDF/text preprocessing scripts
- early single-file Q/A generation scripts
- TinyChat proof-of-concept training, test, and merge scripts
- earlier Qwen training variants that are no longer the main path

Use the root-level files listed above for the current workflow.
