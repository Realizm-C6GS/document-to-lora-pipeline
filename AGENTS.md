# Repository Guidelines

## Project Structure & Module Organization
Top-level Python scripts drive the local data pipeline. Use `pdfs/` and `pdf-storage/` for source documents, `sections/`, `sections-nolearn/`, and `sections-rtslaptop/` for extracted text chunks, and `qa/` plus `qa-rtslaptop/` for generated `.jsonl` training data. Training outputs live under `lora/`, `lora_fpu64/`, `tinychat_lora/`, and `tinychat_merged/`; runtime logs go to `logs/`. Treat `llama.cpp/` as a vendored upstream subtree with its own build, test, and style rules.

## Build, Test, and Development Commands
Run scripts from the repository root with Python.

```powershell
python dump_pdf.py
python split_sections.py
python generate_qa_all.py
python train_lora_qwen3-8b.py
python test_lora.py
```

`dump_pdf.py` extracts raw text from a PDF, `split_sections.py` turns dumps into section files, `generate_qa_all.py` converts section text into `.jsonl` Q/A pairs through the local LM Studio API, `train_lora_qwen3-8b.py` trains the adapter, and `test_lora.py` performs a local inference smoke test. Review and update hardcoded paths, model IDs, and API URLs at the top of each script before running.

## Coding Style & Naming Conventions
Follow existing Python conventions: 4-space indentation, `snake_case` for functions and files, and uppercase names for path and model constants such as `BASE_DIR` or `MODEL_ID`. Keep scripts single-purpose and configuration-first: put filesystem paths and tunable parameters near the top. Prefer standard library modules plus clearly named Hugging Face imports; avoid introducing new abstractions unless multiple scripts will share them.

## Testing Guidelines
There is no formal `pytest` suite at the repo root yet; current testing is script-based smoke validation. Name new checks `test_*.py` and keep them runnable directly with `python test_name.py`. For pipeline changes, verify both output files and logs: confirm new `.jsonl` files appear under `qa/` and inspect `logs/qa_failures.log` and `logs/qa_progress.json` for regressions.

## Commit & Pull Request Guidelines
This repository currently has no local commit history, so use short imperative commit messages with a scope prefix, for example `qa: tighten JSON sanitization` or `train: reduce batch memory`. Pull requests should describe the dataset or model impact, list the commands you ran, and include sample output or log snippets when generation quality changes. Call out large artifacts, path changes, and any updates that also require changes inside `llama.cpp/`.
