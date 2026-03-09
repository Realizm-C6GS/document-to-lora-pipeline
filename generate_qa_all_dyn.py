import argparse
import json
import os
import re
import signal
import sys
import time
from pathlib import Path

import requests


SYSTEM_PROMPT = """
You are generating training data.
Return ONLY a JSON array of 6-8 objects, each having exactly:
  instruction * input * output
No markdown, bullets, or commentary.
Each value must be on one line (no \\n).
The instruction should be a question or start with an imperative verb (Explain, Describe, List, Give, Summarize...).
"""


def resolve_path(value, base_dir):
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def parse_args():
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate Q/A JSONL files from section text files."
    )
    parser.add_argument(
        "--base-dir",
        default=repo_root,
        type=Path,
        help="Repository root. Defaults to the directory containing this script.",
    )
    parser.add_argument(
        "--sections-dir",
        help="Directory containing input section text files. Defaults to <base-dir>/sections.",
    )
    parser.add_argument(
        "--qa-dir",
        help="Directory for generated JSONL output. Defaults to <base-dir>/qa.",
    )
    parser.add_argument(
        "--logs-dir",
        help="Directory for progress and failure logs. Defaults to <base-dir>/logs.",
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("RETRO_API_URL", "http://127.0.0.1:1234/api/v0/chat/completions"),
        help="Chat completions endpoint for the local generation model.",
    )
    parser.add_argument(
        "--model-id",
        default=os.getenv("RETRO_QA_MODEL_ID", "gpt-oss-20b"),
        help="Model identifier to pass to the chat API.",
    )
    parser.add_argument(
        "--input-extension",
        default=".txt",
        help="Section file extension to process.",
    )
    parser.add_argument(
        "--safe-context-ratio",
        type=float,
        default=0.8,
        help="Fraction of model context to reserve for prompt input.",
    )
    parser.add_argument(
        "--default-context",
        type=int,
        default=4096,
        help="Fallback context length if the model endpoint cannot be queried.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for Q/A generation.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per chunk when the API fails or returns invalid JSON.",
    )
    args = parser.parse_args()

    args.base_dir = args.base_dir.resolve()
    args.sections_dir = resolve_path(args.sections_dir or "sections", args.base_dir)
    args.qa_dir = resolve_path(args.qa_dir or "qa", args.base_dir)
    args.logs_dir = resolve_path(args.logs_dir or "logs", args.base_dir)
    args.progress_file = args.logs_dir / "qa_progress.json"
    args.error_log = args.logs_dir / "qa_failures.log"
    return args


def make_models_url(api_url, model_id):
    if api_url.endswith("/chat/completions"):
        return api_url.rsplit("/chat/completions", 1)[0] + f"/models/{model_id}"
    raise ValueError(
        "Could not infer models endpoint from api-url; expected a URL ending in /chat/completions."
    )


def log_error(error_log, message):
    with error_log.open("a", encoding="utf-8") as handle:
        handle.write(message.strip() + "\n")


def get_context_length(api_url, model_id, default_context, error_log):
    try:
        response = requests.get(make_models_url(api_url, model_id), timeout=10)
        response.raise_for_status()
        data = response.json()
        context_length = data.get("max_context_length")
        if context_length is None:
            context_length = data.get("model_info", {}).get("context_length")
        return int(context_length or default_context)
    except Exception as exc:
        log_error(error_log, f"Context query failed: {exc}")
        return default_context


def flatten(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(map(str, value))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def sanitize(obj):
    question_or_verb = re.compile(
        r"^(Explain|Describe|List|Give|Summarize|Create|Formulate|Generate|Why|How|What|When|Where|Who)",
        re.I,
    )
    if not isinstance(obj, dict):
        return None
    instruction = flatten(obj.get("instruction")).strip()
    input_text = flatten(obj.get("input")).strip()
    output = flatten(obj.get("output")).strip()
    if not instruction or not output:
        return None
    if any("\n" in value or "\r" in value for value in (instruction, input_text, output)):
        return None
    if not (instruction.endswith("?") or question_or_verb.match(instruction)):
        return None
    return {"instruction": instruction, "input": input_text, "output": output}


def chunk_text_dynamic(text, safe_context):
    words = re.split(r"\s+", text)
    count = len(words)
    max_words = int(safe_context / 1.3)
    target = max(500, min(max_words, count // 5 if count > max_words * 5 else max_words))
    overlap = int(target * 0.1)
    chunks = []
    index = 0
    while index < len(words):
        chunks.append(" ".join(words[index:index + target]))
        if index + target >= len(words):
            break
        index += target - overlap
    return chunks


def query(api_url, model_id, chunk, max_tokens, temperature):
    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": chunk.strip()},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    response = requests.post(api_url, json=body, timeout=(10, None))
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def extract_json(text):
    text = re.sub(r"<\s*/?\s*think\s*>", "", text, flags=re.I)
    text = re.sub(r",\s*]", "]", text)
    try:
        start = text.index("[")
        end = text.rindex("]") + 1
        value = json.loads(text[start:end])
        return value if isinstance(value, list) else None
    except Exception:
        return None


def load_progress(progress_file):
    if not progress_file.exists():
        return {}
    try:
        return json.loads(progress_file.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_progress(progress_file, progress):
    progress_file.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def process_txt(config, in_path, out_path, rel_path, progress, safe_context):
    text = in_path.read_text(encoding="utf-8")
    chunks = chunk_text_dynamic(text, safe_context=safe_context)

    already = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                    already.add((record["instruction"], record["output"]))
                except Exception:
                    continue

    done_chunks = progress.get(rel_path, 0)

    with out_path.open("a", encoding="utf-8") as handle:
        for chunk_index, chunk in enumerate(chunks, start=1):
            if chunk_index <= done_chunks:
                continue
            if len(chunk.strip()) < 50:
                log_error(config.error_log, f"{rel_path} chunk {chunk_index} skipped (too small)")
                progress[rel_path] = chunk_index
                save_progress(config.progress_file, progress)
                continue

            label = f"{rel_path} : chunk {chunk_index}/{len(chunks)}"
            print(label)
            for _ in range(config.max_retries):
                try:
                    raw = query(
                        config.api_url,
                        config.model_id,
                        chunk,
                        max_tokens=safe_context,
                        temperature=config.temperature,
                    )
                except Exception as exc:
                    log_error(config.error_log, f"{label} HTTP {exc}")
                    time.sleep(1)
                    continue

                data = extract_json(raw)
                if not data:
                    log_error(config.error_log, f"{label} no JSON detected")
                    time.sleep(1)
                    continue

                added = 0
                for obj in data:
                    clean = sanitize(obj)
                    if not clean:
                        continue
                    key = (clean["instruction"], clean["output"])
                    if key in already:
                        continue
                    handle.write(json.dumps(clean, ensure_ascii=False) + "\n")
                    handle.flush()
                    already.add(key)
                    added += 1

                if added:
                    progress[rel_path] = chunk_index
                    save_progress(config.progress_file, progress)
                    break

                log_error(config.error_log, f"{label} no valid objects")
                time.sleep(1)


def main():
    args = parse_args()
    args.qa_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)

    progress = load_progress(args.progress_file)

    def handle_ctrl_c(_sig, _frame):
        print("\nCtrl-C detected, saving progress ...")
        save_progress(args.progress_file, progress)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_ctrl_c)

    max_context = get_context_length(
        args.api_url,
        args.model_id,
        default_context=args.default_context,
        error_log=args.error_log,
    )
    safe_context = int(max_context * args.safe_context_ratio)
    print(f"Detected model context {max_context}, using {safe_context} safe tokens")

    for root, _, files in os.walk(args.sections_dir):
        rel_dir = os.path.relpath(root, args.sections_dir)
        out_dir = args.qa_dir / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in sorted(files):
            if not name.lower().endswith(args.input_extension.lower()):
                continue
            in_path = Path(root) / name
            out_path = out_dir / f"{Path(name).stem}.jsonl"
            rel_path = os.path.relpath(in_path, args.sections_dir)
            process_txt(args, in_path, out_path, rel_path, progress, safe_context)

    save_progress(args.progress_file, progress)
    print(f"\nAll Q/A generation complete -> {args.qa_dir}")


if __name__ == "__main__":
    main()
