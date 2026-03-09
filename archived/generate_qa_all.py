import os, re, json, time, signal, sys, requests

# === basic paths =============================================================
BASE_DIR      = r"I:\ai\retro"
SECTIONS_DIR  = os.path.join(BASE_DIR, "sections")
QA_DIR        = os.path.join(BASE_DIR, "qa")
LOGS_DIR      = os.path.join(BASE_DIR, "logs")
os.makedirs(QA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# === model / API settings ====================================================
API_URL       = "http://127.0.0.1:1234/api/v0/chat/completions"
MODEL_ID      = "gpt-oss-20b"   # whatever LM Studio lists in /api/v0/models
USER_CONTEXT  = 8192          # <<==== set this manually (e.g. 8192, 12288)
SAFE_CONTEXT  = int(USER_CONTEXT * 0.8)  # keep 20% free for output
print(f"Using configured context {USER_CONTEXT}, safe token limit {SAFE_CONTEXT}")

PROGRESS_FILE = os.path.join(LOGS_DIR, "qa_progress.json")
ERROR_LOG     = os.path.join(LOGS_DIR, "qa_failures.log")

SYSTEM_PROMPT = """
You are generating training data.
Return ONLY a JSON array of 6-8 objects, each having exactly:
  instruction • input • output
No markdown, bullets, or commentary.
Each value must be on one line (no \\n).
The instruction should be a question or start with an imperative verb (Explain, Describe, List, Give, Summarize…).
"""

# === utility functions =======================================================

def log_error(msg):
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(msg.strip() + "\n")

def flatten(v):
    if v is None: return ""
    if isinstance(v, (list, tuple)): return " ".join(map(str, v))
    if isinstance(v, dict): return json.dumps(v, ensure_ascii=False)
    return str(v)

def sanitize(obj):
    _qverb = re.compile(
        r"^(Explain|Describe|List|Give|Summarize|Create|Formulate|Generate|Why|How|What|When|Where|Who)",
        re.I,
    )
    if not isinstance(obj, dict):
        return None
    instr = flatten(obj.get("instruction")).strip()
    inp   = flatten(obj.get("input")).strip()
    out   = flatten(obj.get("output")).strip()
    if not instr or not out:
        return None
    if any("\n" in s or "\r" in s for s in (instr, inp, out)):
        return None
    if not (instr.endswith("?") or _qverb.match(instr)):
        return None
    return {"instruction": instr, "input": inp, "output": out}

# === dynamic chunking ========================================================

def chunk_text_dynamic(text):
    words = re.split(r"\s+", text)
    count = len(words)
    # token ≈ words*1.3, target ~80% of SAFE_CONTEXT
    max_words = int(SAFE_CONTEXT / 1.3)
    # try to make between 3 and 10 chunks for huge docs
    target = max(500, min(max_words, count // 5 if count > max_words * 5 else max_words))
    overlap = int(target * 0.1)
    out = []
    i = 0
    while i < len(words):
        out.append(" ".join(words[i:i + target]))
        if i + target >= len(words): break
        i += target - overlap
    return out

# === make chat request =======================================================

def query(chunk, max_tokens=SAFE_CONTEXT):
    body = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": chunk.strip()}
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens,
        "stream": False
    }
    r = requests.post(API_URL, json=body, timeout=(10, None))
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# === progress file management ===============================================

progress = {}
if os.path.exists(PROGRESS_FILE):
    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            progress = json.load(f)
    except Exception:
        progress = {}

def save_progress():
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)

def handle_ctrl_c(sig, frame):
    print("\nCtrl‑C detected, saving progress …")
    save_progress()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_ctrl_c)

# === helpers =================================================================

def extract_json(txt):
    txt = re.sub(r"<\s*/?\s*think\s*>", "", txt, flags=re.I)
    txt = re.sub(r",\s*]", "]", txt)
    try:
        start = txt.index("["); end = txt.rindex("]") + 1
        arr = json.loads(txt[start:end])
        return arr if isinstance(arr, list) else None
    except Exception:
        return None

# === main file loop ==========================================================

def process_txt(in_path, out_path, rel_path):
    with open(in_path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_text_dynamic(text)

    already = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    already.add((d["instruction"], d["output"]))
                except Exception:
                    pass

    done_chunks = progress.get(rel_path, 0)

    with open(out_path, "a", encoding="utf-8") as outf:
        for i, chunk in enumerate(chunks, 1):
            if i <= done_chunks:
                continue
            if len(chunk.strip()) < 50:
                log_error(f"{rel_path} chunk {i} skipped (too small)")
                progress[rel_path] = i; save_progress(); continue

            label = f"{rel_path} : chunk {i}/{len(chunks)}"
            print(label)
            for attempt in range(3):
                try:
                    raw = query(chunk)
                except Exception as e:
                    log_error(f"{label} HTTP {e}")
                    time.sleep(1)
                    continue
                data = extract_json(raw)
                if not data:
                    log_error(f"{label} no JSON detected")
                    time.sleep(1)
                    continue
                added = 0
                for obj in data:
                    clean = sanitize(obj)
                    if not clean: continue
                    key = (clean["instruction"], clean["output"])
                    if key in already: continue
                    outf.write(json.dumps(clean, ensure_ascii=False) + "\n")
                    outf.flush()
                    already.add(key)
                    added += 1
                if added:
                    progress[rel_path] = i
                    save_progress()
                    break
                else:
                    log_error(f"{label} no valid objects")
                    time.sleep(1)

# === entry ===================================================================

def main():
    for root, _, files in os.walk(SECTIONS_DIR):
        rel = os.path.relpath(root, SECTIONS_DIR)
        out_dir = os.path.join(QA_DIR, rel)
        os.makedirs(out_dir, exist_ok=True)
        for f in files:
            if not f.lower().endswith(".txt"):
                continue
            in_path = os.path.join(root, f)
            out_path = os.path.join(out_dir, os.path.splitext(f)[0] + ".jsonl")
            rel_path = os.path.relpath(in_path, SECTIONS_DIR)
            process_txt(in_path, out_path, rel_path)
    save_progress()
    print("\nAll Q/A generation complete →", QA_DIR)

if __name__ == "__main__":
    main()
