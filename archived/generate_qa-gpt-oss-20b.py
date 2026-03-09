import json, re, requests, time

API_URL  = "http://192.168.1.177:1234/v1/chat/completions"
MODEL    = "gpt-oss-20b"
TOKENS   = 8192
TIMEOUT  = 900
INPUT    = r"D:\ai\retro\p4-pipeline.txt"
OUTPUT   = r"D:\ai\retro\training_data.jsonl"

SYSTEM_PROMPT = """
You are generating training data.
Return ONLY a JSON array of 6-8 objects, each having exactly:
  instruction  • input • output
No markdown, bullets, numbering, or explanations.
Each value must be on a single logical line (no \\n inside any field).
The instruction should be phrased as a question *or* begin with an
imperative verb (Explain, Describe, List, Give, Summarize…).
"""

# ---------- helpers ----------

def chunk_text(text, words=800, overlap=150):
    w = re.split(r"\s+", text)
    chunks, start = [], 0
    while start < len(w):
        end = min(start + words, len(w))
        chunks.append(" ".join(w[start:end]))
        if end == len(w):
            break
        start = end - overlap
    return chunks

def query(chunk):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user",   "content": chunk.strip()}
        ],
        "temperature": 0.3,
        "max_tokens": TOKENS
    }
    r = requests.post(API_URL, json=body, timeout=(10, TIMEOUT))
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def extract_json(txt):
    txt = re.sub(r"<\s*/?\s*think\s*>", "", txt, flags=re.I)
    txt = txt.replace("\u2028", "").replace("\u2029", "")
    txt = re.sub(r",\s*]", "]", txt)
    for m in re.finditer(r"\[", txt):
        close = txt.rfind("]")
        if close == -1 or close <= m.start():
            continue
        candidate = txt[m.start():close + 1]
        if '"instruction"' not in candidate or '"output"' not in candidate:
            continue
        try:
            obj = json.loads(candidate)
            if isinstance(obj, list):
                return obj
        except Exception:
            continue
    return None

_qverb = re.compile(r"^(Explain|Describe|List|Give|Summarize|Create|Formulate|Generate|Why|How|What|When|Where|Who)", re.I)

def sanitize(item):
    if not isinstance(item, dict):
        return None
    instr = item.get("instruction", "").strip().rstrip()
    inp   = item.get("input", "").strip()
    out   = item.get("output", "").strip().rstrip()

    # reject blanks
    if not instr or not out:
        return None
    # reject embedded newlines
    if any("\n" in s or "\r" in s for s in (instr, inp, out)):
        return None
    # instruction must end in ? or start with imperative verb
    if not (instr.endswith("?") or _qverb.match(instr)):
        return None
    return {"instruction": instr, "input": inp, "output": out}

def run():
    text   = open(INPUT, "r", encoding="utf-8").read()
    chunks = chunk_text(text)
    seen   = set()
    good   = []

    for i, chunk in enumerate(chunks, 1):
        print(f"chunk {i}/{len(chunks)} …")
        for attempt in range(3):
            try:
                raw = query(chunk)
            except Exception as e:
                print("HTTP error:", e)
                time.sleep(1)
                continue

            data = extract_json(raw)
            if not data:
                print("retry – no JSON found")
                time.sleep(1)
                continue

            added = 0
            for obj in data:
                clean = sanitize(obj)
                if not clean:
                    continue
                key = (clean["instruction"], clean["output"])
                if key in seen:
                    continue
                seen.add(key)
                good.append(clean)
                added += 1

            if added:
                break
            else:
                print("retry – no valid objects")
                time.sleep(1)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for item in good:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"saved {len(good)} pairs → {OUTPUT}")

if __name__ == "__main__":
    run()
