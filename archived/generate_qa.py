import json, re, requests, time

API_URL  = "http://192.168.1.177:1234/v1/chat/completions"
MODEL    = "gpt-oss-20b"
TOKENS   = 8192          # raise if you still truncate
INPUT    = r"D:\ai\retro\p4-pipeline.txt"
OUTPUT   = r"D:\ai\retro\training_data.jsonl"

SYSTEM_PROMPT = """
You are generating structured training data.
Return ONLY a JSON array of objects:
{
  "instruction": "...",
  "input": "",
  "output": "..."
}
No markdown, no bullets, no prose. Aim for 8-10 Q/A pairs per chunk.
"""

# ---------- helpers ----------

def chunk_text(text, words=800, overlap=150):
    w = re.split(r"\s+", text)
    out, start = [], 0
    while start < len(w):
        end = min(start + words, len(w))
        out.append(" ".join(w[start:end]))
        if end == len(w): break
        start = end - overlap
    return out

def query(chunk):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user",   "content": chunk.strip()}
        ],
        "temperature": 0.3,
        "max_tokens": TOKENS,
        "response_format": { "type": "json_object" }   # <<< force JSON
    }
    r = requests.post(API_URL, json=body, timeout=1800)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# robust extractor
def grab_json(txt):
    txt = re.sub(r"<\s*/?\s*think\s*>", "", txt, flags=re.I)
    txt = txt.replace("\u2028", "").replace("\u2029", "")
    # try every [ ... ] span until one parses
    for m in re.finditer(r"\[", txt):
        close = txt.rfind("]")
        if close == -1 or close <= m.start():
            break
        cand = txt[m.start():close+1]
        if '"instruction"' not in cand:   # quick sanity
            continue
        cand = re.sub(r",\s*]", "]", cand)  # trailing comma fixer
        try:
            arr = json.loads(cand)
            if isinstance(arr, list):
                return arr
        except Exception:
            continue
    return None

# ---------- main ----------

def run():
    text   = open(INPUT, "r", encoding="utf-8").read()
    chunks = chunk_text(text)
    out    = []

    for i,c in enumerate(chunks,1):
        print(f"chunk {i}/{len(chunks)} …")
        for _ in range(3):
            try:
                raw  = query(c)
            except requests.HTTPError as e:
                print("HTTP", e); time.sleep(1); continue
            data = grab_json(raw)
            if data:
                out.extend(data)
                break
            print("retry bad JSON"); time.sleep(1)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for item in out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("saved", len(out), "pairs →", OUTPUT)

if __name__ == "__main__":
    run()
