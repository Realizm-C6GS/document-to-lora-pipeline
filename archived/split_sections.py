import re
import os

dump_path = r"D:\ai\retro\microarchitecture_dump.txt"
out_dir = r"D:\ai\retro\sections"

os.makedirs(out_dir, exist_ok=True)

print(f"Loading dump: {dump_path}")
with open(dump_path, "r", encoding="utf-8") as f:
    full_text = f.read()

# Regex: capture numbered section headings like 3.4 or 4.2.1
pattern = re.compile(r'^\s*(\d+(\.\d+)+?)\s+([^\n]+)', re.MULTILINE)
matches = list(pattern.finditer(full_text))

if not matches:
    raise SystemExit("No numbered section headings found. Check your dump formatting or regex pattern.")

for i, m in enumerate(matches):
    sec_num = m.group(1)
    sec_title = m.group(3).strip()
    start = m.start()
    end = matches[i+1].start() if i+1 < len(matches) else len(full_text)

    sec_text = full_text[start:end].strip()

    # --- CLEANUP FIXES ---

    # 1. Remove dotted leaders or headers like ..... 123
    sec_title = re.sub(r'\.{2,}', '', sec_title)
    sec_text = re.sub(r'\.{10,}', '', sec_text)          # kill dotted page headers inside body
    sec_text = re.sub(r'\n\s*\d+\s*\n', '\n', sec_text)  # remove bare page numbers on their own lines

    # 2. Remove trailing page numbers from title like "something ......... 132"
    sec_title = re.sub(r'\s*\d+\s*$', '', sec_title)

    # 3. Basic sanitization for filenames
    filename = f"{sec_num}_{sec_title}"
    filename = filename.strip()
    filename = re.sub(r'[\\/*?:"<>|]', '_', filename)
    filename = re.sub(r'\s+', '_', filename)
    filename += ".txt"

    out_path = os.path.join(out_dir, filename)

    # --- WRITE CLEAN FILE ---
    with open(out_path, "w", encoding="utf-8") as outf:
        outf.write(sec_text)

    print(f"Wrote: {filename}")

print(f"\nSplit {len(matches)} sections into {out_dir}")
