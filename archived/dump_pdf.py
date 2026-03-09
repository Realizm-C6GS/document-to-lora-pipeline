import pdfplumber
import os

pdf_path = r"D:\ai\retro\microarchitecture.pdf"
out_path = r"D:\ai\retro\microarchitecture_dump.txt"

os.makedirs(os.path.dirname(out_path), exist_ok=True)

pages_text = []
with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages, start=1):
        text = page.extract_text()
        if not text:
            print(f"Warning: page {i} had no text")
            continue
        pages_text.append(text)
        print(f"Extracted page {i}/{len(pdf.pages)}")

full_text = "\n\n".join(pages_text)

with open(out_path, "w", encoding="utf-8") as f:
    f.write(full_text)

print(f"\nSaved full text dump to: {out_path}")
print(f"Extracted {len(pages_text)} pages, {len(full_text)} characters total.")
