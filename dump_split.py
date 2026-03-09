import argparse
import re
from pathlib import Path


def resolve_path(value, base_dir):
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def parse_args():
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Extract PDFs from the repo and split them into section text files."
    )
    parser.add_argument(
        "--base-dir",
        default=repo_root,
        type=Path,
        help="Repository root. Defaults to the directory containing this script.",
    )
    parser.add_argument(
        "--pdf-dir",
        help="Directory containing input PDFs. Defaults to <base-dir>/pdfs.",
    )
    parser.add_argument(
        "--sections-dir",
        help="Directory for section text output. Defaults to <base-dir>/sections.",
    )
    parser.add_argument(
        "--logs-dir",
        help="Directory for extraction logs. Defaults to <base-dir>/logs.",
    )
    parser.add_argument(
        "--min-section-words",
        type=int,
        default=100,
        help="Minimum words required before a section is written directly.",
    )
    parser.add_argument(
        "--fallback-part-size",
        type=int,
        default=4000,
        help="Approximate character size for fallback chunking.",
    )
    args = parser.parse_args()

    args.base_dir = args.base_dir.resolve()
    args.pdf_dir = resolve_path(args.pdf_dir or "pdfs", args.base_dir)
    args.sections_dir = resolve_path(args.sections_dir or "sections", args.base_dir)
    args.logs_dir = resolve_path(args.logs_dir or "logs", args.base_dir)
    return args


def safe_filename(value):
    value = re.sub(r'[\\/*?:"<>|]', "_", value)
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"\.{2,}", "_", value)
    return value.strip("_")


def split_sections_from_text(text):
    pattern = re.compile(r"^\s*(\d+(?:\.\d+)+)\s+([^\n]+)", re.MULTILINE)
    matches = list(pattern.finditer(text))
    splits = []

    for index, match in enumerate(matches):
        sec_num = match.group(1)
        sec_title = match.group(2)
        sec_title = re.sub(r"\.{2,}", " ", sec_title)
        sec_title = re.sub(r"\s*\d+\s*$", "", sec_title)
        sec_title = sec_title.strip()

        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        title = safe_filename(f"{sec_num}_{sec_title}")
        splits.append((title, content))
    return splits


def fallback_split(text, min_section_words, part_size):
    splits = []
    text = text.strip()
    if not text:
        return splits

    for index, offset in enumerate(range(0, len(text), part_size), start=1):
        part = text[offset:offset + part_size].strip()
        if len(part.split()) >= min_section_words:
            splits.append((f"part_{index:03d}", part))
    return splits


def dump_pdf(pdf_path, out_root, min_section_words, fallback_part_size):
    import pdfplumber

    pdf_name = pdf_path.stem
    out_dir = out_root / pdf_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "errors.log"

    with log_path.open("w", encoding="utf-8") as log:
        try:
            print(f"Processing {pdf_name}")
            pages = []
            with pdfplumber.open(pdf_path) as pdf:
                for page_number, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if not text:
                        log.write(f"Warning: page {page_number} had no text\n")
                        continue
                    text = "".join(ch for ch in text if ord(ch) >= 32 or ch in "\n\t")
                    pages.append(text)

            full_text = "\n\n".join(pages)
            if not full_text.strip():
                log.write("Fatal: no text extracted.\n")
                return

            sections = split_sections_from_text(full_text)
            unassigned = []

            if not sections:
                log.write("No headings found; using fallback split.\n")
                sections = fallback_split(
                    full_text,
                    min_section_words=min_section_words,
                    part_size=fallback_part_size,
                )

            for title, content in sections:
                words = len(content.split())
                if words < min_section_words:
                    log.write(
                        f"Section '{title}' very short ({words}), will merge later.\n"
                    )
                    unassigned.append(content)
                    continue

                output_path = out_dir / f"{title}.txt"
                output_path.write_text(content, encoding="utf-8")
                print(f"  Wrote {output_path}")

            if unassigned:
                merged_text = "\n\n".join(unassigned).strip()
                log.write(
                    f"Merging {len(unassigned)} short sections into fallback chunks.\n"
                )
                for index, (_, part) in enumerate(
                    fallback_split(
                        merged_text,
                        min_section_words=min_section_words,
                        part_size=fallback_part_size,
                    ),
                    start=1,
                ):
                    output_path = out_dir / f"merged_part_{index:03d}.txt"
                    output_path.write_text(part, encoding="utf-8")
                    print(f"  Fallback chunk: {output_path}")

        except Exception as exc:
            log.write(f"Fatal error for {pdf_name}: {exc}\n")
            print(f"[!] Error processing {pdf_name}, see {log_path}")


def main():
    args = parse_args()
    args.pdf_dir.mkdir(parents=True, exist_ok=True)
    args.sections_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(
        path for path in args.pdf_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"
    )
    if not pdf_files:
        print(f"No PDFs in {args.pdf_dir}")
        return

    for pdf_path in pdf_files:
        dump_pdf(
            pdf_path,
            args.sections_dir,
            min_section_words=args.min_section_words,
            fallback_part_size=args.fallback_part_size,
        )
    print(f"\nAll PDFs processed -> {args.sections_dir}")


if __name__ == "__main__":
    main()
