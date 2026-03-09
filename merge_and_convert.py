import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def resolve_path(value, base_dir):
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def find_quantize_binary(llama_cpp_dir):
    candidates = [
        llama_cpp_dir / "build" / "bin" / "llama-quantize.exe",
        llama_cpp_dir / "build" / "bin" / "llama-quantize",
        llama_cpp_dir / "build" / "bin" / "Release" / "llama-quantize.exe",
        llama_cpp_dir / "build" / "bin" / "Release" / "llama-quantize",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def parse_args():
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter into a local model and optionally convert it to GGUF."
    )
    parser.add_argument(
        "--base-dir",
        default=repo_root,
        type=Path,
        help="Repository root. Defaults to the directory containing this script.",
    )
    parser.add_argument(
        "--model-dir",
        help="Local Hugging Face model directory. Defaults to <base-dir>/qwen3_8b.",
    )
    parser.add_argument(
        "--adapter-dir",
        help="LoRA adapter directory. Defaults to <output-root>/<run-name>/adapter.",
    )
    parser.add_argument(
        "--output-root",
        help="Directory containing LoRA training runs. Defaults to <base-dir>/lora_fpu64.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Training run name. Defaults to <model-name>_lora.",
    )
    parser.add_argument(
        "--merged-dir",
        help="Directory to write the merged model. Defaults to <base-dir>/merged/<model-name>-merged.",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        help="Path to the local llama.cpp checkout. Defaults to <base-dir>/llama.cpp.",
    )
    parser.add_argument(
        "--gguf-out",
        help="Path for the unquantized GGUF output. Defaults to <merged-dir>/<model-name>-f16.gguf.",
    )
    parser.add_argument(
        "--convert-outtype",
        default="f16",
        help="`convert_hf_to_gguf.py` outtype. Defaults to f16.",
    )
    parser.add_argument(
        "--quantize-type",
        default="",
        help="Optional quantization type, for example `Q8_0` or `Q4_K_M`.",
    )
    parser.add_argument(
        "--quantized-out",
        help="Optional path for the quantized GGUF output. Defaults next to --gguf-out.",
    )
    parser.add_argument(
        "--quantize-binary",
        help="Optional explicit path to `llama-quantize`.",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Merge the adapter but do not run GGUF conversion.",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip merging and only run conversion against --merged-dir.",
    )
    args = parser.parse_args()

    args.base_dir = args.base_dir.resolve()
    args.model_dir = resolve_path(args.model_dir or "qwen3_8b", args.base_dir)
    args.output_root = resolve_path(args.output_root or "lora_fpu64", args.base_dir)
    if not args.run_name:
        args.run_name = f"{args.model_dir.name}_lora"
    args.adapter_dir = resolve_path(
        args.adapter_dir or (args.output_root / args.run_name / "adapter"),
        args.base_dir,
    )
    default_merged = args.base_dir / "merged" / f"{args.model_dir.name}-merged"
    args.merged_dir = resolve_path(args.merged_dir or default_merged, args.base_dir)
    args.llama_cpp_dir = resolve_path(args.llama_cpp_dir or "llama.cpp", args.base_dir)

    default_gguf = args.merged_dir / f"{args.model_dir.name}-f16.gguf"
    args.gguf_out = resolve_path(args.gguf_out or default_gguf, args.base_dir)

    if args.quantized_out:
        args.quantized_out = resolve_path(args.quantized_out, args.base_dir)
    elif args.quantize_type:
        suffix = args.quantize_type.lower()
        args.quantized_out = args.gguf_out.with_name(
            f"{args.gguf_out.stem}-{suffix}{args.gguf_out.suffix}"
        )
    else:
        args.quantized_out = None

    if args.quantize_binary:
        args.quantize_binary = resolve_path(args.quantize_binary, args.base_dir)
    else:
        args.quantize_binary = find_quantize_binary(args.llama_cpp_dir)

    return args


def run_command(command):
    printable = " ".join(f'"{part}"' if " " in part else part for part in command)
    print(f"\n> {printable}")
    subprocess.run(command, check=True)


def merge_adapter(model_dir, adapter_dir, merged_dir):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, adapter_dir, local_files_only=True)
    model = model.merge_and_unload()
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"Merged model saved to {merged_dir}")


def convert_to_gguf(llama_cpp_dir, merged_dir, gguf_out, outtype):
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise SystemExit(f"Missing convert script: {convert_script}")
    gguf_out.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            sys.executable,
            str(convert_script),
            str(merged_dir),
            "--outfile",
            str(gguf_out),
            "--outtype",
            outtype,
        ]
    )


def quantize_gguf(quantize_binary, gguf_out, quantized_out, quantize_type):
    if not quantize_binary:
        raise SystemExit(
            "Quantization requested, but `llama-quantize` was not found. "
            "Build llama.cpp first or pass --quantize-binary."
        )
    quantized_out.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            str(quantize_binary),
            str(gguf_out),
            str(quantized_out),
            quantize_type,
        ]
    )


def main():
    args = parse_args()

    if not args.skip_merge:
        if not args.model_dir.exists():
            raise SystemExit(f"Missing model directory: {args.model_dir}")
        if not args.adapter_dir.exists():
            raise SystemExit(f"Missing adapter directory: {args.adapter_dir}")
        merge_adapter(args.model_dir, args.adapter_dir, args.merged_dir)
    elif not args.merged_dir.exists():
        raise SystemExit(
            f"--skip-merge was set, but merged directory does not exist: {args.merged_dir}"
        )

    if args.skip_convert:
        return

    convert_to_gguf(
        args.llama_cpp_dir,
        args.merged_dir,
        args.gguf_out,
        args.convert_outtype,
    )
    print(f"Unquantized GGUF saved to {args.gguf_out}")

    if args.quantize_type:
        quantize_gguf(
            args.quantize_binary,
            args.gguf_out,
            args.quantized_out,
            args.quantize_type,
        )
        print(f"Quantized GGUF saved to {args.quantized_out}")


if __name__ == "__main__":
    main()
