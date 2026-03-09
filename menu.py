import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_header():
    print(r"+------------------------------------------------------+")
    print(r"|                  RETRO WORKFLOW MENU                 |")
    print(r"+------------------------------------------------------+")


def default_config():
    model_name = "qwen3_8b"
    run_name = "qwen3_8b_lora"
    merged_dir = str(Path("merged") / f"{model_name}-merged")
    gguf_out = str(Path(merged_dir) / f"{model_name}-f16.gguf")
    return {
        "base_dir": str(REPO_ROOT),
        "pdf_dir": "pdfs",
        "sections_dir": "sections",
        "qa_dir": "qa",
        "logs_dir": "logs",
        "api_url": os.getenv("RETRO_API_URL", "http://127.0.0.1:1234/api/v0/chat/completions"),
        "qa_model_id": os.getenv("RETRO_QA_MODEL_ID", "gpt-oss-20b"),
        "input_extension": ".txt",
        "model_dir": model_name,
        "output_root": "lora_fpu64",
        "run_name": run_name,
        "max_len": "1024",
        "target_modules": "q_proj,k_proj,v_proj,o_proj",
        "adapter_dir": str(Path("lora_fpu64") / run_name / "adapter"),
        "merged_dir": merged_dir,
        "llama_cpp_dir": "llama.cpp",
        "gguf_out": gguf_out,
        "quantize_type": "",
        "quantized_out": "",
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive ASCII menu for the retro training workflow."
    )
    parser.add_argument(
        "--show-defaults",
        action="store_true",
        help="Print the default menu configuration and exit.",
    )
    return parser.parse_args()


FIELDS = [
    ("pdf_dir", "PDF directory"),
    ("sections_dir", "Sections directory"),
    ("qa_dir", "QA directory"),
    ("logs_dir", "Logs directory"),
    ("api_url", "QA API URL"),
    ("qa_model_id", "QA model ID"),
    ("input_extension", "Input extension"),
    ("model_dir", "Base model directory"),
    ("output_root", "Training output root"),
    ("run_name", "Training run name"),
    ("max_len", "Training max length"),
    ("target_modules", "LoRA target modules"),
    ("adapter_dir", "Adapter directory"),
    ("merged_dir", "Merged model directory"),
    ("llama_cpp_dir", "llama.cpp directory"),
    ("gguf_out", "GGUF output path"),
    ("quantize_type", "Quantize type"),
    ("quantized_out", "Quantized GGUF path"),
]


def build_split_command(config):
    return [
        PYTHON,
        str(REPO_ROOT / "dump_split.py"),
        "--base-dir",
        config["base_dir"],
        "--pdf-dir",
        config["pdf_dir"],
        "--sections-dir",
        config["sections_dir"],
        "--logs-dir",
        config["logs_dir"],
    ]


def build_generate_command(config):
    return [
        PYTHON,
        str(REPO_ROOT / "generate_qa_all_dyn.py"),
        "--base-dir",
        config["base_dir"],
        "--sections-dir",
        config["sections_dir"],
        "--qa-dir",
        config["qa_dir"],
        "--logs-dir",
        config["logs_dir"],
        "--api-url",
        config["api_url"],
        "--model-id",
        config["qa_model_id"],
        "--input-extension",
        config["input_extension"],
    ]


def build_train_command(config):
    return [
        PYTHON,
        str(REPO_ROOT / "train_lora.py"),
        "--base-dir",
        config["base_dir"],
        "--qa-dir",
        config["qa_dir"],
        "--model-dir",
        config["model_dir"],
        "--output-root",
        config["output_root"],
        "--run-name",
        config["run_name"],
        "--max-len",
        config["max_len"],
        "--target-modules",
        config["target_modules"],
    ]


def build_merge_command(config):
    command = [
        PYTHON,
        str(REPO_ROOT / "merge_and_convert.py"),
        "--base-dir",
        config["base_dir"],
        "--model-dir",
        config["model_dir"],
        "--adapter-dir",
        config["adapter_dir"],
        "--merged-dir",
        config["merged_dir"],
        "--llama-cpp-dir",
        config["llama_cpp_dir"],
        "--gguf-out",
        config["gguf_out"],
    ]
    if config["quantize_type"].strip():
        command.extend(["--quantize-type", config["quantize_type"].strip()])
    if config["quantized_out"].strip():
        command.extend(["--quantized-out", config["quantized_out"].strip()])
    return command


def run_command(command):
    print("")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    print("")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}")
    input("\nPress Enter to continue...")


def show_config(config):
    clear_screen()
    print_header()
    print("Current settings:\n")
    for index, (key, label) in enumerate(FIELDS, start=1):
        print(f"{index:2d}. {label:<22} {config[key]}")
    input("\nPress Enter to return...")


def edit_config(config):
    while True:
        clear_screen()
        print_header()
        print("Edit settings:\n")
        for index, (key, label) in enumerate(FIELDS, start=1):
            print(f"{index:2d}. {label:<22} {config[key]}")
        print("\nR. Reset to defaults")
        print("Q. Back")

        choice = input("\nSelect a field: ").strip().lower()
        if choice == "q":
            return
        if choice == "r":
            config.clear()
            config.update(default_config())
            continue
        if not choice.isdigit():
            continue

        index = int(choice) - 1
        if index < 0 or index >= len(FIELDS):
            continue
        key, label = FIELDS[index]
        current = config[key]
        value = input(f"{label} [{current}]: ").strip()
        if value:
            config[key] = value


def show_commands(config):
    clear_screen()
    print_header()
    commands = [
        ("Split PDFs", build_split_command(config)),
        ("Generate QA", build_generate_command(config)),
        ("Train LoRA", build_train_command(config)),
        ("Merge + Convert", build_merge_command(config)),
    ]
    for title, command in commands:
        print(f"{title}:")
        print("  " + " ".join(f'"{part}"' if " " in part else part for part in command))
        print("")
    input("Press Enter to return...")


def main():
    args = parse_args()
    config = default_config()

    if args.show_defaults:
        for key, value in config.items():
            print(f"{key}={value}")
        return

    if not sys.stdin.isatty():
        raise SystemExit(
            "menu.py is interactive. Run it in a terminal, or use --show-defaults."
        )

    while True:
        clear_screen()
        print_header()
        print("1. Split PDFs into sections")
        print("2. Generate QA JSONL")
        print("3. Train LoRA")
        print("4. Merge model and convert to GGUF")
        print("5. Edit settings")
        print("6. Show commands")
        print("7. Show current settings")
        print("Q. Quit")

        choice = input("\nSelect an option: ").strip().lower()
        if choice == "1":
            run_command(build_split_command(config))
        elif choice == "2":
            run_command(build_generate_command(config))
        elif choice == "3":
            run_command(build_train_command(config))
        elif choice == "4":
            run_command(build_merge_command(config))
        elif choice == "5":
            edit_config(config)
        elif choice == "6":
            show_commands(config)
        elif choice == "7":
            show_config(config)
        elif choice == "q":
            return


if __name__ == "__main__":
    main()
