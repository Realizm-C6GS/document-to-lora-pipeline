import argparse
from pathlib import Path


def resolve_path(value, base_dir):
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def parse_args():
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train a LoRA adapter on JSONL Q/A data using a local Hugging Face model."
    )
    parser.add_argument(
        "--base-dir",
        default=repo_root,
        type=Path,
        help="Repository root. Defaults to the directory containing this script.",
    )
    parser.add_argument(
        "--qa-dir",
        help="Directory containing input JSONL files. Defaults to <base-dir>/qa.",
    )
    parser.add_argument(
        "--model-dir",
        help="Local Hugging Face model directory. Defaults to <base-dir>/qwen3_8b.",
    )
    parser.add_argument(
        "--output-root",
        help="Directory where training outputs should be written. Defaults to <base-dir>/lora_fpu64.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Subdirectory name to create under the output root. Defaults to <model-name>_lora.",
    )
    parser.add_argument("--max-len", type=int, default=1024, help="Training sequence length.")
    parser.add_argument("--epochs", type=float, default=1.0, help="Number of training epochs.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device batch size.",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Trainer logging interval.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=200,
        help="Trainer checkpoint interval.",
    )
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout.",
    )
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated module names to target with LoRA.",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Compute dtype for 4-bit quantization.",
    )
    parser.add_argument(
        "--dataset-glob",
        default="**/*.jsonl",
        help="Glob pattern under qa-dir for training files.",
    )
    args = parser.parse_args()

    args.base_dir = args.base_dir.resolve()
    args.qa_dir = resolve_path(args.qa_dir or "qa", args.base_dir)
    args.model_dir = resolve_path(args.model_dir or "qwen3_8b", args.base_dir)
    args.output_root = resolve_path(args.output_root or "lora_fpu64", args.base_dir)
    if not args.run_name:
        args.run_name = f"{args.model_dir.name}_lora"
    return args


def build_prompt(example):
    instruction = example.get("instruction", "").strip()
    input_text = example.get("input", "").strip()
    output = example.get("output", "").strip()
    prompt = f"Instruction:\n{instruction}\n"
    if input_text:
        prompt += f"Input:\n{input_text}\n"
    prompt += "Response:"
    example["text"] = prompt + output
    return example


def main():
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    data_files = sorted(str(path) for path in args.qa_dir.glob(args.dataset_glob))
    if not data_files:
        raise SystemExit(
            f"No Q/A files found under {args.qa_dir}. Run generate_qa_all_dyn.py first."
        )

    print(f"Found {len(data_files)} Q/A files")
    dataset = load_dataset("json", data_files=data_files, split="train")
    dataset = dataset.map(build_prompt, remove_columns=dataset.column_names)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_len,
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    tokenized_dataset = dataset.map(tokenize, batched=True)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype_map[args.compute_dtype],
    )

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    target_modules = [name.strip() for name in args.target_modules.split(",") if name.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        param.requires_grad = "lora" in name

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    train_out = args.output_root / args.run_name
    training_args = TrainingArguments(
        output_dir=str(train_out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        fp16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
    trainer.train()

    adapter_dir = train_out / "adapter"
    model.save_pretrained(adapter_dir)
    print(f"LoRA adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()
