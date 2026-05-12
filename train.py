"""
train.py — LoRA Fine-tuning Script
===================================
Fine-tunes LLaMA-7B with LoRA on a chosen instruction dataset
(Dolly, Alpaca, or OpenAssistant).

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --dataset alpaca
"""

import argparse
import os
import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer


# ──────────────────────────────────────────────
# Argument Parsing
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for LLM Calibration")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override dataset: dolly | alpaca | openassistant")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    return parser.parse_args()


# ──────────────────────────────────────────────
# Config Loader
# ──────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# Dataset Formatters
# ──────────────────────────────────────────────
SYSTEM_PROMPT = "You are a helpful assistant."

def format_dolly(example: dict) -> dict:
    instruction = example.get("instruction", "")
    context = example.get("context", "")
    response = example.get("response", "")
    if context:
        text = (
            f"### System:\n{SYSTEM_PROMPT}\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{context}\n\n"
            f"### Response:\n{response}"
        )
    else:
        text = (
            f"### System:\n{SYSTEM_PROMPT}\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{response}"
        )
    return {"text": text}


def format_alpaca(example: dict) -> dict:
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    if input_text:
        text = (
            f"### System:\n{SYSTEM_PROMPT}\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )
    else:
        text = (
            f"### System:\n{SYSTEM_PROMPT}\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output}"
        )
    return {"text": text}


def format_oa(example: dict) -> dict:
    role = example.get("role", "")
    text_body = example.get("text", "")
    # Extract single-turn English conversations
    if role == "prompter":
        return {"text": None}  # skip prompter-only entries
    instruction = example.get("parent_id", "")
    text = (
        f"### System:\n{SYSTEM_PROMPT}\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{text_body}"
    )
    return {"text": text}


DATASET_CONFIG = {
    "dolly": {
        "hf_path": "databricks/databricks-dolly-15k",
        "split": "train",
        "formatter": format_dolly,
    },
    "alpaca": {
        "hf_path": "tatsu-lab/alpaca",
        "split": "train",
        "formatter": format_alpaca,
    },
    "openassistant": {
        "hf_path": "OpenAssistant/oasst1",
        "split": "train",
        "formatter": format_oa,
    },
}


# ──────────────────────────────────────────────
# Model & LoRA Setup
# ──────────────────────────────────────────────
def load_model_and_tokenizer(cfg: dict):
    model_name = cfg["model"]["base_model"]
    print(f"[INFO] Loading base model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.config.use_cache = False

    lora_cfg = cfg["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ──────────────────────────────────────────────
# Main Training Loop
# ──────────────────────────────────────────────
def main():
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides
    dataset_key = args.dataset or cfg["training"].get("dataset", "dolly").split("/")[-1]
    if dataset_key == "databricks-dolly-15k":
        dataset_key = "dolly"
    num_epochs = args.epochs or cfg["training"]["num_epochs"]
    output_dir = args.output_dir or cfg["training"]["output_dir"]
    output_dir = os.path.join(output_dir, f"{dataset_key}_lora")

    print(f"[INFO] Dataset  : {dataset_key}")
    print(f"[INFO] Epochs   : {num_epochs}")
    print(f"[INFO] Output   : {output_dir}")

    # Load dataset
    ds_cfg = DATASET_CONFIG[dataset_key]
    print(f"[INFO] Loading dataset from HuggingFace: {ds_cfg['hf_path']}")
    raw_dataset = load_dataset(ds_cfg["hf_path"], split=ds_cfg["split"])
    dataset = raw_dataset.map(ds_cfg["formatter"], remove_columns=raw_dataset.column_names)
    dataset = dataset.filter(lambda x: x["text"] is not None)
    dataset = dataset.shuffle(seed=cfg["training"]["seed"])
    print(f"[INFO] Dataset size: {len(dataset)} examples")

    # Load model
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Training arguments
    train_cfg = cfg["training"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_steps=train_cfg["warmup_steps"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        fp16=train_cfg["fp16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        save_strategy=train_cfg["save_strategy"],
        logging_steps=train_cfg["logging_steps"],
        report_to="none",
        seed=train_cfg["seed"],
    )

    # SFT Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=train_cfg["max_seq_length"],
        tokenizer=tokenizer,
    )

    print("[INFO] Starting training...")
    trainer.train()

    print(f"[INFO] Saving final model to {output_dir}/final")
    trainer.model.save_pretrained(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    print("[INFO] Training complete.")


if __name__ == "__main__":
    main()
