"""
train.py — Dolly LoRA Fine-tuning
===================================
Converted from: lama7b-tuning-newdata.ipynb
Authors: Shahab Ahmad, Inam Ul Hassan, Ejaz Ulhaq
Course : NLP — MS AI, FAST NUCES Islamabad

Usage:
    python train.py
    python train.py --epochs 1
    python train.py --dataset alpaca
"""

import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/kaggle/input/models/metaresearch/llama-2/pytorch/7b-hf/1")
    parser.add_argument("--data_path",  type=str, default="/kaggle/working/datasets")
    parser.add_argument("--ckpt_path",  type=str, default="/kaggle/working/checkpoints")
    parser.add_argument("--dataset",    type=str, default="dolly", choices=["dolly", "alpaca", "oa"])
    parser.add_argument("--epochs",     type=int, default=3)
    parser.add_argument("--seed",       type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Seed set to {seed}")


def load_dolly(data_path, seed):
    dolly_path = f"{data_path}/dolly/dolly_data.csv"
    if os.path.exists(dolly_path):
        print(f"Loading Dolly from cache: {dolly_path}")
        dolly_df = pd.read_csv(dolly_path)
    else:
        print("Downloading Dolly from HuggingFace...")
        os.makedirs(f"{data_path}/dolly", exist_ok=True)
        dolly_raw = load_dataset("databricks/databricks-dolly-15k", split="train")

        def format_dolly(example):
            if example.get("context") and example["context"].strip():
                text = (
                    f"### Instruction:\n{example['instruction']}\n"
                    f"### Input:\n{example['context']}\n"
                    f"### Response:\n{example['response']}"
                )
            else:
                text = (
                    f"### Instruction:\n{example['instruction']}\n"
                    f"### Response:\n{example['response']}"
                )
            return {"text": text}

        dolly_formatted = dolly_raw.map(format_dolly)
        dolly_df = pd.DataFrame({
            "text":        dolly_formatted["text"],
            "instruction": dolly_formatted["instruction"],
            "context":     dolly_formatted["context"],
            "response":    dolly_formatted["response"],
            "category":    dolly_formatted["category"],
        })
        dolly_df.to_csv(dolly_path, index=False)
        print(f"Dolly saved: {len(dolly_df)} samples")
        print(dolly_df["category"].value_counts())

    print(f"Dolly loaded: {len(dolly_df)} samples")
    return Dataset.from_pandas(dolly_df[["text"]])


def load_alpaca(data_path, seed):
    alpaca_path = f"{data_path}/alpaca/alpaca_sampled.csv"
    if os.path.exists(alpaca_path):
        print(f"Loading Alpaca from cache: {alpaca_path}")
        alpaca_df = pd.read_csv(alpaca_path)
    else:
        print("Downloading Alpaca from HuggingFace...")
        os.makedirs(f"{data_path}/alpaca", exist_ok=True)
        alpaca_raw = load_dataset("tatsu-lab/alpaca", split="train")

        def format_alpaca(example):
            if example["input"] and example["input"].strip():
                text = (
                    f"### Instruction:\n{example['instruction']}\n"
                    f"### Input:\n{example['input']}\n"
                    f"### Response:\n{example['output']}"
                )
            else:
                text = (
                    f"### Instruction:\n{example['instruction']}\n"
                    f"### Response:\n{example['output']}"
                )
            return {"text": text}

        alpaca_formatted = alpaca_raw.map(format_alpaca)
        alpaca_df = pd.DataFrame({
            "text":        alpaca_formatted["text"],
            "instruction": alpaca_formatted["instruction"],
            "input":       alpaca_formatted["input"],
            "output":      alpaca_formatted["output"],
        })
        alpaca_df = alpaca_df.sample(n=min(11000, len(alpaca_df)), random_state=seed).reset_index(drop=True)
        alpaca_df.to_csv(alpaca_path, index=False)
        print(f"Alpaca saved: {len(alpaca_df)} samples")

    print(f"Alpaca loaded: {len(alpaca_df)} samples")
    return Dataset.from_pandas(alpaca_df[["text"]])


def load_oa(data_path, seed):
    oa_path = f"{data_path}/oa/oa_data.csv"
    if os.path.exists(oa_path):
        print(f"Loading OA from cache: {oa_path}")
        oa_df = pd.read_csv(oa_path)
    else:
        print("Downloading OpenAssistant from HuggingFace...")
        os.makedirs(f"{data_path}/oa", exist_ok=True)
        oa_raw = load_dataset("OpenAssistant/oasst1", split="train")
        oa_samples = []
        for example in oa_raw:
            if (example.get("lang") == "en" and
                    example.get("role") == "prompter" and
                    example.get("text")):
                text = (
                    f"### Instruction:\n{example['text']}\n"
                    f"### Response:\n"
                )
                oa_samples.append({
                    "text":       text,
                    "raw_text":   example["text"],
                    "message_id": example.get("message_id", ""),
                })
        oa_df = pd.DataFrame(oa_samples)
        oa_df = oa_df.sample(n=min(11000, len(oa_df)), random_state=seed).reset_index(drop=True)
        oa_df.to_csv(oa_path, index=False)
        print(f"OA saved: {len(oa_df)} samples")

    print(f"OA loaded: {len(oa_df)} samples")
    return Dataset.from_pandas(oa_df[["text"]])


DATASET_LOADERS = {
    "dolly":  load_dolly,
    "alpaca": load_alpaca,
    "oa":     load_oa,
}


def load_model_and_tokenizer(model_path):
    print(f"Loading LLaMA-7B from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("Model + LoRA ready.")
    return model, tokenizer


def train_one_epoch(model, tokenizer, dataset, ckpt_path, dataset_name, epoch):
    training_args = SFTConfig(
        output_dir=f"{ckpt_path}/{dataset_name}_lora",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=3e-4,
        warmup_steps=100,
        lr_scheduler_type="linear",
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=3,
        report_to="none",
        dataloader_num_workers=2,
        dataset_text_field="text",
        max_seq_length=2048,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
    )
    print(f"\nStarting {dataset_name} LoRA training — Epoch {epoch}...")
    trainer.train()
    print(f"Epoch {epoch} complete.")
    save_path = f"{ckpt_path}/{dataset_name}_lora_epoch{epoch}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Epoch {epoch} saved to: {save_path}")
    return model


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"\nCUDA available : {torch.cuda.is_available()}")
    print(f"GPU count      : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} | {mem:.1f} GB")

    os.makedirs(args.data_path, exist_ok=True)
    os.makedirs(args.ckpt_path, exist_ok=True)

    dataset = DATASET_LOADERS[args.dataset](args.data_path, args.seed)
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    for epoch in range(1, args.epochs + 1):
        model = train_one_epoch(model, tokenizer, dataset, args.ckpt_path, args.dataset, epoch)

    final_path = f"{args.ckpt_path}/{args.dataset}_lora_final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nFinal model saved to: {final_path}")
    print("Training complete.")


if __name__ == "__main__":
    main()
