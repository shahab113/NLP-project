"""
inference.py — Evaluation & ECE Computation
=============================================
Evaluates a (LoRA-tuned or base) LLaMA-7B model on three tasks:
  - CLM  : Causal Language Modeling (PILE)
  - Facts: Factual generation (T-REx)
  - MMLU : Multi-task reasoning (MMLU, 5-shot)

Usage:
    python inference.py --model_path ./checkpoints/dolly_lora/final --task all
    python inference.py --model_path huggyllama/llama-7b --task clm
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import compute_ece, extract_mmlu_answer, build_mmlu_prompt


# ──────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to base model or LoRA adapter directory")
    parser.add_argument("--base_model", type=str, default="huggyllama/llama-7b",
                        help="Base model (needed if model_path is a LoRA adapter)")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--task", type=str, default="all",
                        choices=["all", "clm", "facts", "mmlu"])
    parser.add_argument("--output", type=str, default="results/eval_results.json")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────
def load_model(args):
    print(f"[INFO] Loading tokenizer from: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # Check if model_path is a LoRA adapter or base model
    is_lora = os.path.exists(os.path.join(args.model_path, "adapter_config.json"))

    if is_lora:
        print(f"[INFO] Loading base model: {args.base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.float16, device_map="auto"
        )
        print(f"[INFO] Applying LoRA adapter from: {args.model_path}")
        model = PeftModel.from_pretrained(model, args.model_path)
    else:
        print(f"[INFO] Loading model directly: {args.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.float16, device_map="auto"
        )

    model.eval()
    return model, tokenizer


# ──────────────────────────────────────────────
# CLM Evaluation (PILE)
# ──────────────────────────────────────────────
def eval_clm(model, tokenizer, cfg, seed):
    print("\n[TASK] Causal Language Modeling (PILE)")
    num_samples = cfg["evaluation"]["clm"]["num_samples"]
    dataset = load_dataset("EleutherAI/pile", split="test", streaming=True)

    random.seed(seed)
    samples = list(dataset.take(num_samples * 3))
    random.shuffle(samples)
    samples = samples[:num_samples]

    correct, confidences, labels = [], [], []

    for sample in tqdm(samples, desc="CLM"):
        text = sample.get("text", "")
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = tokens["input_ids"]
        if input_ids.shape[1] < 3:
            continue

        # Pick a random position
        pos = random.randint(1, input_ids.shape[1] - 2)
        context = input_ids[:, :pos].to(model.device)
        target_id = input_ids[0, pos].item()

        with torch.no_grad():
            logits = model(context).logits[0, -1]
            probs = torch.softmax(logits, dim=-1)
            pred_id = torch.argmax(probs).item()
            confidence = probs[pred_id].item()

        is_correct = int(pred_id == target_id)
        correct.append(is_correct)
        confidences.append(confidence)
        labels.append(is_correct)

    acc = np.mean(correct)
    ece = compute_ece(confidences, labels, n_bins=cfg["ece"]["num_bins"])
    print(f"  CLM  → ACC: {acc:.4f} | ECE: {ece:.4f}")
    return {"acc": round(acc, 4), "ece": round(ece, 4)}


# ──────────────────────────────────────────────
# Facts Evaluation (T-REx)
# ──────────────────────────────────────────────
def eval_facts(model, tokenizer, cfg, seed):
    print("\n[TASK] Factual Generation (T-REx)")
    num_samples = cfg["evaluation"]["facts"]["num_samples"]
    dataset = load_dataset("hadyelsahar/t-rex", split="train")

    random.seed(seed)
    indices = random.sample(range(len(dataset)), num_samples)

    correct, confidences, labels = [], [], []

    for idx in tqdm(indices, desc="Facts"):
        sample = dataset[idx]
        # T-REx has sentences with entity spans; use the masked sentence as context
        sentences = sample.get("sentences", [])
        if not sentences:
            continue
        sent = sentences[0]
        text = sent.get("text", "")
        entities = sent.get("entities", [])
        if not entities or not text:
            continue

        entity = entities[0]
        start = entity.get("boundaries", [0, 0])[0]
        # Context is everything before the entity
        context_text = text[:start].strip()
        if not context_text:
            continue

        target_text = text[start:entity.get("boundaries", [0, 1])[1]]
        target_tokens = tokenizer(target_text, add_special_tokens=False)["input_ids"]
        if not target_tokens:
            continue
        target_id = target_tokens[0]

        input_ids = tokenizer(context_text, return_tensors="pt",
                              truncation=True, max_length=512)["input_ids"].to(model.device)

        with torch.no_grad():
            logits = model(input_ids).logits[0, -1]
            probs = torch.softmax(logits, dim=-1)
            pred_id = torch.argmax(probs).item()
            confidence = probs[pred_id].item()

        is_correct = int(pred_id == target_id)
        correct.append(is_correct)
        confidences.append(confidence)
        labels.append(is_correct)

    acc = np.mean(correct) if correct else 0.0
    ece = compute_ece(confidences, labels, n_bins=cfg["ece"]["num_bins"])
    print(f"  Facts → ACC: {acc:.4f} | ECE: {ece:.4f}")
    return {"acc": round(acc, 4), "ece": round(ece, 4)}


# ──────────────────────────────────────────────
# MMLU Evaluation
# ──────────────────────────────────────────────
def eval_mmlu(model, tokenizer, cfg, seed):
    print("\n[TASK] MMLU Reasoning (5-shot)")
    num_samples = cfg["evaluation"]["mmlu"]["num_samples"]
    num_shots = cfg["evaluation"]["mmlu"]["num_shots"]
    dataset = load_dataset("cais/mmlu", "all", split="test")

    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    # Gather few-shot examples from validation split
    val_dataset = load_dataset("cais/mmlu", "all", split="validation")
    few_shot_pool = list(val_dataset)

    correct, confidences, labels = [], [], []
    answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    for idx in tqdm(indices, desc="MMLU"):
        sample = dataset[idx]
        few_shots = random.sample(few_shot_pool, num_shots)
        prompt = build_mmlu_prompt(sample, few_shots)

        inputs = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=2048).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                temperature=1.0,
            )
        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()
        pred_letter = extract_mmlu_answer(generated)

        # Also get confidence via logits over A/B/C/D
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]
        choice_ids = [tokenizer(f" {c}", add_special_tokens=False)["input_ids"][0]
                      for c in ["A", "B", "C", "D"]]
        choice_logits = torch.stack([logits[cid] for cid in choice_ids])
        choice_probs = torch.softmax(choice_logits, dim=-1).cpu().numpy()

        correct_idx = sample["answer"]
        correct_letter = answer_map[correct_idx]

        if pred_letter is None:
            continue

        pred_idx = ["A", "B", "C", "D"].index(pred_letter) if pred_letter in "ABCD" else -1
        confidence = float(choice_probs[pred_idx]) if pred_idx >= 0 else 0.25
        is_correct = int(pred_letter == correct_letter)

        correct.append(is_correct)
        confidences.append(confidence)
        labels.append(is_correct)

    if not correct:
        print("  MMLU → No valid answers extracted (check model format).")
        return {"acc": None, "ece": None}

    acc = np.mean(correct)
    ece = compute_ece(confidences, labels, n_bins=cfg["ece"]["num_bins"])
    print(f"  MMLU  → ACC: {acc:.4f} | ECE: {ece:.4f}")
    return {"acc": round(acc, 4), "ece": round(ece, 4)}


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    args = parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model, tokenizer = load_model(args)
    results = {}

    if args.task in ("all", "clm"):
        results["clm"] = eval_clm(model, tokenizer, cfg, args.seed)

    if args.task in ("all", "facts"):
        results["facts"] = eval_facts(model, tokenizer, cfg, args.seed)

    if args.task in ("all", "mmlu"):
        results["mmlu"] = eval_mmlu(model, tokenizer, cfg, args.seed)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Results saved to {args.output}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
