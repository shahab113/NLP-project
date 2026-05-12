"""
src/model.py — Model & LoRA Loader
====================================
Utilities for loading LLaMA-7B with optional LoRA adapters.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel, TaskType


def load_base_model(model_name: str, dtype=torch.float16):
    """Load a base causal LM and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.config.use_cache = False
    return model, tokenizer


def attach_lora(model, lora_cfg: dict):
    """
    Attach a LoRA adapter to an existing model.

    Args:
        model: a HuggingFace CausalLM
        lora_cfg: dict with keys r, lora_alpha, lora_dropout,
                  target_modules, bias
    Returns:
        PEFT-wrapped model
    """
    config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def load_lora_model(base_model_name: str, adapter_path: str, dtype=torch.float16):
    """
    Load a base model and apply a saved LoRA adapter.

    Args:
        base_model_name: HuggingFace model id or local path for base model
        adapter_path: path to directory containing adapter_config.json
        dtype: torch dtype for loading
    Returns:
        (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer


def count_parameters(model) -> dict:
    """Return trainable and total parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "trainable_pct": round(100 * trainable / total, 4),
    }
