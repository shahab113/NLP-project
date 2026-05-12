"""
src/dataset.py — Dataset Loaders
==================================
Loaders and formatters for all datasets used in this project:
  - Databricks Dolly 15k  (training)
  - Stanford Alpaca       (training baseline)
  - OpenAssistant (oasst1)(training baseline)
  - PILE                  (CLM evaluation)
  - T-REx                 (Facts evaluation)
  - MMLU                  (Reasoning evaluation)
"""

import random
from datasets import load_dataset

SYSTEM_PROMPT = "You are a helpful assistant."


# ──────────────────────────────────────────────
# Training Dataset Loaders
# ──────────────────────────────────────────────

def load_dolly(shuffle: bool = True, seed: int = 42):
    """
    Load Databricks Dolly 15k.
    Returns a HuggingFace Dataset with a 'text' column.
    """
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    dataset = dataset.map(_format_dolly, remove_columns=dataset.column_names)
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    print(f"[dataset] Dolly loaded: {len(dataset)} examples")
    return dataset


def load_alpaca(shuffle: bool = True, seed: int = 42):
    """Load Stanford Alpaca (52k synthetic instruction pairs)."""
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.map(_format_alpaca, remove_columns=dataset.column_names)
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    print(f"[dataset] Alpaca loaded: {len(dataset)} examples")
    return dataset


def load_openassistant(shuffle: bool = True, seed: int = 42):
    """
    Load OpenAssistant oasst1.
    Filters to single-turn English assistant responses only (~11k).
    """
    dataset = load_dataset("OpenAssistant/oasst1", split="train")
    # Keep only assistant messages in English
    dataset = dataset.filter(lambda x: x["role"] == "assistant" and x["lang"] == "en")
    dataset = dataset.map(_format_oa, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda x: x["text"] is not None)
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    print(f"[dataset] OpenAssistant loaded: {len(dataset)} examples")
    return dataset


# ──────────────────────────────────────────────
# Evaluation Dataset Loaders
# ──────────────────────────────────────────────

def load_pile_samples(num_samples: int = 5000, seed: int = 42):
    """Stream N samples from the PILE test set."""
    dataset = load_dataset("EleutherAI/pile", split="test", streaming=True)
    samples = list(dataset.take(num_samples * 2))
    random.seed(seed)
    random.shuffle(samples)
    return samples[:num_samples]


def load_trex_samples(num_samples: int = 5000, seed: int = 42):
    """Sample N examples from T-REx."""
    dataset = load_dataset("hadyelsahar/t-rex", split="train")
    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    return [dataset[i] for i in indices]


def load_mmlu_samples(num_samples: int = 3000, seed: int = 42):
    """Sample N examples from MMLU (all subjects combined)."""
    dataset = load_dataset("cais/mmlu", "all", split="test")
    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    return [dataset[i] for i in indices]


def load_mmlu_fewshot_pool(num_shots: int = 5, seed: int = 42):
    """Load the MMLU validation split for building few-shot prompts."""
    val = load_dataset("cais/mmlu", "all", split="validation")
    return list(val)


# ──────────────────────────────────────────────
# Private Formatters
# ──────────────────────────────────────────────

def _format_dolly(example: dict) -> dict:
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


def _format_alpaca(example: dict) -> dict:
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")
    if inp:
        text = (
            f"### System:\n{SYSTEM_PROMPT}\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n{output}"
        )
    else:
        text = (
            f"### System:\n{SYSTEM_PROMPT}\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output}"
        )
    return {"text": text}


def _format_oa(example: dict) -> dict:
    body = example.get("text", "")
    if not body:
        return {"text": None}
    # Use parent text as instruction if available; fall back to placeholder
    instruction = example.get("parent_id", "Please respond to the following.")
    text = (
        f"### System:\n{SYSTEM_PROMPT}\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{body}"
    )
    return {"text": text}
