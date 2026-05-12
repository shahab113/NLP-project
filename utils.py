"""
src/utils.py — Utility Functions
==================================
Includes:
  - ECE (Expected Calibration Error) computation
  - MMLU prompt builder
  - MMLU answer extractor
  - Reliability diagram plotter
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


# ──────────────────────────────────────────────
# ECE Computation
# ──────────────────────────────────────────────

def compute_ece(confidences: List[float], labels: List[int],
                n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE) with equal-width bins.
    Following Guo et al. (2017).

    Args:
        confidences: list of predicted confidence scores in [0, 1]
        labels:      list of binary correctness labels (1=correct, 0=wrong)
        n_bins:      number of equal-width bins

    Returns:
        ECE as a float
    """
    confidences = np.array(confidences)
    labels = np.array(labels)
    n = len(confidences)
    if n == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include right edge in last bin
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        if mask.sum() == 0:
            continue

        bin_conf = confidences[mask].mean()
        bin_acc = labels[mask].mean()
        bin_size = mask.sum()

        ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_ece_bins(confidences: List[float], labels: List[int],
                     n_bins: int = 10):
    """
    Return bin-level statistics for plotting reliability diagrams.

    Returns:
        bin_centers, bin_accs, bin_confs, bin_sizes
    """
    confidences = np.array(confidences)
    labels = np.array(labels)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs, confs, sizes = [], [], [], []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        centers.append((lo + hi) / 2)
        accs.append(labels[mask].mean())
        confs.append(confidences[mask].mean())
        sizes.append(mask.sum())

    return np.array(centers), np.array(accs), np.array(confs), np.array(sizes)


# ──────────────────────────────────────────────
# MMLU Helpers
# ──────────────────────────────────────────────

MMLU_CHOICES = ["A", "B", "C", "D"]


def build_mmlu_prompt(sample: dict, few_shots: list) -> str:
    """
    Build a 5-shot MMLU prompt.

    Args:
        sample:    the test example dict (question, choices, answer)
        few_shots: list of few-shot example dicts from the val split

    Returns:
        Formatted prompt string
    """
    def fmt_example(ex, include_answer=True):
        q = ex["question"]
        choices = ex["choices"]
        lines = [f"Question: {q}"]
        for i, c in enumerate(choices):
            lines.append(f"{MMLU_CHOICES[i]}. {c}")
        if include_answer:
            lines.append(f"Answer: {MMLU_CHOICES[ex['answer']]}")
        else:
            lines.append("Answer:")
        return "\n".join(lines)

    parts = ["The following are multiple choice questions. Answer with a single letter.\n"]
    for shot in few_shots:
        parts.append(fmt_example(shot, include_answer=True))
        parts.append("")  # blank line
    parts.append(fmt_example(sample, include_answer=False))
    return "\n".join(parts)


def extract_mmlu_answer(text: str) -> Optional[str]:
    """
    Extract the first occurrence of A, B, C, or D from model output.

    Returns:
        One of 'A','B','C','D' or None if not found.
    """
    text = text.strip()
    match = re.search(r'\b([A-D])\b', text)
    if match:
        return match.group(1)
    # Try first character as fallback
    if text and text[0] in "ABCD":
        return text[0]
    return None


# ──────────────────────────────────────────────
# Reliability Diagram
# ──────────────────────────────────────────────

def plot_reliability_diagram(confidences: List[float], labels: List[int],
                             n_bins: int = 10, title: str = "Reliability Diagram",
                             save_path: Optional[str] = None):
    """
    Plot a reliability diagram (confidence vs accuracy per bin).

    Args:
        confidences: model confidence scores
        labels:      binary correctness labels
        n_bins:      number of bins
        title:       plot title
        save_path:   if given, save to this file path
    """
    centers, accs, confs, sizes = compute_ece_bins(confidences, labels, n_bins)
    ece = compute_ece(confidences, labels, n_bins)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1.5)
    bar_width = 1.0 / n_bins
    ax.bar(centers, accs, width=bar_width * 0.8, alpha=0.6,
           color="steelblue", label="Accuracy")
    ax.bar(centers, confs, width=bar_width * 0.8, alpha=0.3,
           color="orange", label="Confidence", bottom=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title}\nECE = {ece:.4f}")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[plot] Saved to {save_path}")
    plt.show()
    return fig


# ──────────────────────────────────────────────
# Misc
# ──────────────────────────────────────────────

def set_seed(seed: int = 42):
    import random, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
