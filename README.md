# On the Calibration of Large Language Models and Alignment

**Course:** Natural Language Processing (NLP) — MS Artificial Intelligence  
**Batch:** Fall 2025 | **Institution:** FAST NUCES, Islamabad  
**Authors:** Shahab Ahmad (25I-7626), Inam Ul Hassan (25I-7617), Ejaz Ulhaq (25I-7624)  
**Supervisor:** Dr. Zohair Ahmed

---

## Overview

This repository contains the full reproduction and extension of:

> Zhu et al. (2023). *On the Calibration of Large Language Models and Alignment.* EMNLP 2023 Findings.

We reproduce the paper's core experiments (pretraining & instruction-tuning calibration), then propose and evaluate **Databricks Dolly 15k** as a superior instruction-tuning dataset for model calibration.

### Key Finding
> Fine-tuning LLaMA-7B with LoRA on the Dolly dataset achieves **lower ECE than both Alpaca and OpenAssistant** across all three evaluation tasks — and even surpasses the untuned baseline on MMLU calibration (ECE: 0.0237 vs 0.0497 baseline).

---

## Repository Structure

```
project-root/
├── README.md
├── requirements.txt
├── train.py                  # LoRA fine-tuning script
├── inference.py              # Evaluation & ECE computation
├── config.yaml               # All hyperparameters
│
├── data/
│   └── sample_data.csv       # 8 sample Dolly-format examples
│
├── notebooks/
│   └── 01_inference_demo.ipynb
│
├── src/
│   ├── model.py              # LoRA model loader
│   ├── dataset.py            # Dataset loaders (Dolly, PILE, T-REx, MMLU)
│   └── utils.py              # ECE computation & helpers
│
├── results/
│   ├── baseline_metrics.json
│   ├── improved_metrics.json
│   └── training_log.csv
│
└── checkpoints/
    └── README.md             # Instructions to download weights
```

---

## Experimental Setup

| Component | Details |
|-----------|---------|
| Base Model | LLaMA-7B |
| Fine-tuning Method | LoRA (rank=8, alpha=32, dropout=0.1) |
| Target Modules | `q_proj`, `v_proj` |
| Training Dataset | Databricks Dolly 15k |
| Hardware | Kaggle GPU-T4 × 2 |
| Epochs | 3 |
| Batch Size | 128 (4 × 16 × 2 grad accum) |
| Learning Rate | 3e-4 (linear scheduler) |
| Eval Metric | Expected Calibration Error (ECE, 10 bins) |

---

## Results Summary

| Model | ECE (CLM) | ECE (Facts) | ECE (MMLU) | ACC (MMLU) |
|-------|-----------|-------------|------------|------------|
| Baseline (LLaMA-7B) | 0.0092 | 0.0602 | 0.0497 | 0.4460 |
| Alpaca LoRA (3 ep) | 0.0856 | 0.2323 | — (bug) | — |
| OA LoRA (3 ep) | 0.0262 | 0.0813 | 0.0705 | 0.4310 |
| **Dolly LoRA (3 ep)** | **0.0201** | **0.0849** | **0.0237** | 0.2263 |

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Fine-tune on Dolly
```bash
python train.py --config config.yaml
```

### 3. Evaluate
```bash
python inference.py --model_path ./checkpoints/dolly_lora --task all
```

---

## Datasets Used

| Task | Dataset | Samples |
|------|---------|---------|
| Causal LM | [PILE](https://pile.eleuther.ai/) | 5,000 |
| Factual Generation | [T-REx](https://hadyelsahar.github.io/t-rex/) | 5,000 |
| Reasoning | [MMLU](https://github.com/hendrycks/test) | 3,000 (5-shot) |
| Training | [Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | 15,011 |

---

## References

1. Touvron et al. (2023). LLaMA: Open and Efficient Foundation Language Models. arXiv:2302.13971.
2. Zhu et al. (2023). On the Calibration of Large Language Models and Alignment. EMNLP 2023.
3. Biderman et al. (2023). Pythia: A Suite for Analyzing Large Language Models. arXiv:2304.01373.
4. Gao et al. (2020). The Pile: An 800GB Dataset of Diverse Text. arXiv:2101.00027.
5. Hendrycks et al. (2021). Measuring Massive Multitask Language Understanding. arXiv:2009.03300.
