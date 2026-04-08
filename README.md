# Distribution Debias
Repository for the paper "Controlling Distributional Bias in Multi-Round LLM Generation via KL-Optimized Fine-Tuning"

# Project Overview

This repository contains four methods:

* **Baseline (Zero-shot)**
* **IFT (Instruction Fine-Tuning)**
* **DPO (Direct Preference Optimization)**
* **Ours (GDPO-based method)**

Each method is organized into a separate folder. You can run the provided `.sh` scripts directly to reproduce the results.

---

# Usage Instructions

## 1. Baseline (Zero-shot), IFT, and DPO

For these three methods, the workflow is similar.

### Training

For example, to train the IFT model:

```bash
./qwen7b_IFT_train.sh
```

### Testing

* **Attribute Evaluation**

```bash
./qwen7b_IFT_test_v2.sh
```

* **Story Generation**

```bash
./qwen7b_IFT_test_v3.sh
```

> Note: Replace `IFT` with `baseline` or `DPO` depending on the method you want to run.

---

## 2. Ours (GDPO)

For our method, training and testing are integrated into a single script:

```bash
./train_scripts/run_gdpo_qwen_7b.sh
```

> ⚠️ **Important:**
> Make sure to modify the corresponding **root paths** in the script before running.

---

# Directory Structure

```
.
├── baseline/
├── IFT/
├── DPO/
├── ours/
└── train_scripts/
```

---

# Citation

If you find this work useful, please cite our paper:

```bibtex
@article{jiang2026controllingdistributionalbiasmultiround,
  title={Controlling Distributional Bias in Multi-Round LLM Generation via KL-Optimized Fine-Tuning},
  author={Yanbei Jiang and Amr Keleg and Ryandito Diandaru and Jey Han Lau and Lea Frermann and Biaoyan Fang and Fajri Koto},
  journal={arXiv preprint arXiv:2604.05756},
  year={2026}
}
```
