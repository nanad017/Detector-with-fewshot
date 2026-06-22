# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

**SD-FSProto** — Static-Dynamic Few-Shot Prototypical Learning for PE Malware Family Classification. A research project to classify malware into families when only 1–5 labeled samples are available per novel family, by learning a joint embedding space from static PE features and dynamic sandbox behavior.

All documentation is in Vietnamese. The project is in **research design phase** — only the sub-projects have existing code; the main SD-FSProto system has not been implemented yet.

## Repository structure

```
solo/
├── deep-malware-detection/   # PyTorch MalConv/MalConv+/RCNN for PE binary classification
│   ├── src/deep_malware_detection/
│   │   ├── models.py         # MalConvBase, MalConvPlus, RCNN, AttentionRCNN
│   │   ├── dataset.py        # MalwareDataset, collate/pad utils
│   │   ├── train.py          # Training CLI (argparse)
│   │   └── utils.py          # Training loop, metrics, early stopping
│   ├── src/bin/              # PE scrapers + extract_header.py (pefile → pickle)
│   └── requirements.txt
├── malware-classification-CNN/  # CNN on Malimg grayscale images (25 malware families + benign)
├── SOREL-20M/                   # Sophos-ReversingLabs 20M benchmark (FFNN + LightGBM baselines)
│   ├── nets.py               # PENetwork (multi-head FFNN: malware + count + tags)
│   ├── train.py              # Training with auxiliary losses
│   ├── dataset.py / generators.py
│   └── evaluate.py / plot.py
└── ember2024/                   # EMBER2024: thrember feature extraction + LightGBM classifiers
    └── src/thrember/          # pip-installable package (Polars, pefile, LightGBM)
```

## Sub-projects as learning foundations

These four repos are existing codebases to study before building SD-FSProto:

| Sub-project | What it teaches | Key techniques |
|---|---|---|
| `deep-malware-detection` | Raw byte PE classification | MalConv, 1D-CNN, GRU/LSTM, attention, residual connections |
| `malware-classification-CNN` | Malware-as-image CNN | Grayscale conversion, ResNet-style CNN for 25 families |
| `SOREL-20M` | Large-scale PE benchmark | Multi-task FFNN (malware + tags + count), LightGBM, LMDB data loading |
| `ember2024` | Feature engineering at scale | EMBER v3 feature vectors, pefile-based extraction, LightGBM baselines |

## Core architecture (5 modules — not yet implemented)

1. **Static Encoder** — Multi-branch: PE grayscale image (CNN/ResNet) + PE metadata (MLP) + optionally import/API strings (Transformer/BiLSTM) → fused to a 512-d vector `z_s`
2. **Dynamic Encoder** — API call sequence (Transformer/BiLSTM) + optionally behavior graph (GNN/GAT) → fused to a 512-d vector `z_d`
3. **Reliability-Aware Fusion** — `z = α_s·z_s + α_d·z_d + α_sd·z_sd` where reliability scores (`α_s`, `α_d`) come from small MLPs trained on PE/sandbox metadata, and `α_sd` from cosine similarity between the two views.
4. **Prototypical Network** — Compute a prototype `p_c` for each family as the mean embedding of its support samples. Classify query samples by Euclidean distance to prototypes. Multi-prototype extension for family variants.
5. **Episodic Meta-Learning** — Train with N-way K-shot episodes that mirror the test scenario. Family-disjoint split is mandatory. Loss: `L = L_cls + λ1·L_con + λ2·L_align + λ3·L_sep`

## Design constraints

- **Family-disjoint split is mandatory** — leaking families across train/test invalidates few-shot results
- **No fixed classifier** — the system must classify novel families from support samples without retraining
- **Reliability gating matters** — packed PE files and sandbox-evading malware produce unreliable views; the fusion module must detect and down-weight them
- **Unknown detection** — samples too far from all prototypes should be flagged as unknown rather than forced into a known family

## Key references

- [ý tưởng.md](ý%20tưởng.md) — Full research proposal (problem, architecture, 14-step pipeline, baselines, novel claims)
- [Kiến thức cốt lõi.md](Kiến%20thức%20cốt%20lõi.md) — 6-track learning roadmap with pseudocode for every module
- [giải thích công thức.md](giải%20thích%20công%20thức.md) — Walkthrough of fusion formula, prototype computation, multi-loss training (with numerical examples)
- [Tài liệu Mảng 3 - Encoder.txt](Tài%20liệu%20Mảng%203%20-%20Encoder.txt) — 8-week self-study plan for MLP, CNN/ResNet, LSTM/Transformer, GNN

## Common commands

### deep-malware-detection

```bash
# Train (from deep-malware-detection/src/deep_malware_detection/)
python train.py --benign_dir=PATH_TO_BENIGN --malware_dir=PATH_TO_MALWARE

# Extract PE headers as pickle features
python -m src.bin.extract_header --input_dir=RAW_PE_DIR --output_dir=OUTPUT_DIR

# Download malware samples for research
python -m src.bin.dasmalwerk
python -m src.bin.malshare

# Code style (from deep-malware-detection/)
make style    # auto-format with black + isort
make quality  # check formatting
```

### ember2024

```bash
# Install thrember package (from ember2024/)
pip install .

# Download models
python -c "import thrember; thrember.download_models('/path/to/models')"

# Download dataset
python -c "import thrember; thrember.download_dataset('/path/to/data')"

# Vectorize + train/evaluate
python examples/train_lgbm.py
python examples/eval_lgbm.py
```

### SOREL-20M

```bash
# Create conda environment (from SOREL-20M/)
conda env create -f environment.yml
conda activate sorel

# Edit config.py to set device + data paths, then:
python train.py train_network
python evaluate.py evaluate_network /results/dir /checkpoints/epoch_9.pt
python plot.py plot_roc_distribution_for_tag /results/ffnn_results.json output.png
```

## Agent skills

### Issue tracker

GitHub Issues on `nanad017/Detector-with-fewshot` is the source of truth for planning and task tracking. Skills use the `gh` CLI to create, read, comment on, and close issues. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical triage roles use their default label names (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context repo. `CONTEXT.md` at the root holds domain knowledge (glossary, entities, invariants); `docs/adr/` holds architectural decisions with rationale. See `docs/agents/domain.md`.

## Planned tech stack

- **Python** with **PyTorch** for all deep learning components
- **pefile** / **LIEF** for static PE feature extraction
- **Cuckoo** or **CAPE** for dynamic behavior traces
- **PyTorch Geometric** for behavior graph encoding (GCN/GAT)
