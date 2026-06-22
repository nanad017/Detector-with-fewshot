#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${SD_FSPROTO_DATASET:-$HOME/An_solo/dataset}"
OUTPUT_ROOT="${SD_FSPROTO_OUTPUT:-$HOME/An_solo/detector/reproduction_output}"

python "$ROOT/reproduction/train_deepmd.py" --dataset-root "$DATASET_ROOT" --output-root "$OUTPUT_ROOT/deepmd"
python "$ROOT/reproduction/train_cnn.py" --dataset-root "$DATASET_ROOT" --output-root "$OUTPUT_ROOT/cnn"
python "$ROOT/reproduction/train_ember2024.py" --dataset-root "$DATASET_ROOT" --output-root "$OUTPUT_ROOT/ember2024" --task binary
python "$ROOT/reproduction/train_ember2024.py" --dataset-root "$DATASET_ROOT" --output-root "$OUTPUT_ROOT/ember2024" --task family
python "$ROOT/reproduction/train_sorel.py" --dataset-root "$DATASET_ROOT" --output-root "$OUTPUT_ROOT/sorel"

