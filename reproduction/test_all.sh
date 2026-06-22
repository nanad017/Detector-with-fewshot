#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${SD_FSPROTO_OUTPUT:-$HOME/An_solo/detector/reproduction_output}"

python "$ROOT/reproduction/test_deepmd.py" --output-root "$OUTPUT_ROOT/deepmd"
python "$ROOT/reproduction/test_cnn.py" --output-root "$OUTPUT_ROOT/cnn"
python "$ROOT/reproduction/test_ember2024.py" --output-root "$OUTPUT_ROOT/ember2024" --task binary
python "$ROOT/reproduction/test_ember2024.py" --output-root "$OUTPUT_ROOT/ember2024" --task family
python "$ROOT/reproduction/test_sorel.py" --output-root "$OUTPUT_ROOT/sorel"

