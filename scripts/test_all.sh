#!/usr/bin/env bash
# =============================================================================
# test_all.sh — Evaluate all 4 trained malware detection models
#
# Usage:
#   chmod +x scripts/test_all.sh
#   bash scripts/test_all.sh
#
# Prerequisites:
#   - All models trained via train_all.sh
#   - Processed data at ~/An_solo/detector/processed_data/
# =============================================================================

set -euo pipefail

# --- Configuration ----------------------------------------------------------
DETECTOR_ROOT="$HOME/An_solo/detector"
VENV_DIR="$DETECTOR_ROOT/venv"
PROCESSED_DIR="$DETECTOR_ROOT/processed_data"
MODELS_DIR="$DETECTOR_ROOT/models"
SCRIPTS_DIR="$DETECTOR_ROOT/scripts"
DEEPMD_SRC="$DETECTOR_ROOT/deep-malware-detection/src/deep_malware_detection"

# --- Colors -----------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_step()  { echo -e "\n${CYAN}========================================${NC}"; echo -e "${CYAN}[EVAL] $*${NC}"; echo -e "${CYAN}========================================${NC}"; }
log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# --- Preflight --------------------------------------------------------------
preflight() {
    log_step "Preflight"

    if [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"
        log_info "Activated virtual env"
    fi
}

# --- Evaluate CNN -----------------------------------------------------------
eval_cnn() {
    log_step "Evaluating CNN"

    if [ ! -f "$MODELS_DIR/cnn/best_model.pt" ]; then
        log_error "CNN model not found at $MODELS_DIR/cnn/best_model.pt"
        return 1
    fi

    python "$SCRIPTS_DIR/eval_cnn.py" \
        --model_path "$MODELS_DIR/cnn/best_model.pt" \
        --test_dir "$PROCESSED_DIR/png/test" \
        --output_dir "$MODELS_DIR/cnn" \
        --image_size 256 \
        --batch_size 32

    log_info "CNN evaluation done -> $MODELS_DIR/cnn/results.json"
}

# --- Evaluate deep-malware-detection -----------------------------------------
eval_deepmd() {
    log_step "Evaluating deep-malware-detection (MalConv+)"

    if [ ! -f "$MODELS_DIR/deepmd/best_model.pt" ]; then
        log_error "deep-md model not found at $MODELS_DIR/deepmd/best_model.pt"
        return 1
    fi

    # Quick evaluation using Python inline
    python -c "
import torch, json, os, sys
sys.path.insert(0, '$DEEPMD_SRC/../..')
sys.path.insert(0, '$DEEPMD_SRC')
from dataset import MalwareDataset, make_loader
from utils import get_accuracy, predict
from models import MalConvPlus
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

device = torch.device('cpu')

test_dataset = MalwareDataset(
    '$PROCESSED_DIR/pickle/test/benign',
    '$PROCESSED_DIR/pickle/test/malware'
)
test_loader = make_loader(test_dataset, batch_size=16, shuffle=False)

model = MalConvPlus(8, 4096, 128, 32, 0.5).to(device)
model.load_state_dict(torch.load('$MODELS_DIR/deepmd/best_model.pt', map_location=device, weights_only=True))

acc = get_accuracy(model, test_loader, device)
y_true, y_probs = predict(model, test_loader, device, apply_sigmoid=True)
y_pred = (y_probs > 0.5).astype(int)

results = {
    'accuracy': float(acc),
    'auc': float(roc_auc_score(y_true, y_probs)),
    'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
}

os.makedirs('$MODELS_DIR/deepmd', exist_ok=True)
with open('$MODELS_DIR/deepmd/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f\"Deep-MD — Acc: {acc:.2f}%, AUC: {results['auc']:.4f}, F1: {results['f1']:.4f}\")
"
    log_info "deep-malware-detection evaluation done -> $MODELS_DIR/deepmd/results.json"
}

# --- Evaluate SOREL-FFNN ----------------------------------------------------
eval_sorel_ffnn() {
    log_step "Evaluating SOREL-FFNN"

    if [ ! -f "$MODELS_DIR/sorel_ffnn/best_model.pt" ]; then
        log_error "SOREL-FFNN model not found at $MODELS_DIR/sorel_ffnn/best_model.pt"
        return 1
    fi

    python "$SCRIPTS_DIR/eval_sorel_ffnn.py" \
        --model_path "$MODELS_DIR/sorel_ffnn/best_model.pt" \
        --test_features "$PROCESSED_DIR/ember_features/X_test.npy" \
        --test_labels "$PROCESSED_DIR/ember_features/y_test_binary.npy" \
        --output_dir "$MODELS_DIR/sorel_ffnn"

    log_info "SOREL-FFNN evaluation done -> $MODELS_DIR/sorel_ffnn/results.json"
}

# --- Evaluate ember2024 LightGBM ---------------------------------------------
eval_ember_lgbm() {
    log_step "Evaluating ember2024 LightGBM"

    # Binary model
    if [ -f "$MODELS_DIR/ember_lgbm/binary_model.model" ]; then
        python "$SCRIPTS_DIR/eval_ember_lgbm.py" \
            --model_path "$MODELS_DIR/ember_lgbm/binary_model.model" \
            --test_features "$PROCESSED_DIR/ember_features/X_test.npy" \
            --test_labels "$PROCESSED_DIR/ember_features/y_test_binary.npy" \
            --output_dir "$MODELS_DIR/ember_lgbm" \
            --model_name binary_model
    else
        log_warn "Binary LGBM model not found, skipping."
    fi

    # Family model
    if [ -f "$MODELS_DIR/ember_lgbm/family_model.model" ] && \
       [ -f "$PROCESSED_DIR/ember_features/y_test_family.npy" ]; then
        python "$SCRIPTS_DIR/eval_ember_lgbm.py" \
            --model_path "$MODELS_DIR/ember_lgbm/family_model.model" \
            --test_features "$PROCESSED_DIR/ember_features/X_test.npy" \
            --test_labels "$PROCESSED_DIR/ember_features/y_test_family.npy" \
            --output_dir "$MODELS_DIR/ember_lgbm" \
            --model_name family_model
    else
        log_warn "Family LGBM model or labels not found, skipping."
    fi

    log_info "ember2024 LGBM evaluation done."
}

# --- Summary -----------------------------------------------------------------
print_summary() {
    log_step "Summary of All Results"

    echo ""
    printf "%-30s %10s %10s %10s\n" "Model" "Accuracy" "AUC" "F1"
    printf "%-30s %10s %10s %10s\n" "------------------------------" "----------" "----------" "----------"

    for model_dir in "$MODELS_DIR"/{cnn,deepmd,sorel_ffnn,ember_lgbm}; do
        local name=$(basename "$model_dir")
        local results_file=""

        # Find the results file (different names per project)
        if [ -f "$model_dir/results.json" ]; then
            results_file="$model_dir/results.json"
            local acc=$(python -c "import json; d=json.load(open('$results_file')); print(f\"{d.get('accuracy',-1):.4f}\")" 2>/dev/null || echo "N/A")
            local auc=$(python -c "import json; d=json.load(open('$results_file')); print(f\"{d.get('auc',-1):.4f}\")" 2>/dev/null || echo "N/A")
            local f1=$(python -c "import json; d=json.load(open('$results_file')); print(f\"{d.get('f1',d.get('f1_macro',-1)):.4f}\")" 2>/dev/null || echo "N/A")
            printf "%-30s %10s %10s %10s\n" "$name" "$acc" "$auc" "$f1"
        fi

        # Also check for named results
        for rf in "$model_dir"/*_results.json; do
            [ -f "$rf" ] || continue
            local subname="$name/$(basename "$rf" _results.json)"
            local acc=$(python -c "import json; d=json.load(open('$rf')); print(f\"{d.get('accuracy',-1):.4f}\")" 2>/dev/null || echo "N/A")
            local auc=$(python -c "import json; d=json.load(open('$rf')); print(f\"{d.get('auc',-1):.4f}\")" 2>/dev/null || echo "N/A")
            local f1=$(python -c "import json; d=json.load(open('$rf')); print(f\"{d.get('f1',d.get('f1_macro',-1)):.4f}\")" 2>/dev/null || echo "N/A")
            printf "%-30s %10s %10s %10s\n" "$subname" "$acc" "$auc" "$f1"
        done
    done

    echo ""
}

# --- Main -------------------------------------------------------------------
main() {
    echo -e "${CYAN}"
    echo "=============================================="
    echo "  TEST ALL — 4 Malware Detection Models"
    echo "=============================================="
    echo -e "${NC}"

    preflight
    eval_cnn
    eval_deepmd
    eval_sorel_ffnn
    eval_ember_lgbm
    print_summary

    echo -e "\n${GREEN}All evaluations complete.${NC}"
    echo "Results saved under: $MODELS_DIR/"
}

main "$@"
