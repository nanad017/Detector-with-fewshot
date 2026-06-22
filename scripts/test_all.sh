#!/usr/bin/env bash
# =============================================================================
# test_all.sh — Evaluate all trained models
# =============================================================================

set -euo pipefail

DETECTOR_ROOT="$HOME/An_solo/detector"
VENV_DIR="$DETECTOR_ROOT/venv"
PROCESSED_DIR="$DETECTOR_ROOT/processed_data"
MODELS_DIR="$DETECTOR_ROOT/models"
SCRIPTS_DIR="$DETECTOR_ROOT/scripts"
DEEPMD_SRC="$DETECTOR_ROOT/deep-malware-detection/src/deep_malware_detection"

RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'
log_step()  { echo -e "\n${GREEN}[EVAL] $*${NC}"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

if [ -f "$VENV_DIR/bin/activate" ]; then source "$VENV_DIR/bin/activate"; fi

# --- Evaluate CNN -----------------------------------------------------------
eval_cnn() {
    log_step "Evaluating CNN"
    if [ ! -f "$MODELS_DIR/cnn/best_model.pt" ]; then log_error "Model not found"; return 1; fi
    python "$SCRIPTS_DIR/eval_cnn.py" \
        --model_path "$MODELS_DIR/cnn/best_model.pt" \
        --test_dir "$PROCESSED_DIR/png/test" \
        --output_dir "$MODELS_DIR/cnn" --image_size 256 --batch_size 32
}

# --- Evaluate deep-malware-detection (all variants) -------------------------
eval_deepmd() {
    log_step "Evaluating deep-malware-detection"
    for pt_file in "$MODELS_DIR/deepmd"/*.pt; do
        [ -f "$pt_file" ] || continue
        local name=$(basename "$pt_file" .pt)
        log_step "  $name"
        python -c "
import torch, json, os, sys
sys.path.insert(0, '$DEEPMD_SRC/../..'); sys.path.insert(0, '$DEEPMD_SRC')
from dataset import MalwareDataset, make_loader
from utils import get_accuracy, predict
from models import MalConvBase, MalConvPlus, RCNN, AttentionRCNN
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import torch.nn as nn

device = torch.device('cpu')
test_dataset = MalwareDataset('$PROCESSED_DIR/pickle/test/benign', '$PROCESSED_DIR/pickle/test/malware')
test_loader = make_loader(test_dataset, batch_size=16, shuffle=False)

# Try each model class
for model_cls in [MalConvBase, MalConvPlus, RCNN, AttentionRCNN]:
    try:
        if 'MalConvBase' in '$name':
            model = MalConvBase(8, 4096, 128, 32)
        elif 'MalConvPlus' in '$name':
            model = MalConvPlus(8, 4096, 128, 32)
        else:
            continue
        break
    except: pass

model.load_state_dict(torch.load('$pt_file', map_location=device, weights_only=True))
acc = get_accuracy(model, test_loader, device)
y_t, y_p = predict(model, test_loader, device, apply_sigmoid=True)
y_pred = (y_p > 0.5).astype(int)
results = {'accuracy': float(acc), 'auc': float(roc_auc_score(y_t, y_p)), 'f1': float(f1_score(y_t, y_pred))}
with open('$MODELS_DIR/deepmd/${name}_results.json','w') as f: json.dump(results,f,indent=2)
print(f'Acc: {acc:.2f}%, AUC: {results[\"auc\"]:.4f}, F1: {results[\"f1\"]:.4f}')
" 2>/dev/null || echo "  SKIP (model not supported by eval script)"
    done
}

# --- Evaluate SOREL-FFNN ----------------------------------------------------
eval_sorel_ffnn() {
    log_step "Evaluating SOREL-FFNN"
    if [ ! -f "$MODELS_DIR/sorel_ffnn/best_model.pt" ]; then log_error "Model not found"; return 1; fi
    python "$SCRIPTS_DIR/eval_sorel_ffnn.py" \
        --model_path "$MODELS_DIR/sorel_ffnn/best_model.pt" \
        --test_features "$PROCESSED_DIR/ember_features/X_test.npy" \
        --test_labels "$PROCESSED_DIR/ember_features/y_test_binary.npy" \
        --output_dir "$MODELS_DIR/sorel_ffnn"
}

# --- Evaluate ember2024 LightGBM ---------------------------------------------
eval_ember_lgbm() {
    log_step "Evaluating ember2024 LightGBM"
    for model_file in "$MODELS_DIR/ember_lgbm"/*.model; do
        [ -f "$model_file" ] || continue
        local name=$(basename "$model_file" .model)
        if [[ "$name" == "binary"* ]]; then
            python "$SCRIPTS_DIR/eval_ember_lgbm.py" \
                --model_path "$model_file" --model_name "$name" \
                --test_features "$PROCESSED_DIR/ember_features/X_test.npy" \
                --test_labels "$PROCESSED_DIR/ember_features/y_test_binary.npy" \
                --output_dir "$MODELS_DIR/ember_lgbm"
        else
            python "$SCRIPTS_DIR/eval_ember_lgbm.py" \
                --model_path "$model_file" --model_name "$name" \
                --test_features "$PROCESSED_DIR/ember_features/X_test.npy" \
                --test_labels "$PROCESSED_DIR/ember_features/y_test_family.npy" \
                --output_dir "$MODELS_DIR/ember_lgbm"
        fi
    done
}

# --- Main -------------------------------------------------------------------
main() {
    eval_cnn
    eval_deepmd
    eval_sorel_ffnn
    eval_ember_lgbm
    echo -e "\n${GREEN}All evaluations complete.${NC}"
}

main "$@"
