#!/usr/bin/env bash
# =============================================================================
# train_all.sh — Train all 4 malware detection models on custom dataset
#
# Usage:
#   chmod +x scripts/train_all.sh
#   bash scripts/train_all.sh
#
# Prerequisites:
#   - Dataset at ~/An_solo/dataset/train/ and ~/An_solo/dataset/test/
#   - Virtual env created at ~/An_solo/detector/venv/
#   - Repos cloned at ~/An_solo/detector/
# =============================================================================

set -euo pipefail

# --- Configuration ----------------------------------------------------------
DETECTOR_ROOT="$HOME/An_solo/detector"
DATASET_ROOT="$HOME/An_solo/dataset"
VENV_DIR="$DETECTOR_ROOT/venv"
PROCESSED_DIR="$DETECTOR_ROOT/processed_data"
MODELS_DIR="$DETECTOR_ROOT/models"
SCRIPTS_DIR="$DETECTOR_ROOT/scripts"
DEEPMD_SRC="$DETECTOR_ROOT/deep-malware-detection/src/deep_malware_detection"

# --- Colors for output ------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_step()  { echo -e "\n${CYAN}========================================${NC}"; echo -e "${CYAN}[STEP] $*${NC}"; echo -e "${CYAN}========================================${NC}"; }
log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# --- Preflight checks -------------------------------------------------------
preflight() {
    log_step "Preflight checks"

    if [ ! -d "$DATASET_ROOT/train" ]; then
        log_error "Training dataset not found at $DATASET_ROOT/train"
        exit 1
    fi
    if [ ! -d "$DATASET_ROOT/test" ]; then
        log_error "Test dataset not found at $DATASET_ROOT/test"
        exit 1
    fi

    # Activate venv if it exists
    if [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"
        log_info "Activated virtual env: $VENV_DIR"
    else
        log_warn "Virtual env not found at $VENV_DIR. Using system Python."
        log_warn "Create it: python3 -m venv $VENV_DIR && source $VENV_DIR/bin/activate"
    fi

    # Check Python
    python -c "import torch; print('PyTorch:', torch.__version__)"
    python -c "import thrember; print('thrember OK')"
    python -c "import lightgbm; print('LightGBM:', lightgbm.__version__)"
    python -c "import pefile; print('pefile OK')"

    # Create output directories
    mkdir -p "$MODELS_DIR"/{cnn,deepmd,sorel_ffnn,ember_lgbm}
    mkdir -p "$PROCESSED_DIR"/{raw,png,pickle,ember_features}

    # Flatten raw data for deep-malware-detection
    mkdir -p "$PROCESSED_DIR/raw/train"/{malware,benign}
    mkdir -p "$PROCESSED_DIR/raw/test"/{malware,benign}

    log_info "All checks passed."
}

# --- Step 1: Data prep for CNN (exe -> png) ---------------------------------
prep_cnn() {
    log_step "1/7: Converting PE -> PNG for CNN"

    python "$SCRIPTS_DIR/convert_exe_to_png.py" \
        --input_dir "$DATASET_ROOT/train" \
        --output_dir "$PROCESSED_DIR/png/train" \
        --target_size 256

    python "$SCRIPTS_DIR/convert_exe_to_png.py" \
        --input_dir "$DATASET_ROOT/test" \
        --output_dir "$PROCESSED_DIR/png/test" \
        --target_size 256

    log_info "CNN data prep done."
}

# --- Step 2: Flatten malware families for deep-malware-detection -------------
prep_deepmd_raw() {
    log_step "2/7: Flattening malware families for deep-malware-detection"

    for split in train test; do
        log_info "Processing $split set..."

        # Copy all malware family files into flat malware/ dir
        if [ -d "$DATASET_ROOT/$split/malware" ]; then
            for family_dir in "$DATASET_ROOT/$split/malware"/*/; do
                [ -d "$family_dir" ] || continue
                family_name=$(basename "$family_dir")
                log_info "  Copying family: $family_name"
                cp "$family_dir"/* "$PROCESSED_DIR/raw/$split/malware/" 2>/dev/null || true
            done
        fi

        # Copy benign
        if [ -d "$DATASET_ROOT/$split/benign" ]; then
            cp "$DATASET_ROOT/$split/benign"/* "$PROCESSED_DIR/raw/$split/benign/" 2>/dev/null || true
        fi

        log_info "  Malware files: $(ls "$PROCESSED_DIR/raw/$split/malware" | wc -l)"
        log_info "  Benign files:  $(ls "$PROCESSED_DIR/raw/$split/benign" | wc -l)"
    done
}

# --- Step 3: Extract PE headers -> pickle (deep-malware-detection) -----------
prep_deepmd_pickle() {
    log_step "3/7: Extracting PE headers -> pickle"

    export PYTHONPATH="$DETECTOR_ROOT/deep-malware-detection:$PYTHONPATH"

    # The extract_header module is at src/bin/extract_header.py
    # We need to run from the deep-malware-detection directory for imports to work
    pushd "$DETECTOR_ROOT/deep-malware-detection" > /dev/null

    for split in train test; do
        for cls in malware benign; do
            log_info "Extracting $split/$cls headers..."

            python -m src.bin.extract_header \
                --input_dir "$PROCESSED_DIR/raw/$split/$cls" \
                --output_dir "$PROCESSED_DIR/pickle/$split/$cls" \
                || log_warn "Some files in $split/$cls could not be processed (non-PE, skipped)"
        done
    done

    popd > /dev/null
    log_info "PE header extraction done."
}

# --- Step 4: Extract EMBER v3 features (shared by SOREL-FFNN + ember2024) ---
prep_ember_features() {
    log_step "4/7: Extracting EMBER v3 features"

    for split in train test; do
        log_info "Extracting features for $split set..."
        python "$SCRIPTS_DIR/extract_ember_features.py" \
            --input_dir "$DATASET_ROOT/$split" \
            --output_dir "$PROCESSED_DIR/ember_features"
    done

    # The script saves with prefix from directory name (train/test)
    # Verify output files exist
    ls -lh "$PROCESSED_DIR/ember_features/X_train.npy" \
           "$PROCESSED_DIR/ember_features/X_test.npy" \
           "$PROCESSED_DIR/ember_features/y_train_binary.npy" \
           "$PROCESSED_DIR/ember_features/y_test_binary.npy"

    log_info "EMBER v3 feature extraction done."
}

# --- Step 5: Train CNN ------------------------------------------------------
train_cnn() {
    log_step "5/7: Training CNN (malware family classification)"

    python "$SCRIPTS_DIR/train_cnn.py" \
        --train_dir "$PROCESSED_DIR/png/train" \
        --test_dir "$PROCESSED_DIR/png/test" \
        --output_dir "$MODELS_DIR/cnn" \
        --epochs 30 \
        --batch_size 32 \
        --image_size 256 \
        --seed 42

    log_info "CNN training done. Model: $MODELS_DIR/cnn/best_model.pt"
}

# --- Step 6: Train deep-malware-detection (MalConv+) ------------------------
train_deepmd() {
    log_step "6/7: Training deep-malware-detection (MalConv+ binary)"

    # The train.py script is at deep-malware-detection/src/deep_malware_detection/train.py
    # It uses relative paths internally, so cd to its directory
    pushd "$DEEPMD_SRC" > /dev/null

    python train.py \
        --device cpu \
        --model MalConvPlus \
        --benign_dir "$PROCESSED_DIR/pickle/train/benign" \
        --malware_dir "$PROCESSED_DIR/pickle/train/malware" \
        --batch_size 16 \
        --val_size 0.15 \
        --test_size 0.0 \
        --checkpoint_dir "$MODELS_DIR/deepmd" \
        --tag best_model \
        --seed 42

    popd > /dev/null
    log_info "deep-malware-detection training done. Model: $MODELS_DIR/deepmd/best_model.pt"
}

# --- Step 7: Train SOREL-FFNN -----------------------------------------------
train_sorel_ffnn() {
    log_step "7a/7: Training SOREL-FFNN (binary)"

    python "$SCRIPTS_DIR/train_sorel_ffnn.py" \
        --train_features "$PROCESSED_DIR/ember_features/X_train.npy" \
        --train_labels "$PROCESSED_DIR/ember_features/y_train_binary.npy" \
        --output_dir "$MODELS_DIR/sorel_ffnn" \
        --epochs 50 \
        --batch_size 128 \
        --seed 42

    log_info "SOREL-FFNN training done. Model: $MODELS_DIR/sorel_ffnn/best_model.pt"
}

# --- Step 8: Train ember2024 LightGBM ---------------------------------------
train_ember_lgbm() {
    log_step "7b/7: Training ember2024 LightGBM (binary + family)"

    # Binary model
    python "$SCRIPTS_DIR/train_ember_lgbm.py" \
        --train_features "$PROCESSED_DIR/ember_features/X_train.npy" \
        --train_labels "$PROCESSED_DIR/ember_features/y_train_binary.npy" \
        --output_dir "$MODELS_DIR/ember_lgbm" \
        --model_name binary_model \
        --objective binary \
        --seed 42

    # Family model (only if family labels exist)
    if [ -f "$PROCESSED_DIR/ember_features/y_train_family.npy" ]; then
        python "$SCRIPTS_DIR/train_ember_lgbm.py" \
            --train_features "$PROCESSED_DIR/ember_features/X_train.npy" \
            --train_labels "$PROCESSED_DIR/ember_features/y_train_family.npy" \
            --output_dir "$MODELS_DIR/ember_lgbm" \
            --model_name family_model \
            --objective multiclass \
            --seed 42
    else
        log_warn "Family labels not found, skipping family model."
    fi

    log_info "ember2024 LightGBM training done."
}

# --- Main -------------------------------------------------------------------
main() {
    echo -e "${CYAN}"
    echo "=============================================="
    echo "  TRAIN ALL — 4 Malware Detection Models"
    echo "=============================================="
    echo -e "${NC}"

    local START_TIME=$(date +%s)

    preflight
    prep_cnn
    prep_deepmd_raw
    prep_deepmd_pickle
    prep_ember_features
    train_cnn
    train_deepmd
    train_sorel_ffnn
    train_ember_lgbm

    local END_TIME=$(date +%s)
    local DURATION=$((END_TIME - START_TIME))

    echo -e "\n${GREEN}=============================================="
    echo "  ALL TRAINING COMPLETE"
    echo "  Duration: ${DURATION}s (~$((DURATION / 60)) min)"
    echo "=============================================="
    echo -e "${NC}"

    echo "Models saved to:"
    echo "  CNN:              $MODELS_DIR/cnn/best_model.pt"
    echo "  deep-md:          $MODELS_DIR/deepmd/best_model.pt"
    echo "  SOREL-FFNN:       $MODELS_DIR/sorel_ffnn/best_model.pt"
    echo "  ember LGBM (bin): $MODELS_DIR/ember_lgbm/binary_model.model"
    echo "  ember LGBM (fam): $MODELS_DIR/ember_lgbm/family_model.model"
}

main "$@"
