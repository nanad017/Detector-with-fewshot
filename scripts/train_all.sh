#!/usr/bin/env bash
# =============================================================================
# train_all.sh — Train all 4 malware detection models on custom dataset
#
# Usage:
#   chmod +x scripts/train_all.sh
#   bash scripts/train_all.sh
#
# Dataset structure:
#   ~/An_solo/dataset/Virus/Virus train/{Locker,Mediyes,Winwebsec,Zbot,Zeroaccess}/
#   ~/An_solo/dataset/Virus/Virus test/{Locker,Mediyes,Winwebsec,Zbot,Zeroaccess}/
#   ~/An_solo/dataset/Benign/Benign train/
#   ~/An_solo/dataset/Benign/Benign test/
# =============================================================================

set -euo pipefail

# --- Configuration ----------------------------------------------------------
DETECTOR_ROOT="$HOME/An_solo/detector"
DATASET_ROOT="$HOME/An_solo/dataset"
MALWARE_TRAIN="$DATASET_ROOT/Virus/Virus train"
MALWARE_TEST="$DATASET_ROOT/Virus/Virus test"
BENIGN_TRAIN="$DATASET_ROOT/Benign/Benign train"
BENIGN_TEST="$DATASET_ROOT/Benign/Benign test"
VENV_DIR="$DETECTOR_ROOT/venv"
PROCESSED_DIR="$DETECTOR_ROOT/processed_data"
MODELS_DIR="$DETECTOR_ROOT/models"
SCRIPTS_DIR="$DETECTOR_ROOT/scripts"
DEEPMD_SRC="$DETECTOR_ROOT/deep-malware-detection/src/deep_malware_detection"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log_step()  { echo -e "\n${CYAN}[STEP] $*${NC}"; }
log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# --- Preflight --------------------------------------------------------------
preflight() {
    log_step "Preflight checks"

    if [ ! -d "$MALWARE_TRAIN" ]; then
        log_error "Malware train not found: $MALWARE_TRAIN"; exit 1
    fi
    if [ ! -d "$MALWARE_TEST" ]; then
        log_error "Malware test not found: $MALWARE_TEST"; exit 1
    fi
    if [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"
        log_info "Activated virtual env"
    else
        log_warn "Virtual env not found, using system Python"
    fi
    python -c "import torch; print('PyTorch:', torch.__version__)"
    python -c "import thrember; print('thrember OK')"
    python -c "import lightgbm; print('LightGBM:', lightgbm.__version__)"
    python -c "import pefile; print('pefile OK')"

    mkdir -p "$MODELS_DIR"/{cnn,deepmd,sorel_ffnn,ember_lgbm}
    mkdir -p "$PROCESSED_DIR"/{raw/malware,raw/benign,png/train,png/test,pickle/malware,pickle/benign,ember_features}
    mkdir -p "$PROCESSED_DIR/raw/train"/{malware,benign}
    mkdir -p "$PROCESSED_DIR/raw/test"/{malware,benign}
    log_info "All checks passed."
}

# --- Step 1: Convert PE -> PNG for CNN --------------------------------------
prep_cnn() {
    log_step "1/7: Converting PE -> PNG (CNN)"

    # Convert malware families train
    python "$SCRIPTS_DIR/convert_exe_to_png.py" \
        --input_dir "$MALWARE_TRAIN" \
        --output_dir "$PROCESSED_DIR/png/train" \
        --target_size 256

    # Convert malware families test
    python "$SCRIPTS_DIR/convert_exe_to_png.py" \
        --input_dir "$MALWARE_TEST" \
        --output_dir "$PROCESSED_DIR/png/test" \
        --target_size 256

    # Convert benign train + test (flat dir, pass class_name)
    python "$SCRIPTS_DIR/convert_exe_to_png.py" \
        --input_dir "$BENIGN_TRAIN" \
        --output_dir "$PROCESSED_DIR/png/train" \
        --target_size 256 --class_name Benign

    python "$SCRIPTS_DIR/convert_exe_to_png.py" \
        --input_dir "$BENIGN_TEST" \
        --output_dir "$PROCESSED_DIR/png/test" \
        --target_size 256 --class_name Benign

    log_info "CNN data prep done."
}

# --- Step 2: Flatten malware families for deep-malware-detection -------------
prep_deepmd_raw() {
    log_step "2/7: Flattening malware families"

    for split_dir in "$MALWARE_TRAIN" "$MALWARE_TEST"; do
        local split_name
        if [[ "$split_dir" == *"train" ]]; then split_name="train"; else split_name="test"; fi
        log_info "Processing $split_name..."

        # Copy all family files into flat malware dir
        for family_dir in "$split_dir"/*/; do
            [ -d "$family_dir" ] || continue
            cp "$family_dir"/* "$PROCESSED_DIR/raw/$split_name/malware/" 2>/dev/null || true
        done

        log_info "  Malware files: $(ls "$PROCESSED_DIR/raw/$split_name/malware" 2>/dev/null | wc -l)"
    done

    cp "$BENIGN_TRAIN"/* "$PROCESSED_DIR/raw/train/benign/" 2>/dev/null || true
    cp "$BENIGN_TEST"/* "$PROCESSED_DIR/raw/test/benign/" 2>/dev/null || true
    log_info "  Benign files: $(ls "$PROCESSED_DIR/raw/train/benign" 2>/dev/null | wc -l) train, $(ls "$PROCESSED_DIR/raw/test/benign" 2>/dev/null | wc -l) test"
}

# --- Step 3: Extract PE headers -> pickle (deep-malware-detection) -----------
prep_deepmd_pickle() {
    log_step "3/7: Extracting PE headers -> pickle"

    export PYTHONPATH="$DETECTOR_ROOT/deep-malware-detection:$PYTHONPATH"
    pushd "$DETECTOR_ROOT/deep-malware-detection" > /dev/null

    for split in train test; do
        for cls in malware benign; do
            local in_dir="$PROCESSED_DIR/raw/$split/$cls"
            local out_dir="$PROCESSED_DIR/pickle/$split/$cls"
            if [ -d "$in_dir" ] && [ "$(ls -A "$in_dir" 2>/dev/null)" ]; then
                log_info "Extracting $split/$cls headers..."
                python -m src.bin.extract_header --input_dir "$in_dir" --output_dir "$out_dir" || log_warn "Some files in $split/$cls could not be processed"
            else
                log_warn "No files in $split/$cls, skipping"
            fi
        done
    done

    popd > /dev/null
    log_info "PE header extraction done."
}

# --- Step 4: Extract EMBER v3 features --------------------------------------
prep_ember_features() {
    log_step "4/7: Extracting EMBER v3 features"

    python "$SCRIPTS_DIR/extract_ember_features.py" \
        --malware_train "$MALWARE_TRAIN" \
        --malware_test "$MALWARE_TEST" \
        --benign_train "$BENIGN_TRAIN" \
        --benign_test "$BENIGN_TEST" \
        --output_dir "$PROCESSED_DIR/ember_features"

    ls -lh "$PROCESSED_DIR/ember_features/"*.npy
    log_info "EMBER v3 feature extraction done."
}

# --- Step 5: Train CNN ------------------------------------------------------
train_cnn() {
    log_step "5/7: Training CNN (family classification)"

    python "$SCRIPTS_DIR/train_cnn.py" \
        --train_dir "$PROCESSED_DIR/png/train" \
        --test_dir "$PROCESSED_DIR/png/test" \
        --output_dir "$MODELS_DIR/cnn" \
        --epochs 30 --batch_size 32 --image_size 256 --seed 42

    log_info "CNN done."
}

# --- Step 6: Train deep-malware-detection ------------------------------------
train_deepmd() {
    log_step "6/7: Training deep-malware-detection"

    pushd "$DEEPMD_SRC" > /dev/null

    # MalConvBase
    python train.py --device cpu --model MalConvBase \
        --benign_dir "$PROCESSED_DIR/pickle/train/benign" \
        --malware_dir "$PROCESSED_DIR/pickle/train/malware" \
        --batch_size 16 --val_size 0.15 --test_size 0.0 \
        --checkpoint_dir "$MODELS_DIR/deepmd" --tag MalConvBase --seed 42

    # MalConvPlus variants
    python train.py --device cpu --model MalConvPlus \
        --benign_dir "$PROCESSED_DIR/pickle/train/benign" \
        --malware_dir "$PROCESSED_DIR/pickle/train/malware" \
        --batch_size 16 --val_size 0.15 --test_size 0.0 \
        --checkpoint_dir "$MODELS_DIR/deepmd" --tag MalConvPlus --seed 42

    python train.py --device cpu --model MalConvPlus \
        --benign_dir "$PROCESSED_DIR/pickle/train/benign" \
        --malware_dir "$PROCESSED_DIR/pickle/train/malware" \
        --batch_size 16 --val_size 0.15 --test_size 0.0 \
        --checkpoint_dir "$MODELS_DIR/deepmd" --tag MalConvPlus_E16 --seed 42 --embed_dim 16

    python train.py --device cpu --model MalConvPlus \
        --benign_dir "$PROCESSED_DIR/pickle/train/benign" \
        --malware_dir "$PROCESSED_DIR/pickle/train/malware" \
        --batch_size 16 --val_size 0.15 --test_size 0.0 \
        --checkpoint_dir "$MODELS_DIR/deepmd" --tag MalConvPlus_W64 --seed 42 --window_size 64

    python train.py --device cpu --model MalConvPlus \
        --benign_dir "$PROCESSED_DIR/pickle/train/benign" \
        --malware_dir "$PROCESSED_DIR/pickle/train/malware" \
        --batch_size 16 --val_size 0.15 --test_size 0.0 \
        --checkpoint_dir "$MODELS_DIR/deepmd" --tag MalConvPlus_E16W64 --seed 42 --embed_dim 16 --window_size 64

    python train.py --device cpu --model MalConvPlus \
        --benign_dir "$PROCESSED_DIR/pickle/train/benign" \
        --malware_dir "$PROCESSED_DIR/pickle/train/malware" \
        --batch_size 16 --val_size 0.15 --test_size 0.0 \
        --checkpoint_dir "$MODELS_DIR/deepmd" --tag MalConvPlus_C256 --seed 42 --out_channels 256

    popd > /dev/null
    log_info "deep-malware-detection done."
}

# --- Step 7: Train SOREL-FFNN -----------------------------------------------
train_sorel_ffnn() {
    log_step "7a/7: Training SOREL-FFNN"

    python "$SCRIPTS_DIR/train_sorel_ffnn.py" \
        --train_features "$PROCESSED_DIR/ember_features/X_train.npy" \
        --train_labels "$PROCESSED_DIR/ember_features/y_train_binary.npy" \
        --output_dir "$MODELS_DIR/sorel_ffnn" \
        --epochs 50 --batch_size 128 --seed 42

    log_info "SOREL-FFNN done."
}

# --- Step 8: Train ember2024 LightGBM ---------------------------------------
train_ember_lgbm() {
    log_step "7b/7: Training ember2024 LightGBM"

    python "$SCRIPTS_DIR/train_ember_lgbm.py" \
        --train_features "$PROCESSED_DIR/ember_features/X_train.npy" \
        --train_labels "$PROCESSED_DIR/ember_features/y_train_binary.npy" \
        --output_dir "$MODELS_DIR/ember_lgbm" --model_name binary_model \
        --objective binary --seed 42

    if [ -f "$PROCESSED_DIR/ember_features/y_train_family.npy" ]; then
        python "$SCRIPTS_DIR/train_ember_lgbm.py" \
            --train_features "$PROCESSED_DIR/ember_features/X_train.npy" \
            --train_labels "$PROCESSED_DIR/ember_features/y_train_family.npy" \
            --output_dir "$MODELS_DIR/ember_lgbm" --model_name family_model \
            --objective multiclass --seed 42
    else
        log_warn "Family labels not found, skipping family LGBM"
    fi

    log_info "ember2024 LGBM done."
}

# --- Main -------------------------------------------------------------------
main() {
    local START_TIME=$(date +%s)
    preflight; prep_cnn; prep_deepmd_raw; prep_deepmd_pickle; prep_ember_features
    train_cnn; train_deepmd; train_sorel_ffnn; train_ember_lgbm
    local DURATION=$(( $(date +%s) - START_TIME ))

    echo -e "\n${GREEN}ALL TRAINING COMPLETE — ${DURATION}s (~$((DURATION/60)) min)${NC}"
    echo "Models saved to:"
    echo "  CNN:              $MODELS_DIR/cnn/best_model.pt"
    echo "  deep-md:          $MODELS_DIR/deepmd/*.pt"
    echo "  SOREL-FFNN:       $MODELS_DIR/sorel_ffnn/best_model.pt"
    echo "  ember LGBM:       $MODELS_DIR/ember_lgbm/*.model"
}

main "$@"
