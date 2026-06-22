#!/usr/bin/env python3
"""
Extract EMBER v3 feature vectors from raw PE files.

Produces .npy files for training SOREL-FFNN and ember2024 LightGBM models.
Supports two label modes:
  - binary: 0 = benign, 1 = malware (all families merged)
  - family: integer per family subdirectory + 0 for benign (if present)

Directory structure expected:
  input_dir/
    malware/
      FamilyA/
        *.exe
      FamilyB/
        *.exe
    benign/
      *.exe
"""
import argparse
import logging
import os
import sys
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_features_for_directory(input_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Walk input_dir, extract EMBER v3 features from each PE file.

    Returns:
        X: feature vectors (2386-dim)
        y_binary: 0/1 labels
        y_family: integer family labels (-1 for benign if no family)
        sha256_list: placeholder list of indices
    """
    from thrember.features import PEFeatureExtractor

    extractor = PEFeatureExtractor()
    features_list = []
    binary_labels = []
    family_labels = []

    # Discover all PE files
    all_files = []
    malware_root = os.path.join(input_dir, "malware")
    benign_root = os.path.join(input_dir, "benign")

    # Index families
    family_names = []
    family_to_idx = {}

    if os.path.isdir(malware_root):
        for family in sorted(os.listdir(malware_root)):
            family_dir = os.path.join(malware_root, family)
            if os.path.isdir(family_dir):
                if family not in family_to_idx:
                    family_to_idx[family] = len(family_names)
                    family_names.append(family)
                for fname in os.listdir(family_dir):
                    if fname.lower().endswith(('.exe', '.dll')):
                        all_files.append((os.path.join(family_dir, fname), 1, family_to_idx[family]))

    if os.path.isdir(benign_root):
        for fname in os.listdir(benign_root):
            if fname.lower().endswith(('.exe', '.dll')):
                # Benign gets family label = len(family_names) if we have families, else -1
                benign_family = len(family_names)  # own class
                all_files.append((os.path.join(benign_root, fname), 0, benign_family))

    if benign_root and os.path.isdir(benign_root) and len(family_names) > 0:
        family_names.append("benign")

    logger.info("Found %d PE files across %d families + benign", len(all_files), len(family_names) - (1 if "benign" in family_names else 0))

    skipped = 0
    for filepath, bin_label, fam_label in tqdm(all_files, desc="Extracting features"):
        try:
            with open(filepath, "rb") as f:
                bytez = f.read()
            vec = extractor.feature_vector(bytez)
            features_list.append(vec)
            binary_labels.append(bin_label)
            family_labels.append(fam_label)
        except Exception:
            skipped += 1
            continue

    if skipped > 0:
        logger.warning("Skipped %d files (parse error)", skipped)

    X = np.array(features_list, dtype=np.float32)
    y_bin = np.array(binary_labels, dtype=np.int32)
    y_fam = np.array(family_labels, dtype=np.int32)

    return X, y_bin, y_fam


def main():
    parser = argparse.ArgumentParser(description="Extract EMBER v3 features from PE files")
    parser.add_argument("--input_dir", required=True, help="Directory containing malware/ and benign/ subdirs")
    parser.add_argument("--output_dir", required=True, help="Directory to save .npy files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    X, y_bin, y_fam = extract_features_for_directory(args.input_dir)

    # Use filename prefix to distinguish train vs test
    # Determine prefix from input_dir name
    dir_name = os.path.basename(os.path.normpath(args.input_dir))  # "train" or "test"
    prefix = dir_name if dir_name in ("train", "test") else "data"

    np.save(os.path.join(args.output_dir, f"X_{prefix}.npy"), X)
    np.save(os.path.join(args.output_dir, f"y_{prefix}_binary.npy"), y_bin)
    np.save(os.path.join(args.output_dir, f"y_{prefix}_family.npy"), y_fam)

    logger.info("Saved features to %s: X_%s.npy (%s), y_%s_binary.npy, y_%s_family.npy",
                args.output_dir, prefix, X.shape, prefix, prefix)


if __name__ == "__main__":
    main()
