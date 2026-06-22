#!/usr/bin/env python3
"""
Extract EMBER v3 feature vectors from PE files in actual dataset structure.

Expected structure:
  ~/An_solo/dataset/
    Virus/Virus train/{Locker,Mediyes,Winwebsec,Zbot,Zeroaccess}/*.exe
    Virus/Virus test/{Locker,Mediyes,Winwebsec,Zbot,Zeroaccess}/*.exe
    Benign/Benign train/*.exe
    Benign/Benign test/*.exe

Outputs (to --output_dir):
  X_train.npy, y_train_binary.npy, y_train_family.npy
  X_test.npy,  y_test_binary.npy,  y_test_family.npy
"""
import argparse
import logging
import os
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def collect_files(malware_dir: str, benign_dir: str) -> tuple:
    """Walk malware families + benign dir, return (paths, binary_labels, family_labels, family_names)."""
    files = []
    binary_labels = []
    family_labels = []
    family_names = []
    family_idx = {}

    # Collect malware by family
    if os.path.isdir(malware_dir):
        for family in sorted(os.listdir(malware_dir)):
            fdir = os.path.join(malware_dir, family)
            if not os.path.isdir(fdir):
                continue
            if family not in family_idx:
                family_idx[family] = len(family_names)
                family_names.append(family)
            for fname in os.listdir(fdir):
                if fname.lower().endswith(('.exe', '.dll')):
                    files.append(os.path.join(fdir, fname))
                    binary_labels.append(1)
                    family_labels.append(family_idx[family])

    # Collect benign
    benign_fid = len(family_names)
    family_names.append("Benign")
    if os.path.isdir(benign_dir):
        for fname in os.listdir(benign_dir):
            if fname.lower().endswith(('.exe', '.dll')):
                files.append(os.path.join(benign_dir, fname))
                binary_labels.append(0)
                family_labels.append(benign_fid)

    return files, np.array(binary_labels, dtype=np.int32), np.array(family_labels, dtype=np.int32), family_names


def extract_split(bytez_list: list, output_dir: str, prefix: str):
    """Extract EMBER v3 features from bytez and save as .npy."""
    from thrember.features import PEFeatureExtractor

    extractor = PEFeatureExtractor()
    X_list = []
    skipped = 0

    for bytez in tqdm(bytez_list, desc=f"Extract {prefix}"):
        try:
            vec = extractor.feature_vector(bytez)
            X_list.append(vec)
        except Exception:
            skipped += 1

    if skipped:
        logger.warning("Skipped %d files in %s", skipped, prefix)

    X = np.array(X_list, dtype=np.float32)
    np.save(os.path.join(output_dir, f"X_{prefix}.npy"), X)
    logger.info("Saved X_%s.npy (%s)", prefix, X.shape)


def main():
    parser = argparse.ArgumentParser(description="Extract EMBER v3 features")
    parser.add_argument("--malware_train", required=True)
    parser.add_argument("--malware_test", required=True)
    parser.add_argument("--benign_train", required=True)
    parser.add_argument("--benign_test", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for split, mdir, bdir in [("train", args.malware_train, args.benign_train),
                               ("test", args.malware_test, args.benign_test)]:
        files, y_bin, y_fam, families = collect_files(mdir, bdir)
        logger.info("%s: %d files, %d families: %s", split, len(files), len(families), families)

        if not files:
            logger.error("No files found in %s!", split)
            continue

        # Read all bytes
        bytez_list = []
        for fp in files:
            try:
                with open(fp, "rb") as f:
                    bytez_list.append(f.read())
            except Exception:
                pass

        extract_split(bytez_list, args.output_dir, split)
        np.save(os.path.join(args.output_dir, f"y_{split}_binary.npy"), y_bin[:len(bytez_list)])
        np.save(os.path.join(args.output_dir, f"y_{split}_family.npy"), y_fam[:len(bytez_list)])
        logger.info("Saved y_%s_binary.npy, y_%s_family.npy", split, split)


if __name__ == "__main__":
    main()
