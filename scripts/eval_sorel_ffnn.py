#!/usr/bin/env python3
"""Evaluate trained SOREL-FFNN model on test set."""
import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from train_sorel_ffnn import SorelFFNN


def main():
    parser = argparse.ArgumentParser(description="Evaluate SOREL-FFNN model")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_features", required=True, help="Path to X_test.npy")
    parser.add_argument("--test_labels", required=True, help="Path to y_test.npy")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    X_test = np.load(args.test_features)
    y_test = np.load(args.test_labels).astype(np.float32)
    logger.info("Test set: %d samples, %d features", X_test.shape[0], X_test.shape[1])

    feature_dim = X_test.shape[1]
    model = SorelFFNN(feature_dim=feature_dim).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()

    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    all_preds = []
    all_labels = []
    for inputs, labels in tqdm(test_loader, desc="Evaluating"):
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        all_preds.extend(outputs.cpu().numpy())
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    preds_bin = (all_preds >= 0.5).astype(int)

    results = {
        "num_samples": len(all_labels),
        "auc": float(roc_auc_score(all_labels, all_preds)),
        "accuracy": float(accuracy_score(all_labels, preds_bin)),
        "f1": float(f1_score(all_labels, preds_bin, zero_division=0)),
        "confusion_matrix": confusion_matrix(all_labels, preds_bin).tolist(),
    }

    logger.info("AUC: %.4f, Accuracy: %.4f, F1: %.4f",
                results["auc"], results["accuracy"], results["f1"])

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results saved to %s/results.json", args.output_dir)


if __name__ == "__main__":
    main()
