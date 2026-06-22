#!/usr/bin/env python3
"""Evaluate trained LightGBM model on test set."""
import argparse
import json
import logging
import os
import sys

import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate LightGBM model")
    parser.add_argument("--model_path", required=True, help="Path to .model file")
    parser.add_argument("--test_features", required=True, help="Path to X_test.npy")
    parser.add_argument("--test_labels", required=True, help="Path to y_test.npy")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="model", help="Name prefix for results file")
    parser.add_argument("--class_names", nargs="*", default=None, help="Optional class name list")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    X_test = np.load(args.test_features)
    y_test = np.load(args.test_labels)
    logger.info("Test set: %d samples, %d unique labels", X_test.shape[0], len(np.unique(y_test)))

    model = lgb.Booster(model_file=args.model_path)

    # Determine task from model params
    task = model.params.get("objective", "binary")
    is_multiclass = "multiclass" in task

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    if is_multiclass:
        y_pred_class = y_pred.argmax(axis=1)
    else:
        y_pred_class = (y_pred > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred_class)
    logger.info("Test accuracy: %.4f", acc)

    results = {
        "num_samples": len(y_test),
        "accuracy": float(acc),
        "confusion_matrix": confusion_matrix(y_test, y_pred_class).tolist(),
    }

    if is_multiclass:
        f1_macro = f1_score(y_test, y_pred_class, average="macro", zero_division=0)
        results["f1_macro"] = float(f1_macro)
        logger.info("Macro F1: %.4f", f1_macro)

        if args.class_names:
            cr = classification_report(y_test, y_pred_class,
                                       target_names=args.class_names, zero_division=0)
        else:
            cr = classification_report(y_test, y_pred_class, zero_division=0)
        logger.info("Classification report:\n%s", cr)
    else:
        auc = roc_auc_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred_class, zero_division=0)
        results["auc"] = float(auc)
        results["f1"] = float(f1)
        logger.info("AUC: %.4f, F1: %.4f", auc, f1)

    out_path = os.path.join(args.output_dir, f"{args.model_name}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
