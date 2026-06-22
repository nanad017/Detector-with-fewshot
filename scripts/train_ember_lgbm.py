#!/usr/bin/env python3
"""
Train LightGBM model using EMBER v3 features.

Supports:
  - binary classification (malware vs benign)
  - multiclass classification (malware family)

Input: .npy feature files from extract_ember_features.py
"""
import argparse
import json
import logging
import os
import sys

import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM model for malware classification")
    parser.add_argument("--train_features", required=True, help="Path to X_train.npy")
    parser.add_argument("--train_labels", required=True, help="Path to y_train.npy (binary or family)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="model", help="Name prefix for saved model")
    parser.add_argument("--objective", choices=["binary", "multiclass"], required=True,
                        help="binary for malware/benign, multiclass for family classification")
    parser.add_argument("--val_size", type=float, default=0.15, help="Validation split from training data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    # Load data
    X = np.load(args.train_features)
    y = np.load(args.train_labels)
    logger.info("Loaded features: %s, labels: %s", X.shape, y.shape)
    logger.info("Unique labels: %s, counts: %s", np.unique(y), np.bincount(y.astype(int)))

    # Split
    if args.objective == "binary":
        stratify = y
    else:
        stratify = y

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, random_state=args.seed, stratify=stratify
    )

    num_classes = len(np.unique(y))
    logger.info("Train: %d, Val: %d, Num classes: %d", len(X_train), len(X_val), num_classes)

    # LightGBM parameters
    if args.objective == "binary":
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "seed": args.seed,
        }
    else:
        params = {
            "objective": "multiclass",
            "num_class": num_classes,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "seed": args.seed,
        }

    train_set = lgb.Dataset(X_train, y_train)
    val_set = lgb.Dataset(X_val, y_val, reference=train_set)

    logger.info("Training LightGBM...")
    model = lgb.train(
        params,
        train_set,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        num_boost_round=500,
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(50),
        ],
    )

    # Evaluate on validation set
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    if args.objective == "multiclass":
        y_pred_class = y_pred.argmax(axis=1)
    else:
        y_pred_class = (y_pred > 0.5).astype(int)

    acc = accuracy_score(y_val, y_pred_class)
    logger.info("Validation accuracy: %.4f", acc)
    if args.objective == "binary":
        auc = roc_auc_score(y_val, y_pred)
        logger.info("Validation AUC: %.4f", auc)
    logger.info("Best iteration: %d", model.best_iteration)

    # Save model
    model_path = os.path.join(args.output_dir, f"{args.model_name}.model")
    model.save_model(model_path)
    logger.info("Model saved to %s", model_path)


if __name__ == "__main__":
    main()
