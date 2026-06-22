#!/usr/bin/env python3
"""
Train a feed-forward neural network (SOREL-FFNN) for malware binary classification.

Architecture inspired by SOREL-20M's PENetwork:
  Linear(feature_dim, 512) → LayerNorm → ELU → Dropout
  → Linear(512, 512) → LayerNorm → ELU → Dropout
  → Linear(512, 128) → LayerNorm → ELU → Dropout
  → Linear(128, 1) → Sigmoid

Input: EMBER v3 feature vectors (2386-dim) from extract_ember_features.py
"""
import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SorelFFNN(nn.Module):
    """Feed-forward network loosely based on SOREL-20M PENetwork."""
    def __init__(self, feature_dim: int = 2386, layer_sizes: list = None, dropout: float = 0.05):
        super().__init__()
        if layer_sizes is None:
            layer_sizes = [512, 512, 128]

        layers = []
        prev_dim = feature_dim
        for ls in layer_sizes:
            layers.append(nn.Linear(prev_dim, ls))
            layers.append(nn.LayerNorm(ls))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = ls
        self.base = nn.Sequential(*layers)
        self.output = nn.Sequential(
            nn.Linear(layer_sizes[-1], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.base(x)
        return self.output(x).squeeze(1)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    for inputs, labels in tqdm(loader, desc="Train", leave=False):
        inputs, labels = inputs.to(device), labels.to(torch.float32).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    return total_loss / len(all_labels), auc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    for inputs, labels in tqdm(loader, desc="Eval", leave=False):
        inputs, labels = inputs.to(device), labels.to(torch.float32).to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0)
        all_preds.extend(outputs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    # Accuracy at threshold 0.5
    preds_bin = (np.array(all_preds) >= 0.5).astype(int)
    acc = (preds_bin == np.array(all_labels)).mean()
    return total_loss / len(all_labels), auc, acc


def main():
    parser = argparse.ArgumentParser(description="Train SOREL-FFNN for binary malware classification")
    parser.add_argument("--train_features", required=True, help="Path to X_train.npy")
    parser.add_argument("--train_labels", required=True, help="Path to y_train_binary.npy")
    parser.add_argument("--output_dir", required=True, help="Directory for model and history")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val_size", type=float, default=0.15, help="Validation split from training data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    X = np.load(args.train_features)
    y = np.load(args.train_labels).astype(np.float32)
    logger.info("Loaded features: %s, labels: %s", X.shape, y.shape)

    # Split train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, random_state=args.seed, stratify=(y > 0.5).astype(int)
    )

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    feature_dim = X.shape[1]
    model = SorelFFNN(feature_dim=feature_dim).to(device)
    logger.info("Model params: %d, feature dim: %d", sum(p.numel() for p in model.parameters()), feature_dim)

    # Compute pos_weight for class imbalance
    pos_count = (y_train > 0.5).sum()
    neg_count = len(y_train) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    best_auc = 0
    history = {"train_loss": [], "train_auc": [], "val_loss": [], "val_auc": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        train_loss, train_auc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(float(train_loss))
        history["train_auc"].append(float(train_auc))
        history["val_loss"].append(float(val_loss))
        history["val_auc"].append(float(val_auc))
        history["val_acc"].append(float(val_acc))

        logger.info("Epoch %d/%d — Train loss: %.4f, AUC: %.4f | Val loss: %.4f, AUC: %.4f, Acc: %.4f",
                    epoch, args.epochs, train_loss, train_auc, val_loss, val_auc, val_acc)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            logger.info("  -> Saved best model (AUC=%.4f)", best_auc)

    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Training complete. Best val AUC: %.4f", best_auc)


if __name__ == "__main__":
    main()
