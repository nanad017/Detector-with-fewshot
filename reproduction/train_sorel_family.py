#!/usr/bin/env python3
"""Train a SOREL PENetwork backbone with a malware-family classification head."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from common import DatasetPaths, default_dataset_root, default_output_root
from sorel_family_data import prepare_sorel_family_data
from sorel_family_model import (
    SorelFamilyDataset,
    SorelFamilyNetwork,
    compute_class_weights,
)


ROOT = Path(__file__).resolve().parents[1]
SOREL_ROOT = ROOT / "SOREL-20M"
if str(SOREL_ROOT) not in sys.path:
    sys.path.insert(0, str(SOREL_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument("--output-root", type=Path, default=default_output_root() / "sorel_family")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = DatasetPaths(args.dataset_root.expanduser())
    dataset.validate()
    output = args.output_root.expanduser()
    feature_dimension, family_names = prepare_sorel_family_data(dataset, output / "data")

    import config

    config.device = args.device
    config.batch_size = args.batch_size

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_dataset = SorelFamilyDataset(output / "data", "train")
    if not train_dataset:
        raise RuntimeError("No SOREL family training samples were loaded")

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    model = SorelFamilyNetwork(
        feature_dimension=feature_dimension,
        num_families=len(family_names),
    ).to(device)
    class_weights = compute_class_weights(train_dataset.labels, len(family_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters())

    checkpoint_dir = output / "models"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = int(labels.numel())
            total_loss += float(loss.detach().cpu()) * batch_size
            correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu())
            total += batch_size
        average_loss = total_loss / total if total else 0.0
        accuracy = correct / total if total else 0.0
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"loss={average_loss:.4f} accuracy={accuracy:.4f} samples={total}",
            flush=True,
        )
        torch.save(model.state_dict(), checkpoint_dir / f"epoch_{epoch}.pt")


if __name__ == "__main__":
    main()
