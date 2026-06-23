#!/usr/bin/env python3
"""Evaluate the SOREL PENetwork family-classification adaptation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from common import default_output_root
from sorel_family_model import SorelFamilyDataset, SorelFamilyNetwork


ROOT = Path(__file__).resolve().parents[1]
SOREL_ROOT = ROOT / "SOREL-20M"
if str(SOREL_ROOT) not in sys.path:
    sys.path.insert(0, str(SOREL_ROOT))


def load_state_dict(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=default_output_root() / "sorel_family")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--feature-dimension", type=int)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import config

    config.device = args.device
    config.batch_size = args.batch_size

    output = args.output_root.expanduser()
    checkpoint = args.checkpoint or output / "models" / f"epoch_{args.epoch}.pt"
    dataset_info = json.loads((output / "data" / "dataset_info.json").read_text(encoding="utf-8"))
    family_names = json.loads((output / "data" / "family_names.json").read_text(encoding="utf-8"))
    feature_dimension = args.feature_dimension or int(dataset_info["feature_dimension"])
    device = torch.device(args.device)

    model = SorelFamilyNetwork(
        feature_dimension=feature_dimension,
        num_families=len(family_names),
    ).to(device)
    model.load_state_dict(load_state_dict(checkpoint, device))
    model.eval()

    dataset = SorelFamilyDataset(output / "data", "test")
    if not dataset:
        raise RuntimeError("No SOREL family test samples were loaded")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    labels = []
    predictions = []
    with torch.no_grad():
        for features, batch_labels in loader:
            logits = model(features.to(device))
            predictions.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
            labels.extend(batch_labels.numpy().tolist())
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    class_ids = list(range(len(family_names)))
    macro_precision, macro_recall, macro_f1, _support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=class_ids,
        average="macro",
        zero_division=0,
    )
    results = {
        "samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=class_ids,
            target_names=family_names,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=class_ids).tolist(),
    }
    (output / "test_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
