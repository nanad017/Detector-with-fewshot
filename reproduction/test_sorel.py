#!/usr/bin/env python3
"""Evaluate the original SOREL PENetwork malware head on the external test split."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from common import default_output_root


ROOT = Path(__file__).resolve().parents[1]
SOREL_ROOT = ROOT / "SOREL-20M"
sys.path.insert(0, str(SOREL_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=default_output_root() / "sorel")
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
    from dataset import Dataset
    from generators import get_generator
    from nets import PENetwork

    output = args.output_root.expanduser()
    checkpoint = args.checkpoint or output / "models" / f"epoch_{args.epoch}.pt"
    dataset_info = json.loads((output / "data" / "dataset_info.json").read_text(encoding="utf-8"))
    feature_dimension = args.feature_dimension or int(dataset_info["feature_dimension"])
    device = torch.device(args.device)
    model = PENetwork(
        use_malware=True,
        use_counts=True,
        use_tags=True,
        n_tags=len(Dataset.tags),
        feature_dimension=feature_dimension,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    generator = get_generator(
        mode="test",
        path=str(output / "data"),
        use_malicious_labels=True,
        use_count_labels=False,
        use_tag_labels=False,
        batch_size=args.batch_size,
        num_workers=args.workers,
        remove_missing_features=False,
        shuffle=False,
    )
    labels = []
    probabilities = []
    with torch.no_grad():
        for features, batch_labels in generator:
            output_dict = model(features.to(device))
            probabilities.extend(output_dict["malware"].detach().cpu().numpy().ravel())
            labels.extend(batch_labels["malware"].numpy().ravel())
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= 0.5).astype(int)
    results = {
        "samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }
    (output / "test_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
