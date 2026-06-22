#!/usr/bin/env python3
"""Evaluate an original DeepMD checkpoint on the custom external test split."""

import argparse
import json
import sys
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from common import default_output_root


ROOT = Path(__file__).resolve().parents[1]
DEEPMD_SRC = ROOT / "deep-malware-detection" / "src" / "deep_malware_detection"
sys.path.insert(0, str(DEEPMD_SRC))

from dataset import MalwareDataset, make_loader  # noqa: E402
from models import MalConvBase, MalConvPlus  # noqa: E402
from utils import predict  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=default_output_root() / "deepmd")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model", choices=["MalConvBase", "MalConvPlus"], default="MalConvPlus")
    parser.add_argument("--embed-dim", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--out-channels", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output = args.output_root.expanduser()
    checkpoint = args.checkpoint or output / "models" / f"{args.model}.pt"
    device = torch.device(args.device)
    model_class = MalConvPlus if args.model == "MalConvPlus" else MalConvBase
    model = model_class(
        args.embed_dim, args.max_len, args.out_channels, args.window_size, args.dropout
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))

    dataset = MalwareDataset(
        str(output / "pickle" / "test" / "benign"),
        str(output / "pickle" / "test" / "malware"),
    )
    loader = make_loader(dataset, args.batch_size, shuffle=False)
    y_true, probabilities = predict(model, loader, device, apply_sigmoid=True)
    predictions = (probabilities >= 0.5).astype(int)
    results = {
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }
    results_path = output / "test_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

