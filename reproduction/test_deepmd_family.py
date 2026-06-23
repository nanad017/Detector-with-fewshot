#!/usr/bin/env python3
"""Evaluate the DeepMD MalConv family-classification adaptation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from common import default_output_root
from deepmd_family import (
    DeepMDFamilyDataset,
    build_family_model,
    make_family_loader,
    write_missing_records,
)


DEFAULT_CONFIG = {
    "model": "MalConvPlus",
    "embed_dim": 8,
    "max_len": 4096,
    "out_channels": 128,
    "window_size": 32,
    "dropout": 0.5,
    "batch_size": 8,
    "tag": "MalConvPlus_family",
    "checkpoint_name": "MalConvPlus_family.pt",
}


def resolve_config(output: Path, args) -> dict:
    config = dict(DEFAULT_CONFIG)
    config_path = output / "training_config.json"
    has_training_config = config_path.is_file()
    if has_training_config:
        config.update(json.loads(config_path.read_text(encoding="utf-8")))
    for key in (
        "model",
        "embed_dim",
        "max_len",
        "out_channels",
        "window_size",
        "dropout",
        "batch_size",
        "tag",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.tag is not None:
        config["checkpoint_name"] = f"{args.tag}.pt"
    elif not has_training_config and args.model is not None:
        config["checkpoint_name"] = f"{config['model']}_family.pt"
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", type=Path, default=default_output_root() / "deepmd_family"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", choices=["MalConvBase", "MalConvPlus"])
    parser.add_argument("--embed-dim", type=int)
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--out-channels", type=int)
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--tag")
    args = parser.parse_args()

    output = args.output_root.expanduser()
    config = resolve_config(output, args)
    family_names = json.loads((output / "class_names.json").read_text(encoding="utf-8"))
    checkpoint = args.checkpoint or output / "models" / config["checkpoint_name"]
    device = torch.device(args.device)

    model = build_family_model(
        config["model"],
        len(family_names),
        config["embed_dim"],
        config["max_len"],
        config["out_channels"],
        config["window_size"],
        config["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    test_dataset = DeepMDFamilyDataset(output, "test", family_names)
    write_missing_records(test_dataset, output)
    if len(test_dataset) == 0:
        raise RuntimeError("No DeepMD family test samples were loaded")
    test_loader = make_family_loader(test_dataset, config["batch_size"], shuffle=False)

    labels: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for inputs, batch_labels in test_loader:
            logits = model(inputs.to(device))
            predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
            labels.extend(batch_labels.to(torch.long).cpu().tolist())

    class_ids = list(range(len(family_names)))
    precision, recall, f1, _support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=class_ids,
        average="macro",
        zero_division=0,
    )
    results = {
        "samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "class_names": family_names,
        "classification_report": classification_report(
            labels,
            predictions,
            labels=class_ids,
            target_names=family_names,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=class_ids
        ).tolist(),
    }
    results_path = output / "test_results.json"
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
