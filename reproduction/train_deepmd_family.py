#!/usr/bin/env python3
"""Train a DeepMD MalConv backbone with a malware-family classification head."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Subset

from common import DatasetPaths, default_dataset_root, default_output_root
from deepmd_family import (
    DEEPMD_SRC,
    ROOT,
    DeepMDFamilyDataset,
    build_family_model,
    compute_class_weights,
    make_family_loader,
    prepare_deepmd_family_inputs,
    split_train_val_indices,
    write_missing_records,
)


DEEPMD_ROOT = ROOT / "deep-malware-detection"


def run(command: list[str], **kwargs) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, **kwargs)


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def run_epoch(
    model: nn.Module,
    data_loader,
    device: torch.device,
    criterion,
    optimizer=None,
) -> float:
    total_loss = 0.0
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device, dtype=torch.long)
        logits = model(inputs)
        loss = criterion(logits, labels)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument(
        "--output-root", type=Path, default=default_output_root() / "deepmd_family"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", choices=["MalConvBase", "MalConvPlus"], default="MalConvPlus")
    parser.add_argument("--embed-dim", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--out-channels", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--tag")
    args = parser.parse_args()
    tag = args.tag or f"{args.model}_family"

    set_seed(args.seed)
    dataset_paths = DatasetPaths(args.dataset_root.expanduser())
    dataset_paths.validate()
    output = args.output_root.expanduser()
    family_names = prepare_deepmd_family_inputs(dataset_paths, output)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(DEEPMD_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    for split in ("train", "test"):
        for family in family_names:
            run(
                [
                    sys.executable,
                    "-m",
                    "src.bin.extract_header",
                    "--input_dir",
                    str(output / "raw" / split / family),
                    "--output_dir",
                    str(output / "pickle" / split / family),
                ],
                cwd=DEEPMD_ROOT,
                env=env,
            )

    train_dataset = DeepMDFamilyDataset(output, "train", family_names)
    write_missing_records(train_dataset, output)
    if len(train_dataset) == 0:
        raise RuntimeError("No DeepMD family training samples were loaded")

    train_indices, val_indices = split_train_val_indices(
        train_dataset.labels, args.val_size, args.seed
    )
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices) if val_indices else None
    train_loader = make_family_loader(train_subset, args.batch_size, shuffle=True)
    val_loader = (
        make_family_loader(val_subset, args.batch_size, shuffle=False)
        if val_subset is not None
        else None
    )

    train_labels = [train_dataset.labels[index] for index in train_indices]
    class_weights = compute_class_weights(train_labels, len(family_names)).to(args.device)
    model = build_family_model(
        args.model,
        len(family_names),
        args.embed_dim,
        args.max_len,
        args.out_channels,
        args.window_size,
        args.dropout,
    ).to(args.device)
    device = torch.device(args.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=args.patience
    )

    models_dir = output / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / f"{tag}.pt"
    best_loss: float | None = None
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = run_epoch(model, train_loader, device, criterion, optimizer)
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_loss = run_epoch(model, val_loader, device, criterion)
            monitor_loss = val_loss
        else:
            val_loss = None
            monitor_loss = train_loss

        scheduler.step(monitor_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        val_text = f"{val_loss:.4f}" if val_loss is not None else "n/a"
        print(
            f"Epoch [{epoch}/{args.epochs}], "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_text}",
            flush=True,
        )

        if best_loss is None or monitor_loss < best_loss:
            best_loss = monitor_loss
            bad_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            bad_epochs += 1
            if bad_epochs > args.patience:
                break

    config = {
        "model": args.model,
        "embed_dim": args.embed_dim,
        "max_len": args.max_len,
        "out_channels": args.out_channels,
        "window_size": args.window_size,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "tag": tag,
        "checkpoint_name": checkpoint_path.name,
        "num_classes": len(family_names),
        "class_names": family_names,
        "train_samples": len(train_subset),
        "val_samples": len(val_subset) if val_subset is not None else 0,
    }
    (output / "training_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "train_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"Saved checkpoint: {checkpoint_path}", flush=True)
    print(f"DeepMD family classes: {len(family_names)}", flush=True)
    print(f"DeepMD source imported from: {DEEPMD_SRC}", flush=True)


if __name__ == "__main__":
    main()
