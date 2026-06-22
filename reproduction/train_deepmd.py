#!/usr/bin/env python3
"""Prepare custom PE data and call deep-malware-detection's original trainer."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from common import DatasetPaths, copy_deepmd_inputs, default_dataset_root, default_output_root


ROOT = Path(__file__).resolve().parents[1]
DEEPMD_ROOT = ROOT / "deep-malware-detection"
DEEPMD_TRAIN = DEEPMD_ROOT / "src" / "deep_malware_detection"


def run(command: list[str], **kwargs) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument("--output-root", type=Path, default=default_output_root() / "deepmd")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default="MalConvPlus")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="MalConvPlus")
    args = parser.parse_args()

    dataset = DatasetPaths(args.dataset_root.expanduser())
    dataset.validate()
    output = args.output_root.expanduser()
    copy_deepmd_inputs(dataset, output)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(DEEPMD_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    for split in ("train", "test"):
        for label in ("benign", "malware"):
            run(
                [
                    sys.executable,
                    "-m",
                    "src.bin.extract_header",
                    "--input_dir",
                    str(output / "raw" / split / label),
                    "--output_dir",
                    str(output / "pickle" / split / label),
                ],
                cwd=DEEPMD_ROOT,
                env=env,
            )

    run(
        [
            sys.executable,
            "train.py",
            "--device",
            args.device,
            "--model",
            args.model,
            "--benign_dir",
            str(output / "pickle" / "train" / "benign"),
            "--malware_dir",
            str(output / "pickle" / "train" / "malware"),
            "--checkpoint_dir",
            str(output / "models"),
            "--tag",
            args.tag,
            "--batch_size",
            str(args.batch_size),
            "--val_size",
            str(args.val_size),
            "--test_size",
            str(args.test_size),
            "--seed",
            str(args.seed),
        ],
        cwd=DEEPMD_TRAIN,
        env=env,
    )


if __name__ == "__main__":
    main()

