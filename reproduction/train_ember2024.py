#!/usr/bin/env python3
"""Prepare custom data, then execute ember2024's original LightGBM example."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from common import DatasetPaths, default_dataset_root, default_output_root
from ember_data import prepare_ember_data


ROOT = Path(__file__).resolve().parents[1]
EMBER_ROOT = ROOT / "ember2024"
EXAMPLES = EMBER_ROOT / "examples"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument("--output-root", type=Path, default=default_output_root() / "ember2024")
    parser.add_argument("--task", choices=["binary", "family"], default="binary")
    args = parser.parse_args()

    dataset = DatasetPaths(args.dataset_root.expanduser())
    dataset.validate()
    output = args.output_root.expanduser() / args.task
    prepare_ember_data(dataset, output, args.task)
    config = EXAMPLES / ("lgbm_config.json" if args.task == "binary" else "lgbm_config_family.json")
    model_path = output / "model.txt"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(EMBER_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        str(EXAMPLES / "train_lgbm.py"),
        str(output),
        str(model_path),
        "--config-file",
        str(config),
    ]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=EXAMPLES, env=env)


if __name__ == "__main__":
    main()

