#!/usr/bin/env python3
"""Build SOREL stores and call the original binary PENetwork training function."""

import argparse
import sys
from pathlib import Path

from common import DatasetPaths, default_dataset_root, default_output_root
from sorel_data import prepare_sorel_data


ROOT = Path(__file__).resolve().parents[1]
SOREL_ROOT = ROOT / "SOREL-20M"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument("--output-root", type=Path, default=default_output_root() / "sorel")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = DatasetPaths(args.dataset_root.expanduser())
    dataset.validate()
    output = args.output_root.expanduser()
    feature_dimension = prepare_sorel_data(dataset, output / "data")

    sys.path.insert(0, str(SOREL_ROOT))
    import config

    config.device = args.device
    config.batch_size = args.batch_size
    import train

    train.device = args.device
    train.train_network(
        train_db_path=str(output / "data"),
        checkpoint_dir=str(output / "models"),
        max_epochs=args.epochs,
        use_malicious_labels=True,
        use_count_labels=False,
        use_tag_labels=False,
        feature_dimension=feature_dimension,
        random_seed=args.seed,
        workers=args.workers,
        remove_missing_features=False,
    )


if __name__ == "__main__":
    main()

