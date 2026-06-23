"""DeepMD family-classification adaptation helpers."""

from __future__ import annotations

import json
import pickle
import random
import shutil
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common import DatasetPaths, iter_labeled_files, sha256_file


ROOT = Path(__file__).resolve().parents[1]
DEEPMD_SRC = ROOT / "deep-malware-detection" / "src" / "deep_malware_detection"
sys.path.insert(0, str(DEEPMD_SRC))

from dataset import collate_fn  # noqa: E402
from models import MalConvBase, MalConvPlus  # noqa: E402


def build_family_names(paths: DatasetPaths) -> list[str]:
    malware_families = sorted(
        family_dir.name
        for malware_root in (paths.malware_train, paths.malware_test)
        for family_dir in malware_root.iterdir()
        if family_dir.is_dir()
    )
    return ["Benign", *sorted(set(malware_families))]


def prepare_deepmd_family_inputs(paths: DatasetPaths, output_root: Path) -> list[str]:
    """Copy PE files into family folders without flattening malware labels."""
    family_names = build_family_names(paths)
    family_map = {name: index for index, name in enumerate(family_names)}
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "class_names.json").write_text(
        json.dumps(family_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifests: dict[str, list[dict]] = {"train": [], "test": []}
    seen_samples: dict[str, dict] = {}
    duplicate_conflicts: list[dict] = []
    skipped_duplicates: list[dict] = []

    for split in ("train", "test"):
        for family in family_names:
            (output_root / "raw" / split / family).mkdir(parents=True, exist_ok=True)
            (output_root / "pickle" / split / family).mkdir(parents=True, exist_ok=True)

        for source, family, binary_label in iter_labeled_files(paths, split):
            sha256 = sha256_file(source)
            raw_name = f"{sha256}{source.suffix.lower()}"
            sample = {
                "sha256": sha256,
                "split": split,
                "family": family,
                "binary_label": int(binary_label),
                "source": str(source),
                "raw_name": raw_name,
                "pickle_name": f"{raw_name}.pickle",
                "label": family_map[family],
            }
            previous = seen_samples.get(sha256)
            if previous is not None:
                duplicate = {"first": previous, "duplicate": sample}
                if previous["split"] == split and previous["family"] == family:
                    skipped_duplicates.append(duplicate)
                    continue
                duplicate_conflicts.append(duplicate)
                continue

            seen_samples[sha256] = sample
            destination = output_root / "raw" / split / family / raw_name
            if not destination.exists():
                shutil.copy2(source, destination)
            manifests[split].append(sample)

    (output_root / "skipped_duplicate_sha256.json").write_text(
        json.dumps(skipped_duplicates, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "duplicate_sha256.json").write_text(
        json.dumps(duplicate_conflicts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if duplicate_conflicts:
        raise RuntimeError(
            "Duplicate SHA-256 samples found across different DeepMD family labels "
            "or splits. See duplicate_sha256.json before training."
        )

    for split, records in manifests.items():
        (output_root / f"manifest_{split}.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return family_names


class DeepMDFamilyDataset(Dataset):
    def __init__(self, output_root: Path, split: str, class_names: list[str] | None = None):
        if split not in {"train", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        self.output_root = output_root
        self.split = split
        self.class_names = class_names or json.loads(
            (output_root / "class_names.json").read_text(encoding="utf-8")
        )
        manifest_path = output_root / f"manifest_{split}.json"
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.samples: list[tuple[Path, int]] = []
        self.missing_records: list[dict] = []

        for record in records:
            pickle_path = (
                output_root
                / "pickle"
                / split
                / record["family"]
                / record["pickle_name"]
            )
            if pickle_path.is_file():
                self.samples.append((pickle_path, int(record["label"])))
            else:
                self.missing_records.append(record)

        self.labels = [label for _path, label in self.samples]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        pickle_path, label = self.samples[index]
        with pickle_path.open("rb") as file:
            features = torch.tensor(pickle.load(file))
        return features, label

    def __len__(self) -> int:
        return len(self.samples)


def make_family_loader(dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )


def write_missing_records(dataset: DeepMDFamilyDataset, output_root: Path) -> None:
    if dataset.missing_records:
        missing_path = output_root / f"missing_{dataset.split}_pickles.json"
        missing_path.write_text(
            json.dumps(dataset.missing_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"[WARN] skipped {len(dataset.missing_records)} {dataset.split} samples "
            f"without extracted PE headers; see {missing_path}",
            flush=True,
        )


def split_train_val_indices(
    labels: list[int], val_size: float, seed: int
) -> tuple[list[int], list[int]]:
    if val_size < 0 or val_size >= 1:
        raise ValueError("val_size must be in [0, 1)")

    rng = random.Random(seed)
    by_label: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        by_label.setdefault(label, []).append(index)

    train_indices: list[int] = []
    val_indices: list[int] = []
    for indices in by_label.values():
        indices = list(indices)
        rng.shuffle(indices)
        if val_size == 0 or len(indices) == 1:
            train_indices.extend(indices)
            continue
        num_val = max(1, round(len(indices) * val_size))
        num_val = min(num_val, len(indices) - 1)
        val_indices.extend(indices[:num_val])
        train_indices.extend(indices[num_val:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    if not train_indices:
        raise ValueError("No DeepMD family training samples were available")
    return train_indices, val_indices


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=num_classes)
    missing = [str(index) for index, count in enumerate(counts.tolist()) if count == 0]
    if missing:
        raise ValueError(f"No training samples for class label(s): {', '.join(missing)}")
    return counts.sum().float() / (len(counts) * counts.float())


def build_family_model(
    model_name: str,
    num_classes: int,
    embed_dim: int,
    max_len: int,
    out_channels: int,
    window_size: int,
    dropout: float,
) -> nn.Module:
    model_classes = {"MalConvBase": MalConvBase, "MalConvPlus": MalConvPlus}
    if model_name not in model_classes:
        raise ValueError(f"Unsupported DeepMD family model: {model_name}")
    model = model_classes[model_name](
        embed_dim, max_len, out_channels, window_size, dropout
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
