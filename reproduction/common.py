"""Shared paths and data preparation for reproduction scripts."""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class DatasetPaths:
    root: Path

    @property
    def malware_train(self) -> Path:
        return self.root / "Virus" / "Virus train"

    @property
    def malware_test(self) -> Path:
        return self.root / "Virus" / "Virus test"

    @property
    def benign_train(self) -> Path:
        return self.root / "Benign" / "Benign train"

    @property
    def benign_test(self) -> Path:
        return self.root / "Benign" / "Benign test"

    def validate(self) -> None:
        missing = [
            path
            for path in (
                self.malware_train,
                self.malware_test,
                self.benign_train,
                self.benign_test,
            )
            if not path.is_dir()
        ]
        if missing:
            formatted = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(f"Missing dataset directories:\n{formatted}")


def default_dataset_root() -> Path:
    return Path(os.environ.get("SD_FSPROTO_DATASET", "~/An_solo/dataset")).expanduser()


def default_output_root() -> Path:
    return Path(os.environ.get("SD_FSPROTO_OUTPUT", "~/An_solo/detector/reproduction_output")).expanduser()


def iter_pe_files(directory: Path) -> Iterator[Path]:
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in {".exe", ".dll"}:
            yield path


def iter_labeled_files(paths: DatasetPaths, split: str) -> Iterator[tuple[Path, str, int]]:
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    malware_root = paths.malware_train if split == "train" else paths.malware_test
    benign_root = paths.benign_train if split == "train" else paths.benign_test
    for family_dir in sorted(malware_root.iterdir()):
        if family_dir.is_dir():
            for file_path in iter_pe_files(family_dir):
                yield file_path, family_dir.name, 1
    for file_path in iter_pe_files(benign_root):
        yield file_path, "Benign", 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_deepmd_inputs(paths: DatasetPaths, output_root: Path) -> None:
    """Flatten the custom dataset without allowing same-name samples to collide."""
    for split in ("train", "test"):
        for label in ("benign", "malware"):
            (output_root / "raw" / split / label).mkdir(parents=True, exist_ok=True)
        for source, _family, binary_label in iter_labeled_files(paths, split):
            label = "malware" if binary_label else "benign"
            destination = output_root / "raw" / split / label / f"{sha256_file(source)}{source.suffix.lower()}"
            if not destination.exists():
                shutil.copy2(source, destination)

