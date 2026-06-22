"""Build EMBER2024 memmaps from the user's already-split PE dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common import DatasetPaths, iter_labeled_files


def prepare_ember_data(paths: DatasetPaths, output_root: Path, task: str = "binary") -> None:
    from thrember.features import PEFeatureExtractor

    if task not in {"binary", "family"}:
        raise ValueError("task must be binary or family")
    extractor = PEFeatureExtractor()
    family_names = sorted(
        {family for split in ("train", "test") for _path, family, _label in iter_labeled_files(paths, split)}
    )
    family_map = {name: index for index, name in enumerate(family_names)}
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "family_names.json").write_text(
        json.dumps(family_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for split in ("train", "test"):
        rows = []
        labels = []
        failures = []
        samples = list(iter_labeled_files(paths, split))
        for file_path, family, binary_label in tqdm(samples, desc=f"EMBER {split}"):
            try:
                rows.append(extractor.feature_vector(file_path.read_bytes()))
                labels.append(binary_label if task == "binary" else family_map[family])
            except Exception as exc:
                failures.append({"path": str(file_path), "error": str(exc)})
        if not rows:
            raise RuntimeError(f"No features extracted for {split}")
        X = np.asarray(rows, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int32)
        X_memmap = np.memmap(output_root / f"X_{split}.dat", dtype=np.float32, mode="w+", shape=X.shape)
        X_memmap[:] = X
        X_memmap.flush()
        y_memmap = np.memmap(output_root / f"y_{split}.dat", dtype=np.int32, mode="w+", shape=y.shape)
        y_memmap[:] = y
        y_memmap.flush()
        del X_memmap, y_memmap
        (output_root / f"failed_{split}.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"{split}: X={X.shape}, y={y.shape}, failed={len(failures)}")

