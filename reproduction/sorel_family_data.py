"""Create SOREL-compatible stores for malware-family classification."""

from __future__ import annotations

import json
import sqlite3
import zlib
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common import DatasetPaths, iter_labeled_files, sha256_file
from sorel_data import TEST_TIMESTAMP, TRAIN_TIMESTAMP


def build_family_names(paths: DatasetPaths) -> list[str]:
    families = {
        family
        for split in ("train", "test")
        for _path, family, _binary_label in iter_labeled_files(paths, split)
    }
    malware_families = sorted(family for family in families if family != "Benign")
    return ["Benign", *malware_families]


def prepare_sorel_family_data(paths: DatasetPaths, output_root: Path) -> tuple[int, list[str]]:
    """Store EMBER v3 vectors with truthful family labels, including Benign."""
    import lmdb
    import msgpack
    from thrember.features import PEFeatureExtractor

    output_root.mkdir(parents=True, exist_ok=True)
    family_names = build_family_names(paths)
    family_map = {name: index for index, name in enumerate(family_names)}
    (output_root / "family_names.json").write_text(
        json.dumps(family_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lmdb_path = output_root / "ember_features"
    environment = lmdb.open(str(lmdb_path), map_size=100 * 1024**3)
    extractor = PEFeatureExtractor()
    records = []
    failures = []
    duplicates = []
    skipped_duplicates = []
    seen_samples = {}
    with environment.begin(write=True) as transaction:
        for split in ("train", "test"):
            samples = list(iter_labeled_files(paths, split))
            for file_path, family, binary_label in tqdm(samples, desc=f"SOREL family {split}"):
                sha256 = sha256_file(file_path)
                sample = {
                    "path": str(file_path),
                    "split": split,
                    "family": family,
                    "binary_label": int(binary_label),
                }
                try:
                    vector = np.asarray(extractor.feature_vector(file_path.read_bytes()), dtype=np.float32)
                    if sha256 in seen_samples:
                        duplicate_record = {
                            "sha256": sha256,
                            "first": seen_samples[sha256],
                            "duplicate": sample,
                        }
                        if (
                            seen_samples[sha256]["split"] == sample["split"]
                            and seen_samples[sha256]["family"] == sample["family"]
                            and seen_samples[sha256]["binary_label"] == sample["binary_label"]
                        ):
                            skipped_duplicates.append(duplicate_record)
                        else:
                            duplicates.append(duplicate_record)
                        continue
                    seen_samples[sha256] = sample
                    payload = zlib.compress(msgpack.packb([vector.tolist()], use_bin_type=True))
                    transaction.put(sha256.encode("ascii"), payload)
                    timestamp = TRAIN_TIMESTAMP if split == "train" else TEST_TIMESTAMP
                    records.append((sha256, int(binary_label), family_map[family], timestamp))
                except Exception as exc:
                    failures.append({"path": str(file_path), "family": family, "error": str(exc)})
    environment.sync()
    environment.close()

    (output_root / "duplicate_sha256.json").write_text(
        json.dumps(duplicates, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "skipped_duplicate_sha256.json").write_text(
        json.dumps(skipped_duplicates, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "failed_features.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if duplicates:
        raise ValueError(
            "Duplicate SHA-256 samples found while preparing SOREL family data. "
            f"See {output_root / 'duplicate_sha256.json'}"
        )

    if not records:
        raise RuntimeError("No SOREL family features were extracted")

    connection = sqlite3.connect(output_root / "meta.db")
    cursor = connection.cursor()
    cursor.execute("DROP TABLE IF EXISTS meta")
    cursor.execute(
        "CREATE TABLE meta ("
        "sha256 TEXT PRIMARY KEY, "
        "is_malware INTEGER NOT NULL, "
        "family_label INTEGER NOT NULL, "
        "rl_fs_t REAL NOT NULL)"
    )
    cursor.executemany(
        "INSERT INTO meta (sha256, is_malware, family_label, rl_fs_t) VALUES (?, ?, ?, ?)",
        records,
    )
    connection.commit()
    connection.close()

    (output_root / "dataset_info.json").write_text(
        json.dumps(
            {
                "feature_dimension": extractor.dim,
                "feature_source": "EMBER v3 (thrember)",
                "num_families": len(family_names),
                "samples": len(records),
                "failed": len(failures),
                "duplicates": len(duplicates),
                "skipped_duplicates": len(skipped_duplicates),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        "Stored "
        f"{len(records)} samples; failed={len(failures)}; "
        f"skipped_duplicates={len(skipped_duplicates)}; "
        f"families={len(family_names)}; feature_dim={extractor.dim}"
    )
    return extractor.dim, family_names
