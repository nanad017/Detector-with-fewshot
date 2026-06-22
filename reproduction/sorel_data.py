"""Create SOREL-compatible LMDB and SQLite stores from the custom dataset."""

from __future__ import annotations

import json
import sqlite3
import zlib
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common import DatasetPaths, iter_labeled_files, sha256_file


TRAIN_TIMESTAMP = 1540000000.0
TEST_TIMESTAMP = 1550000000.0


def prepare_sorel_data(paths: DatasetPaths, output_root: Path) -> int:
    """Store EMBER v3 vectors in SOREL's storage format with truthful binary labels."""
    import lmdb
    import msgpack
    from thrember.features import PEFeatureExtractor

    output_root.mkdir(parents=True, exist_ok=True)
    lmdb_path = output_root / "ember_features"
    environment = lmdb.open(str(lmdb_path), map_size=100 * 1024**3)
    extractor = PEFeatureExtractor()
    records = []
    failures = []
    with environment.begin(write=True) as transaction:
        for split in ("train", "test"):
            samples = list(iter_labeled_files(paths, split))
            for file_path, _family, binary_label in tqdm(samples, desc=f"SOREL {split}"):
                sha256 = sha256_file(file_path)
                try:
                    vector = np.asarray(extractor.feature_vector(file_path.read_bytes()), dtype=np.float32)
                    payload = zlib.compress(msgpack.packb([vector.tolist()], use_bin_type=True))
                    transaction.put(sha256.encode("ascii"), payload)
                    timestamp = TRAIN_TIMESTAMP if split == "train" else TEST_TIMESTAMP
                    records.append((sha256, int(binary_label), timestamp))
                except Exception as exc:
                    failures.append({"path": str(file_path), "error": str(exc)})
    environment.sync()
    environment.close()

    connection = sqlite3.connect(output_root / "meta.db")
    cursor = connection.cursor()
    cursor.execute("DROP TABLE IF EXISTS meta")
    cursor.execute(
        "CREATE TABLE meta (sha256 TEXT PRIMARY KEY, is_malware INTEGER NOT NULL, rl_fs_t REAL NOT NULL)"
    )
    cursor.executemany(
        "INSERT INTO meta (sha256, is_malware, rl_fs_t) VALUES (?, ?, ?)", records
    )
    connection.commit()
    connection.close()
    (output_root / "failed_features.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "dataset_info.json").write_text(
        json.dumps(
            {
                "feature_dimension": extractor.dim,
                "feature_source": "EMBER v3 (thrember)",
                "samples": len(records),
                "failed": len(failures),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Stored {len(records)} samples; failed={len(failures)}; feature_dim={extractor.dim}")
    return extractor.dim
