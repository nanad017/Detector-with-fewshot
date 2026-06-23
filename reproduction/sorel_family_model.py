"""SOREL PENetwork family-classification adaptation."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
SOREL_ROOT = ROOT / "SOREL-20M"
if str(SOREL_ROOT) not in sys.path:
    sys.path.insert(0, str(SOREL_ROOT))

from nets import PENetwork  # noqa: E402


SOREL_TAG_COUNT = 11


class SorelFamilyNetwork(nn.Module):
    """Reuse SOREL's feature backbone and replace the output with family logits."""

    def __init__(self, feature_dimension: int, num_families: int):
        super().__init__()
        self.sorel = PENetwork(
            use_malware=True,
            use_counts=True,
            use_tags=True,
            n_tags=SOREL_TAG_COUNT,
            feature_dimension=feature_dimension,
        )
        self.family_head = nn.Linear(128, num_families)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.family_head(self.sorel.model_base(features))


class SorelFamilyDataset(Dataset):
    def __init__(self, data_root: Path, mode: str):
        import config
        from dataset import LMDBReader, features_postproc_func

        self.features = LMDBReader(
            str(data_root / "ember_features"),
            postproc_func=features_postproc_func,
        )
        connection = sqlite3.connect(data_root / "meta.db")
        try:
            query = "SELECT sha256, family_label FROM meta"
            if mode == "train":
                query += f" WHERE rl_fs_t <= {config.train_validation_split}"
            elif mode == "validation":
                query += (
                    f" WHERE rl_fs_t >= {config.train_validation_split}"
                    f" AND rl_fs_t < {config.validation_test_split}"
                )
            elif mode == "test":
                query += f" WHERE rl_fs_t >= {config.validation_test_split}"
            else:
                raise ValueError(f"invalid mode: {mode}")
            rows = connection.execute(query).fetchall()
        finally:
            connection.close()
        self.keys = [row[0] for row in rows]
        self.labels = [int(row[1]) for row in rows]

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.features(self.keys[index])
        label = self.labels[index]
        return torch.from_numpy(feature).float(), torch.tensor(label, dtype=torch.long)


def compute_class_weights(labels: list[int], num_families: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_families)
    missing = np.flatnonzero(counts == 0)
    if len(missing):
        missing_text = ", ".join(str(int(label)) for label in missing)
        raise ValueError(f"No training samples for family label(s): {missing_text}")
    weights = counts.sum() / (num_families * counts)
    return torch.tensor(weights, dtype=torch.float32)
