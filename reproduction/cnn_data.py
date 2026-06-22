"""PE-to-image conversion matching the original CNN notebook algorithm."""

from __future__ import annotations

from math import log
from pathlib import Path

import numpy as np
from PIL import Image

from common import DatasetPaths, iter_labeled_files, sha256_file


def pe_to_original_image(source: Path, destination: Path) -> None:
    byte_array = np.frombuffer(source.read_bytes(), dtype=np.uint8)
    complete_length = (len(byte_array) // 16) * 16
    byte_array = byte_array[:complete_length]
    if complete_length == 0:
        raise ValueError("PE has no complete 16-byte row")
    rows = byte_array.reshape(-1, 16)
    width = int((rows.shape[0] * 16) ** 0.5)
    width = 2 ** (int(log(width) / log(2)) + 1)
    height = rows.shape[0] * 16 // width
    pixels = rows[: height * width // 16].reshape(height, width)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels.astype(np.uint8)).save(destination, "PNG")


def prepare_cnn_images(paths: DatasetPaths, output_root: Path) -> dict[str, int]:
    counts = {}
    for split in ("train", "test"):
        counts[split] = 0
        for source, family, _binary_label in iter_labeled_files(paths, split):
            destination = output_root / split / family / f"{sha256_file(source)}.png"
            if not destination.exists():
                pe_to_original_image(source, destination)
            counts[split] += 1
    return counts

