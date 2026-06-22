#!/usr/bin/env python3
"""
Convert PE (.exe/.dll) files to grayscale PNG images for CNN training.

Replicates the data_conversion.ipynb pipeline from malware-classification-CNN:
  1. Read .exe bytes → write .bytes hex dump (16 bytes per line)
  2. Read .bytes → reshape to 2D array → save as grayscale PNG
  3. Resize all PNGs to target size

Directory structure preserved: each subdirectory name becomes the class label.
"""
import argparse
import logging
import os
import sys
from math import log2
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def exe_to_bytes_hex(file_path: str) -> list[int]:
    """Read .exe as byte array and convert each byte to hex value (0-255)."""
    with open(file_path, "rb") as f:
        data = bytearray(f.read())
    return list(data)


def hex_to_png(hex_values: list[int], output_path: str, target_size: int = 256) -> bool:
    """
    Reshape a 1D list of byte values into a 2D grayscale image and save as PNG.

    Heuristic from the original paper: choose a width that is a power of 2
    near sqrt(total_pixels), and height = total_pixels / width.
    """
    total = len(hex_values)
    if total == 0:
        return False

    # Width fixed at 16 (from original implementation), height derived
    width = 16
    # Compute height as nearest power of 2 of (total / width)
    height_float = total / width
    height = 2 ** (int(log2(height_float)) + 1)

    # Truncate or compute usable pixels
    usable = min(total, width * height)
    arr = np.array(hex_values[:usable], dtype=np.uint8)
    # Pad if needed
    if usable < width * height:
        arr = np.pad(arr, (0, width * height - usable), mode='constant', constant_values=0)

    arr = arr.reshape((height, width))

    # Skip all-zero or all-same images (likely corrupted)
    if arr.max() == arr.min():
        return False

    img = Image.fromarray(arr, mode="L")
    img = img.resize((target_size, target_size), Image.LANCZOS)
    img.save(output_path, "PNG")
    return True


def convert_directory(input_dir: str, output_dir: str, target_size: int = 256,
                      class_name: str = None) -> tuple[int, int]:
    """
    Walk input_dir recursively. Each subdirectory = one class.
    If class_name is given, all files go to output_dir/class_name/ (flat input).
    Returns (success_count, skip_count).
    """
    success = 0
    skipped = 0

    # Flat mode: all files → output_dir/class_name/
    if class_name:
        out_root = os.path.join(output_dir, class_name)
        os.makedirs(out_root, exist_ok=True)
        files_to_process = []
        for root, dirs, files in os.walk(input_dir):
            for fname in files:
                if fname.lower().endswith(('.exe', '.dll')):
                    files_to_process.append(os.path.join(root, fname))
        for in_path in files_to_process:
            base = os.path.splitext(os.path.basename(in_path))[0]
            out_path = os.path.join(out_root, f"{base}.png")
            if os.path.exists(out_path):
                continue
            try:
                hex_vals = exe_to_bytes_hex(in_path)
                if not hex_to_png(hex_vals, out_path, target_size):
                    skipped += 1
                else:
                    success += 1
            except Exception:
                skipped += 1
        return success, skipped

    # Tree mode: subdirectories = classes
    for root, dirs, files in os.walk(input_dir):
        rel_path = os.path.relpath(root, input_dir)
        out_root = os.path.join(output_dir, rel_path)
        os.makedirs(out_root, exist_ok=True)

        for fname in files:
            if not fname.lower().endswith(('.exe', '.dll')):
                continue
            in_path = os.path.join(root, fname)
            base = os.path.splitext(fname)[0]
            out_path = os.path.join(out_root, f"{base}.png")
            if os.path.exists(out_path):
                continue
            try:
                hex_vals = exe_to_bytes_hex(in_path)
                if not hex_to_png(hex_vals, out_path, target_size):
                    skipped += 1
                else:
                    success += 1
            except Exception:
                skipped += 1

    return success, skipped


def main():
    parser = argparse.ArgumentParser(description="Convert PE files to PNG images for CNN training")
    parser.add_argument("--input_dir", required=True, help="Root directory with class subdirs or flat .exe files")
    parser.add_argument("--output_dir", required=True, help="Directory to write .png files")
    parser.add_argument("--target_size", type=int, default=256, help="PNG image size (default: 256)")
    parser.add_argument("--class_name", default=None, help="Class name for flat directories (e.g. Benign)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    success, skipped = convert_directory(args.input_dir, args.output_dir, args.target_size, args.class_name)

    logger.info("Conversion complete. %d success, %d skipped", success, skipped)
    if skipped > 0:
        logger.warning("%d files were skipped (non-PE, corrupted, or empty)", skipped)


if __name__ == "__main__":
    main()
