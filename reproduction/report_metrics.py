#!/usr/bin/env python3
"""Print comparable metrics for every completed reproduction model."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common import default_output_root


@dataclass(frozen=True)
class ModelResult:
    name: str
    results_path: Path
    class_names_path: Path | None = None
    default_classes: tuple[str, ...] = ()


def read_class_names(model: ModelResult) -> list[str]:
    if model.class_names_path is None:
        return list(model.default_classes)
    class_data = json.loads(model.class_names_path.read_text(encoding="utf-8"))
    if isinstance(class_data, dict):
        return [
            name
            for name, _index in sorted(class_data.items(), key=lambda item: item[1])
        ]
    return list(class_data)


def calculate_metrics(confusion_matrix: np.ndarray) -> dict:
    true_positives = np.diag(confusion_matrix)
    support = confusion_matrix.sum(axis=1)
    predicted = confusion_matrix.sum(axis=0)

    precision = np.divide(
        true_positives,
        predicted,
        out=np.zeros_like(true_positives, dtype=float),
        where=predicted != 0,
    )
    recall = np.divide(
        true_positives,
        support,
        out=np.zeros_like(true_positives, dtype=float),
        where=support != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    total = confusion_matrix.sum()
    accuracy = float(true_positives.sum() / total) if total else 0.0
    return {
        "accuracy": accuracy,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    lines = ["  ".join(header.ljust(width) for header, width in zip(headers, widths))]
    lines.append("  ".join("-" * width for width in widths))
    lines.extend(
        "  ".join(value.ljust(width) for value, width in zip(row, widths))
        for row in rows
    )
    return "\n".join(lines)


def print_model_report(model: ModelResult) -> None:
    result = json.loads(model.results_path.read_text(encoding="utf-8"))
    matrix = np.asarray(result["confusion_matrix"], dtype=np.int64)
    classes = read_class_names(model)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Confusion matrix is not square: {matrix.shape}")
    if matrix.shape[0] != len(classes):
        raise ValueError(
            f"Confusion matrix has {matrix.shape[0]} classes but class mapping has "
            f"{len(classes)}: {model.name}"
        )

    metrics = calculate_metrics(matrix)
    print("\n" + "=" * 72)
    print(model.name)
    print("=" * 72)
    print(f"Accuracy        : {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall   : {metrics['macro_recall']:.4f}")
    print(f"Macro F1       : {metrics['macro_f1']:.4f}")
    print(f"Test samples   : {int(matrix.sum())}")

    rows = []
    for index, class_name in enumerate(classes):
        rows.append(
            [
                class_name,
                f"{metrics['precision'][index]:.4f}",
                f"{metrics['recall'][index]:.4f}",
                f"{metrics['f1'][index]:.4f}",
                str(int(metrics["support"][index])),
            ]
        )
    print("\nPer-class Precision/Recall/F1:")
    print(format_table(["Class", "Precision", "Recall", "F1", "Support"], rows))

    matrix_rows = [
        [f"True:{class_name}", *[str(int(value)) for value in matrix[index]]]
        for index, class_name in enumerate(classes)
    ]
    print("\nConfusion Matrix (row=true, column=predicted):")
    print(format_table(["", *[f"Pred:{name}" for name in classes]], matrix_rows))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show Accuracy, Macro Precision/Recall/F1 and confusion matrices"
    )
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    args = parser.parse_args()
    root = args.output_root.expanduser()

    binary_classes = ("Benign", "Malware")
    models = [
        ModelResult(
            "DeepMD (binary)",
            root / "deepmd" / "test_results.json",
            default_classes=binary_classes,
        ),
        ModelResult(
            "CNN (family)",
            root / "cnn" / "test_results.json",
            root / "cnn" / "class_indices.json",
        ),
        ModelResult(
            "EMBER2024 (binary)",
            root / "ember2024" / "binary" / "test_results.json",
            default_classes=binary_classes,
        ),
        ModelResult(
            "EMBER2024 (family)",
            root / "ember2024" / "family" / "test_results.json",
            root / "ember2024" / "family" / "family_names.json",
        ),
        ModelResult(
            "SOREL PENetwork (binary adaptation)",
            root / "sorel" / "test_results.json",
            default_classes=binary_classes,
        ),
        ModelResult(
            "SOREL PENetwork (family adaptation)",
            root / "sorel_family" / "test_results.json",
            root / "sorel_family" / "data" / "family_names.json",
        ),
    ]

    completed = 0
    for model in models:
        if not model.results_path.is_file():
            print(f"[SKIP] {model.name}: not found {model.results_path}")
            continue
        try:
            print_model_report(model)
            completed += 1
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            print(f"[ERROR] {model.name}: {exc}")

    if completed == 0:
        raise SystemExit("No completed test result files were found.")


if __name__ == "__main__":
    main()
