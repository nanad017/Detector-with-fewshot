#!/usr/bin/env python3
"""Evaluate the local Keras CNN on the existing external test split."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from common import default_output_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=default_output_root() / "cnn")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    output = args.output_root.expanduser()
    expected_classes = json.loads((output / "class_indices.json").read_text(encoding="utf-8"))
    datagen = ImageDataGenerator(rescale=1 / 255.0)
    test_gen = datagen.flow_from_directory(
        output / "images" / "test",
        classes=[name for name, _index in sorted(expected_classes.items(), key=lambda item: item[1])],
        target_size=(256, 256),
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    model = load_model(output / "model.keras")
    probabilities = model.predict(test_gen)
    predictions = np.argmax(probabilities, axis=1)
    names = [name for name, _index in sorted(expected_classes.items(), key=lambda item: item[1])]
    results = {
        "samples": int(len(test_gen.classes)),
        "accuracy": float(accuracy_score(test_gen.classes, predictions)),
        "classification_report": classification_report(
            test_gen.classes,
            predictions,
            labels=list(range(len(names))),
            target_names=names,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(
            test_gen.classes, predictions, labels=list(range(len(names)))
        ).tolist(),
    }
    (output / "test_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
