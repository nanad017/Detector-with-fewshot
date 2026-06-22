#!/usr/bin/env python3
"""Local Keras conversion of the original combined-classifier notebook."""

import argparse
import json
from pathlib import Path

import numpy as np

from common import DatasetPaths, default_dataset_root, default_output_root
from cnn_data import prepare_cnn_images


def build_original_model(num_classes: int):
    from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
    from tensorflow.keras.models import Sequential

    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", input_shape=(256, 256, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
        weighted_metrics=["accuracy"],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument("--output-root", type=Path, default=default_output_root() / "cnn")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    dataset = DatasetPaths(args.dataset_root.expanduser())
    dataset.validate()
    output = args.output_root.expanduser()
    image_root = output / "images"
    print("Prepared:", prepare_cnn_images(dataset, image_root))

    datagen = ImageDataGenerator(rescale=1 / 255.0)
    train_gen = datagen.flow_from_directory(
        image_root / "train",
        target_size=(256, 256),
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=42,
    )
    labels = train_gen.classes
    weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(labels), y=labels
    )
    class_weights = dict(zip(np.unique(labels), weights))
    model = build_original_model(len(train_gen.class_indices))
    model.summary()
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        class_weight=class_weights,
    )
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / "model.keras")
    (output / "class_indices.json").write_text(
        json.dumps(train_gen.class_indices, indent=2), encoding="utf-8"
    )
    (output / "history.json").write_text(
        json.dumps({key: [float(value) for value in values] for key, values in history.history.items()}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

