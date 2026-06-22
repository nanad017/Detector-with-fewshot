#!/usr/bin/env python3
"""Evaluate trained CNN model on test set."""
import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from train_cnn import MalwareCNN


def main():
    parser = argparse.ArgumentParser(description="Evaluate CNN model")
    parser.add_argument("--model_path", required=True, help="Path to best_model.pt")
    parser.add_argument("--test_dir", required=True, help="Directory with test PNG images")
    parser.add_argument("--output_dir", required=True, help="Directory to save results")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    test_dataset = datasets.ImageFolder(root=args.test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    class_names = test_dataset.classes
    logger.info("Test set: %d images, %d classes", len(test_dataset), len(class_names))

    # Load model
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    # Infer num_classes from checkpoint
    fc3_weight = checkpoint["fc3.weight"]
    num_classes = fc3_weight.shape[0]
    logger.info("Detected %d classes from checkpoint", num_classes)

    model = MalwareCNN(num_classes).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Predict
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    for inputs, labels in tqdm(test_loader, desc="Evaluating"):
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += inputs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = correct / total
    logger.info("Test accuracy: %.4f", acc)

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    logger.info("Per-class results:\n%s", classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)

    results = {
        "accuracy": float(acc),
        "num_samples": total,
        "num_classes": num_classes,
        "class_names": class_names,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results saved to %s/results.json", args.output_dir)


if __name__ == "__main__":
    main()
