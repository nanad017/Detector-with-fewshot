#!/usr/bin/env python3
"""
Train CNN model for malware family classification.

Architecture (from malware-classification-CNN):
  Conv2D(64,3) → MaxPool → Conv2D(32,3) → MaxPool → Conv2D(32,3) → MaxPool
  → Conv2D(16,3) → MaxPool → Dropout → Flatten → Dense(128) → Dropout
  → Dense(50) → Dropout → Dense(num_classes, softmax)

Input: 256x256 grayscale PNG images organized in class subdirectories.
"""
import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class MalwareCNN(nn.Module):
    """CNN matching the malware-classification-CNN architecture."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)   # grayscale input
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3)
        self.dropout1 = nn.Dropout(0.25)

        # Compute FC input size dynamically
        self._fc_in = None
        self._compute_fc_in(1, 256)  # channels=1, image_size=256

        self.fc1 = nn.Linear(self._fc_in, 128)
        self.dropout2 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 50)
        self.dropout3 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(50, num_classes)

    def _compute_fc_in(self, channels: int, size: int):
        x = torch.zeros(1, channels, size, size)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        self._fc_in = x.view(1, -1).size(1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout3(x)
        x = self.fc3(x)
        return x


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for inputs, labels in tqdm(loader, desc="Train", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += inputs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    for inputs, labels in tqdm(loader, desc="Eval", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += inputs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


def get_class_weights(dataset, num_classes, device):
    """Compute balanced class weights for imbalanced dataset."""
    labels = [dataset.samples[i][1] for i in range(len(dataset))]
    counts = np.bincount(labels, minlength=num_classes)
    weights = 1.0 / (counts + 1)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32).to(device)


def main():
    parser = argparse.ArgumentParser(description="Train CNN for malware family classification")
    parser.add_argument("--train_dir", required=True, help="Directory with class subdirs of PNG images (train)")
    parser.add_argument("--test_dir", required=True, help="Directory with class subdirs of PNG images (test)")
    parser.add_argument("--output_dir", required=True, help="Directory to save model and results")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    os.makedirs(args.output_dir, exist_ok=True)

    # Data transforms
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(root=args.train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=args.test_dir, transform=transform)

    num_classes = len(train_dataset.classes)
    logger.info("Train: %d images, %d classes: %s", len(train_dataset), num_classes, train_dataset.classes)
    logger.info("Test: %d images", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = MalwareCNN(num_classes).to(device)
    logger.info("Model params: %d", sum(p.numel() for p in model.parameters()))

    class_weights = get_class_weights(train_dataset, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    best_acc = 0
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step(test_loss)

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["test_loss"].append(float(test_loss))
        history["test_acc"].append(float(test_acc))

        logger.info("Epoch %d/%d — Train loss: %.4f, acc: %.4f | Test loss: %.4f, acc: %.4f",
                    epoch, args.epochs, train_loss, train_acc, test_loss, test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            logger.info("  -> Saved best model (acc=%.4f)", best_acc)

    # Save history
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Training complete. Best test accuracy: %.4f", best_acc)


if __name__ == "__main__":
    main()
