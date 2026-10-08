import os
import json
import copy
import random
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tqdm.auto import tqdm

from model import HybridCBAMViT


# =========================
# Config
# =========================
TRAIN_DIR = "/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset/Training"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.2
NUM_WORKERS = 2
SEED = 42

MODEL_PATH = "best_hybrid_cbam_vit_brain_tumor.pth"
HISTORY_PATH = "history_hybrid_cbam_vit.json"
CLASS_NAMES_PATH = "class_names.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Seed
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# =========================
# Transforms
# =========================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# =========================
# Dataset
# =========================
full_dataset = datasets.ImageFolder(TRAIN_DIR)
class_names = full_dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)
print("Total training images:", len(full_dataset))
print("Using device:", DEVICE)

with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f)

indices = np.arange(len(full_dataset))
np.random.shuffle(indices)

val_size = int(len(indices) * VAL_SPLIT)
val_indices = indices[:val_size]
train_indices = indices[val_size:]

train_dataset = copy.deepcopy(full_dataset)
val_dataset = copy.deepcopy(full_dataset)
train_dataset.transform = train_transform
val_dataset.transform = val_transform

train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(val_dataset, val_indices)

train_targets = [full_dataset.samples[i][1] for i in train_indices]
class_counts = Counter(train_targets)
class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
sample_weights = [class_weights[label] for label in train_targets]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

PIN_MEMORY = torch.cuda.is_available()

train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)


# =========================
# Model
# =========================
model = HybridCBAMViT(num_classes=num_classes).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2
)


# =========================
# Train / Eval
# =========================
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        logits = outputs["logits"]
        uncertainty = outputs["uncertainty"]

        cls_loss = criterion(logits, labels)
        unc_reg = torch.mean(torch.nn.functional.softplus(uncertainty))
        loss = cls_loss + 0.05 * unc_reg

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        batch_acc = (preds == labels).float().mean().item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.4f}")

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="weighted")
    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch, total_epochs):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Val]", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        logits = outputs["logits"]
        uncertainty = outputs["uncertainty"]

        cls_loss = criterion(logits, labels)
        unc_reg = torch.mean(torch.nn.functional.softplus(uncertainty))
        loss = cls_loss + 0.05 * unc_reg

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        batch_acc = (preds == labels).float().mean().item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.4f}")

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="weighted")
    return epoch_loss, epoch_acc, epoch_f1, all_labels, all_preds


# =========================
# Training loop
# =========================
best_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())

history = {
    "train_loss": [],
    "train_acc": [],
    "train_f1": [],
    "val_loss": [],
    "val_acc": [],
    "val_f1": []
}

epoch_bar = tqdm(range(1, EPOCHS + 1), desc="Training Progress")

for epoch in epoch_bar:
    train_loss, train_acc, train_f1 = train_one_epoch(
        model, train_loader, criterion, optimizer, DEVICE, epoch, EPOCHS
    )
    val_loss, val_acc, val_f1, y_true, y_pred = evaluate(
        model, val_loader, criterion, DEVICE, epoch, EPOCHS
    )

    scheduler.step(val_acc)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["train_f1"].append(train_f1)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)

    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "num_classes": num_classes
        }, MODEL_PATH)

    lr_now = optimizer.param_groups[0]["lr"]

    epoch_bar.set_postfix({
        "lr": f"{lr_now:.6f}",
        "train_acc": f"{train_acc:.4f}",
        "val_acc": f"{val_acc:.4f}",
        "val_f1": f"{val_f1:.4f}"
    })

    print(
        f"\nEpoch [{epoch}/{EPOCHS}] | "
        f"LR: {lr_now:.6f} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
    )

model.load_state_dict(best_model_wts)

with open(HISTORY_PATH, "w") as f:
    json.dump(history, f, indent=2)

val_loss, val_acc, val_f1, y_true, y_pred = evaluate(
    model, val_loader, criterion, DEVICE, EPOCHS, EPOCHS
)

print("\nBest Validation Accuracy:", round(best_acc, 4))
print("\nValidation Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

print("Validation Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print(f"\nBest model saved to: {MODEL_PATH}")
print(f"History saved to: {HISTORY_PATH}")