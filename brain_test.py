import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    precision_score, recall_score,
)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import cv2
from PIL import Image
import timm


# ═══════════════════════════════════════════════
#  Model – must match train_optimized.py exactly
# ═══════════════════════════════════════════════
class BrainTumorModel(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "efficientnet_b3"):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=False, num_classes=0, global_pool="avg")
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# =========================
# Config
# =========================
TEST_DIR   = "/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset/Testing"
IMG_SIZE   = 224
BATCH_SIZE = 32
NUM_WORKERS = 2
MODEL_PATH  = "/kaggle/working/best_brain_tumor_model.pth"
OUTPUT_DIR  = "eval_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
           "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]

print(f"Device : {DEVICE}")


# =========================
# Transform
# =========================
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# =========================
# Dataset
# =========================
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
test_loader  = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)
print(f"Total test images : {len(test_dataset)}")


# =========================
# Load checkpoint
# ── torch.compile() saves keys with "_orig_mod." prefix.
# ── We strip it so the plain model can load the weights.
# =========================
checkpoint  = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
class_names = checkpoint["class_names"]
num_classes = checkpoint["num_classes"]
print(f"Classes : {class_names}")

raw_sd = checkpoint["model_state_dict"]

# Strip "_orig_mod." prefix produced by torch.compile()
clean_sd = {}
for k, v in raw_sd.items():
    new_key = k.replace("_orig_mod.", "", 1)
    clean_sd[new_key] = v

model = BrainTumorModel(num_classes=num_classes).to(DEVICE)
missing, unexpected = model.load_state_dict(clean_sd, strict=True)
print(f"Missing keys   : {missing}")
print(f"Unexpected keys: {unexpected}")
model.eval()
print("Model loaded successfully ✓")


# =========================
# Inference
# =========================
all_labels, all_preds, all_probs = [], [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        logits = model(images)
        probs  = F.softmax(logits, dim=1)
        preds  = logits.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_labels = np.array(all_labels)
all_preds  = np.array(all_preds)
all_probs  = np.array(all_probs)

# Confidence = max softmax probability (used as proxy for uncertainty)
all_confidence  = all_probs.max(axis=1)
all_uncertainty = 1.0 - all_confidence   # higher = more uncertain

acc     = accuracy_score(all_labels, all_preds)
wf1     = f1_score(all_labels, all_preds, average="weighted")
avg_unc = float(all_uncertainty.mean())

print(f"\nTest Accuracy        : {acc:.4f}")
print(f"Weighted F1-score    : {wf1:.4f}")
print(f"Avg Uncertainty Score: {avg_unc:.4f}")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))


# ═══════════════════════════════════════════════
#  PLOT 1 – Confusion Matrix (raw + normalised)
# ═══════════════════════════════════════════════
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cmap    = LinearSegmentedColormap.from_list("cb", ["#f0f4ff", "#1a3a6b"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Confusion Matrix", fontsize=16, fontweight="bold", y=1.01)

    for ax, data, title, fmt in zip(
        axes, [cm, cm_norm], ["Raw Counts", "Normalised (Row %)"], ["d", ".2%"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap,
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.5, linecolor="white",
                    annot_kws={"size": 11, "weight": "bold"},
                    ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title(title, fontsize=13, pad=12)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True",      fontsize=11)
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")

plot_confusion_matrix(all_labels, all_preds, class_names,
                      os.path.join(OUTPUT_DIR, "confusion_matrix.png"))


# ═══════════════════════════════════════════════
#  PLOT 2 – Per-class metrics bar chart
# ═══════════════════════════════════════════════
def plot_per_class_metrics(y_true, y_pred, class_names, save_path):
    prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec  = recall_score   (y_true, y_pred, average=None, zero_division=0)
    f1   = f1_score       (y_true, y_pred, average=None, zero_division=0)
    x    = np.arange(len(class_names)); w = 0.26

    fig, ax = plt.subplots(figsize=(12, 6))
    for offset, vals, label, color in [
        (-w, prec, "Precision", "#4C72B0"),
        ( 0, rec,  "Recall",    "#55A868"),
        ( w, f1,   "F1-Score",  "#C44E52"),
    ]:
        bars = ax.bar(x + offset, vals, w, label=label, color=color, edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels(class_names, fontsize=11)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Precision / Recall / F1", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11); ax.yaxis.grid(True, linestyle="--", alpha=0.6); ax.set_axisbelow(True)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved → {save_path}")

plot_per_class_metrics(all_labels, all_preds, class_names,
                       os.path.join(OUTPUT_DIR, "per_class_metrics.png"))


# ═══════════════════════════════════════════════
#  PLOT 3 – ROC curves
# ═══════════════════════════════════════════════
def plot_roc_curves(y_true, y_probs, class_names, save_path):
    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(9, 7))

    for i, (cls, col) in enumerate(zip(class_names, PALETTE)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        ax.plot(fpr, tpr, lw=2, color=col, label=f"{cls}  (AUC={auc(fpr,tpr):.3f})")

    fpr_m, tpr_m, _ = roc_curve(y_bin.ravel(), y_probs.ravel())
    ax.plot(fpr_m, tpr_m, lw=2.5, color="black", linestyle="--",
            label=f"Macro avg (AUC={auc(fpr_m,tpr_m):.3f})")
    ax.plot([0,1],[0,1],"k:",lw=1)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10); ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved → {save_path}")

plot_roc_curves(all_labels, all_probs, class_names,
                os.path.join(OUTPUT_DIR, "roc_curves.png"))


# ═══════════════════════════════════════════════
#  PLOT 4 – Precision-Recall curves
# ═══════════════════════════════════════════════
def plot_pr_curves(y_true, y_probs, class_names, save_path):
    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(9, 7))

    for i, (cls, col) in enumerate(zip(class_names, PALETTE)):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
        ap = average_precision_score(y_bin[:, i], y_probs[:, i])
        ax.plot(rec, prec, lw=2, color=col, label=f"{cls}  (AP={ap:.3f})")

    ax.set_xlabel("Recall", fontsize=12); ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02]); ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved → {save_path}")

plot_pr_curves(all_labels, all_probs, class_names,
               os.path.join(OUTPUT_DIR, "pr_curves.png"))


# ═══════════════════════════════════════════════
#  PLOT 5 – Confidence distribution + calibration
# ═══════════════════════════════════════════════
def plot_confidence_distribution(y_true, y_probs, y_preds, save_path):
    confidence = y_probs.max(axis=1)
    correct    = (y_true == y_preds)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(confidence[correct],  bins=30, alpha=0.7, color="#55A868", label="Correct",   edgecolor="white")
    ax.hist(confidence[~correct], bins=30, alpha=0.7, color="#C44E52", label="Incorrect", edgecolor="white")
    ax.set_xlabel("Confidence (max softmax prob)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Confidence: Correct vs Incorrect", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11); ax.grid(True, linestyle="--", alpha=0.4)

    ax = axes[1]
    bins_ = np.linspace(0, 1, 11)
    bin_acc, bin_conf = [], []
    for lo, hi in zip(bins_[:-1], bins_[1:]):
        mask = (confidence >= lo) & (confidence < hi)
        if mask.sum() > 0:
            bin_acc.append(correct[mask].mean())
            bin_conf.append(confidence[mask].mean())
    ax.bar(bin_conf, bin_acc, width=0.08, alpha=0.7, color="#4C72B0",
           edgecolor="white", label="Actual accuracy")
    ax.plot([0,1],[0,1],"k--",lw=1.5, label="Perfect calibration")
    ax.set_xlabel("Mean Confidence", fontsize=12); ax.set_ylabel("Fraction Correct", fontsize=12)
    ax.set_title("Reliability Diagram (Calibration)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11); ax.set_xlim([0,1]); ax.set_ylim([0,1]); ax.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Confidence Analysis", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved → {save_path}")

plot_confidence_distribution(all_labels, all_probs, all_preds,
                             os.path.join(OUTPUT_DIR, "confidence_distribution.png"))


# ═══════════════════════════════════════════════
#  PLOT 6 – Uncertainty distribution (1 - confidence)
# ═══════════════════════════════════════════════
def plot_uncertainty(y_true, y_preds, uncertainty, class_names, save_path):
    correct = (y_true == y_preds)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(uncertainty[correct],  bins=30, alpha=0.7, color="#55A868", label="Correct",   edgecolor="white")
    ax.hist(uncertainty[~correct], bins=30, alpha=0.7, color="#C44E52", label="Incorrect", edgecolor="white")
    ax.set_xlabel("Uncertainty (1 − confidence)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Uncertainty: Correct vs Incorrect", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11); ax.grid(True, linestyle="--", alpha=0.4)

    ax = axes[1]
    data_by_class = [uncertainty[y_true == i] for i in range(len(class_names))]
    bp = ax.boxplot(data_by_class, labels=class_names, patch_artist=True)
    for patch, col in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    ax.set_xlabel("Class", fontsize=12); ax.set_ylabel("Uncertainty", fontsize=12)
    ax.set_title("Uncertainty per Class", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    plt.suptitle("Uncertainty Analysis", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved → {save_path}")

plot_uncertainty(all_labels, all_preds, all_uncertainty, class_names,
                 os.path.join(OUTPUT_DIR, "uncertainty_analysis.png"))


# ═══════════════════════════════════════════════
#  PLOT 7 – Softmax probability heatmap
# ═══════════════════════════════════════════════
def plot_probability_heatmap(y_true, y_probs, class_names, save_path, n_samples=80):
    idx_list = []
    per_cls  = n_samples // len(class_names)
    for c in range(len(class_names)):
        idx_list.extend(np.where(y_true == c)[0][:per_cls].tolist())
    idx_list = np.array(idx_list)

    probs_sub  = y_probs[idx_list]
    labels_sub = y_true[idx_list]
    sort_key   = np.lexsort((probs_sub.max(axis=1), labels_sub))
    probs_sorted  = probs_sub[sort_key]
    labels_sorted = labels_sub[sort_key]

    fig, ax = plt.subplots(figsize=(10, max(6, len(idx_list) // 6)))
    im = ax.imshow(probs_sorted, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Predicted probability")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=10)
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel(f"Sample (n={len(idx_list)}, sorted by true label)", fontsize=11)
    ax.set_title("Softmax Probability Heatmap", fontsize=14, fontweight="bold")

    boundary = 0
    for c in range(len(class_names)):
        boundary += int(np.sum(labels_sorted == c))
        ax.axhline(boundary - 0.5, color="white", lw=1.5)

    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved → {save_path}")

plot_probability_heatmap(all_labels, all_probs, class_names,
                         os.path.join(OUTPUT_DIR, "probability_heatmap.png"))


# ═══════════════════════════════════════════════
#  PLOT 8 – Grad-CAM
# ═══════════════════════════════════════════════
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(
            lambda _, __, out: setattr(self, "activations", out.detach()))
        target_layer.register_full_backward_hook(
            lambda _, __, grad: setattr(self, "gradients", grad[0].detach()))

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        # BrainTumorModel returns a plain tensor
        logits = self.model(input_tensor)
        if isinstance(logits, dict):
            logits = logits["logits"]
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam     = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam     = cam.squeeze().cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def plot_gradcam(model, dataset, class_names, device, save_path, n_per_class=2):
    # For EfficientNet-B3 target the last conv before global pool
    target_layer = None
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            target_layer = m          # keeps updating → ends up as last Conv2d
    if target_layer is None:
        print("No Conv2d found – skipping Grad-CAM.")
        return

    gradcam = GradCAM(model, target_layer)

    class_indices = {c: [] for c in range(len(class_names))}
    for idx, (_, lbl) in enumerate(dataset.samples):
        class_indices[lbl].append(idx)

    fig, axes = plt.subplots(
        len(class_names), n_per_class * 3,
        figsize=(n_per_class * 9, len(class_names) * 3.2)
    )
    if len(class_names) == 1:
        axes = [axes]

    for row, cls_idx in enumerate(range(len(class_names))):
        for col, sample_idx in enumerate(class_indices[cls_idx][:n_per_class]):
            img_path = dataset.samples[sample_idx][0]
            raw_img  = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            raw_np   = np.array(raw_img)
            tensor   = test_transform(raw_img).unsqueeze(0).to(device)

            cam, pred_idx = gradcam.generate(tensor)
            heatmap = cv2.cvtColor(
                cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET),
                cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(raw_np, 0.55, heatmap, 0.45, 0)

            bc = col * 3
            axes[row][bc].imshow(raw_np)
            axes[row][bc].set_title(f"Original\nTrue: {class_names[cls_idx]}", fontsize=8)
            axes[row][bc+1].imshow(cam, cmap="jet")
            axes[row][bc+1].set_title(f"Grad-CAM\nPred: {class_names[pred_idx]}", fontsize=8)
            axes[row][bc+2].imshow(overlay)
            axes[row][bc+2].set_title("Overlay", fontsize=8)

        for ax in axes[row]:
            ax.axis("off")

    plt.suptitle("Grad-CAM Visualisation", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout(); plt.savefig(save_path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"Saved → {save_path}")

plot_gradcam(model, test_dataset, class_names, DEVICE,
             os.path.join(OUTPUT_DIR, "gradcam.png"), n_per_class=2)


# ═══════════════════════════════════════════════
#  PLOT 9 – Misclassified samples gallery
# ═══════════════════════════════════════════════
def plot_misclassified(dataset, y_true, y_preds, y_probs, class_names, save_path, n=16):
    wrong_idx = np.where(y_true != y_preds)[0][:n]
    if len(wrong_idx) == 0:
        print("No misclassified samples – skipping.")
        return

    cols = 4; rows = int(np.ceil(len(wrong_idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.8))
    axes = axes.flatten()

    for ax, idx in zip(axes, wrong_idx):
        img = Image.open(dataset.samples[idx][0]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        ax.imshow(img)
        ax.set_title(
            f"True : {class_names[y_true[idx]]}\n"
            f"Pred : {class_names[y_preds[idx]]} ({y_probs[idx, y_preds[idx]]:.1%})",
            fontsize=8, color="#C44E52")
        ax.axis("off")

    for ax in axes[len(wrong_idx):]:
        ax.axis("off")

    plt.suptitle("Misclassified Samples", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(save_path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"Saved → {save_path}")

plot_misclassified(test_dataset, all_labels, all_preds, all_probs, class_names,
                   os.path.join(OUTPUT_DIR, "misclassified_samples.png"))


# ═══════════════════════════════════════════════
#  PLOT 10 – Summary dashboard
# ═══════════════════════════════════════════════
def plot_summary_dashboard(y_true, y_preds, y_probs, class_names,
                           acc, wf1, avg_unc, save_path):
    f1   = f1_score(y_true, y_preds, average=None, zero_division=0)
    fig  = plt.figure(figsize=(18, 10))
    gs   = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # KPI panel
    ax_kpi = fig.add_subplot(gs[0, 0]); ax_kpi.axis("off")
    kpis   = [("Accuracy",        f"{acc:.4f}",     "#4C72B0"),
              ("Weighted F1",     f"{wf1:.4f}",     "#55A868"),
              ("Avg Uncertainty", f"{avg_unc:.4f}", "#DD8452"),
              ("Test Samples",    str(len(y_true)), "#8172B3")]
    y_pos = 0.85
    for label, value, color in kpis:
        ax_kpi.text(0.05, y_pos,        label, transform=ax_kpi.transAxes, fontsize=11, color="grey")
        ax_kpi.text(0.05, y_pos - 0.09, value, transform=ax_kpi.transAxes,
                    fontsize=22, fontweight="bold", color=color)
        y_pos -= 0.22
    ax_kpi.set_title("Key Metrics", fontsize=12, fontweight="bold")

    # Confusion matrix
    ax_cm   = fig.add_subplot(gs[0, 1:])
    cm_norm = confusion_matrix(y_true, y_preds).astype(float)
    cm_norm /= cm_norm.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("cb", ["#f0f4ff", "#1a3a6b"])
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap=cmap,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor="white", annot_kws={"size": 10},
                ax=ax_cm, cbar=False)
    ax_cm.set_title("Normalised Confusion Matrix", fontsize=12, fontweight="bold")
    ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("True")
    ax_cm.tick_params(axis="x", rotation=30)

    # F1 per class
    ax_f1 = fig.add_subplot(gs[1, 0])
    bars  = ax_f1.barh(class_names, f1, color=PALETTE[:len(class_names)], edgecolor="white")
    ax_f1.set_xlim(0, 1.1)
    for bar, val in zip(bars, f1):
        ax_f1.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                   f"{val:.3f}", va="center", fontsize=9)
    ax_f1.set_title("F1-Score per Class", fontsize=12, fontweight="bold")
    ax_f1.grid(axis="x", linestyle="--", alpha=0.5)

    # Confidence histogram
    ax_conf = fig.add_subplot(gs[1, 1])
    conf    = y_probs.max(axis=1); correct = (y_true == y_preds)
    ax_conf.hist(conf[correct],  bins=25, alpha=0.75, color="#55A868", label="Correct",   edgecolor="white")
    ax_conf.hist(conf[~correct], bins=25, alpha=0.75, color="#C44E52", label="Incorrect", edgecolor="white")
    ax_conf.set_xlabel("Confidence"); ax_conf.set_ylabel("Count")
    ax_conf.set_title("Confidence Distribution", fontsize=12, fontweight="bold")
    ax_conf.legend(fontsize=9); ax_conf.grid(linestyle="--", alpha=0.4)

    # ROC
    ax_roc = fig.add_subplot(gs[1, 2])
    y_bin  = label_binarize(y_true, classes=list(range(len(class_names))))
    for i, (cls, col) in enumerate(zip(class_names, PALETTE)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        ax_roc.plot(fpr, tpr, lw=1.8, color=col, label=f"{cls} ({auc(fpr,tpr):.2f})")
    ax_roc.plot([0,1],[0,1],"k:",lw=1)
    ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
    ax_roc.set_title("ROC Curves", fontsize=12, fontweight="bold")
    ax_roc.legend(fontsize=8, loc="lower right"); ax_roc.grid(linestyle="--", alpha=0.4)

    fig.suptitle("Brain Tumour MRI – Evaluation Dashboard",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved → {save_path}")

plot_summary_dashboard(all_labels, all_preds, all_probs, class_names,
                       acc, wf1, avg_unc,
                       os.path.join(OUTPUT_DIR, "summary_dashboard.png"))


print(f"\n{'='*55}")
print(f"  All plots saved to: ./{OUTPUT_DIR}/")
print(f"{'='*55}")
for i, name in enumerate([
    "confusion_matrix.png", "per_class_metrics.png", "roc_curves.png",
    "pr_curves.png", "confidence_distribution.png", "uncertainty_analysis.png",
    "probability_heatmap.png", "gradcam.png", "misclassified_samples.png",
    "summary_dashboard.png",
], 1):
    print(f"  {i:2d}. {name}")
print(f"{'='*55}")