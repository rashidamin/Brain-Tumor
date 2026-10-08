

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# =========================
# SE Block
# =========================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# =========================
# Channel Attention
# =========================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        hidden = max(in_planes // ratio, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, hidden, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, in_planes, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# =========================
# Spatial Attention
# =========================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


# =========================
# CBAM Block
# =========================
class CBAMBlock(nn.Module):
    def __init__(self, channels, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, ratio=ratio)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# =========================
# CNN Branch
# =========================
class CNNBranch(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        # EfficientNet feature extractor
        self.features = backbone.features
        self.se = SEBlock(1280)
        self.cbam = CBAMBlock(1280)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)          # [B,1280,7,7]
        x = self.se(x)
        x = self.cbam(x)
        feat_map = x
        x = self.pool(x).flatten(1)   # [B,1280]
        x = self.proj(x)              # [B,out_dim]
        return x, feat_map


# =========================
# ViT Branch
# =========================
class ViTBranch(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        vit_in = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Identity()

        self.proj = nn.Sequential(
            nn.Linear(vit_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.vit(x)               # [B,768]
        x = self.proj(x)              # [B,out_dim]
        return x


# =========================
# Feature Fusion
# =========================
class FusionHead(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=256):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, cnn_feat, vit_feat):
        x = torch.cat([cnn_feat, vit_feat], dim=1)
        return self.fusion(x)


# =========================
# Proposed Hybrid Model
# =========================
class HybridCBAMViT(nn.Module):
    def __init__(self, num_classes=4, feature_dim=256):
        super().__init__()
        self.cnn_branch = CNNBranch(out_dim=feature_dim)
        self.vit_branch = ViTBranch(out_dim=feature_dim)
        self.fusion_head = FusionHead(in_dim=feature_dim * 2, hidden_dim=256)

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        self.uncertainty_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        cnn_feat, feat_map = self.cnn_branch(x)
        vit_feat = self.vit_branch(x)
        fused = self.fusion_head(cnn_feat, vit_feat)

        logits = self.classifier(fused)
        uncertainty = self.uncertainty_head(fused)

        return {
            "logits": logits,
            "uncertainty": uncertainty,
            "cnn_feat": cnn_feat,
            "vit_feat": vit_feat,
            "fused_feat": fused,
            "feature_map": feat_map
        }