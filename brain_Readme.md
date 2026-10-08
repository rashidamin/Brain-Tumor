# 🧠 Brain Tumor Detection using Hybrid CBAM + ViT Model

This project presents a **hybrid deep learning architecture** that combines:
- CNN (EfficientNet-B0)
- Attention mechanisms (SE + CBAM)
- Vision Transformer (ViT)

for **brain tumor classification** from MRI images.

---

## 🚀 Model Overview

The proposed model (`HybridCBAMViT`) consists of:

### 1. CNN Branch
- Backbone: EfficientNet-B0
- Attention:
  - SE Block
  - CBAM (Channel + Spatial Attention)
- Output: Feature vector + feature map

### 2. Transformer Branch
- Backbone: Vision Transformer (ViT-B/16)
- Captures global dependencies

### 3. Fusion Module
- Concatenates CNN + ViT features
- Fully connected fusion head

### 4. Output Heads
- Classification Head → Tumor classes
- Uncertainty Head → Confidence estimation

---

## 📂 Project Structure
project/
│── model.py
│── train.py, test.py
│── dataset/
│── README.md
│── requirements.txt


---

## ⚙️ Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/your-repo/brain-tumor-hybrid.git
cd brain-tumor-hybrid

### step 2 create env
conda create -n brain python=3.10
conda activate brain

### install dependencies 
pip install -r requirements.txt