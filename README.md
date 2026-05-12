# Nodule Classification Model (Group 2)

## Overview
This model classifies detected lung nodules as:
- benign (0)
- malignant 
The model is based on a 3D Convolutional Neural Network (3D CNN) trained on CT nodule patches from LUNA25 dataset.

# Input
The model expects a preprocessed 3D CT nodule patch.

## Input Format
- Shape: [batch_size: 1, 64, 64, 64]
- Type: torch.float32
- Content: 3D CT nodule patch
- Preprocessing: 
    - normalized to [0,1]
    - resized/cropped to 64x64x64

# Output
- Raw output: logits
- Probability: apply sigmoid
- Interpretation:
    - closer to 0 --> likely benign 
    - closer to 1 --> likely malignant

# Threshold
Recommended threshold: 0.3

# Files provided 
- model.py --> CNN architecture 
- inference.py --> model loading and prediciton functions 
- group2_nodule_cnn_weights.pth --> trained model weights
- README.md --> usage instruction

# Installation
Required packages: pip install torch numpy

# Model Loading
from inference import load_model
model, device = load_model("group2_nodule_cnn_weights.pth")
import torch 
from model import NoduleCNN

model = NoduleCNN()
model.load_state_dict(torch.load("group2_nodule_cnn_weights.pth", map_location="cpu"))

Inference Example
with torch.no_grad():
    logits = model(x)
    probs = torch sigmoid(logits)
    probs = (probs > 0.3).int()
