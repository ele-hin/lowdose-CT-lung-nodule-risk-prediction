import torch
import numpy as np
from pathlib import Path
from model import NoduleCNN

THRESHOLD = 0.3

def load_model(weights_path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    device = torch.device(device)
    
    model = NoduleCNN().to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, device

def predict_patch(model, x, device, threshold=THRESHOLD):
    
    x = x.to(device)
    
    with torch.no_grad():
        logits = model(x)
        probability = torch.sigmoid(logits)
        prediciton = (probability > threshold).int()
        
    return {
        "malignancy_probability": float(probability.cpu().item()),
        "predicted_label": int(prediciton.cpu().item())
    }