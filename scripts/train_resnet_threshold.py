# scripts/train_resnet_threshold.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from train_dataset import FTIRDataset
from models.resnet_threshold import ResNetThreshold1D, apply_soft_threshold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNetThreshold1D().to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-4)
criterion = nn.MSELoss()

dataset = FTIRDataset("data/dataset")
loader = DataLoader(dataset, batch_size=8, shuffle=True)


def normalize_to_100(x):
    max_val = torch.max(x, dim=2, keepdim=True)[0] + 1e-8
    return (x / max_val) * 100


epochs = 80

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)

        noisy_norm = normalize_to_100(noisy)

        optimizer.zero_grad()

        threshold = model(noisy_norm)
        denoised = apply_soft_threshold(noisy_norm, threshold)

        loss = criterion(denoised, clean)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/resnet_threshold.pth")
print("Model saved.")