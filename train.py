import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import SpectraDataset
from model import ResNetThreshold
from utils import apply_threshold

num_samples = 500
spectrum_length = 1024

clean_data = np.random.rand(num_samples, spectrum_length)
noise = np.random.normal(0, 0.05, clean_data.shape)
noisy_data = clean_data + noise

dataset = SpectraDataset(noisy_data, clean_data)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetThreshold().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

epochs = 20
for epoch in range(epochs):
    total_loss = 0

    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)

        optimizer.zero_grad()

        threshold = model(noisy)
        denoised = apply_threshold(noisy, threshold)

        loss = criterion(denoised, clean)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.6f}")

torch.save(model.state_dict(), "resnet_threshold_model.pth")