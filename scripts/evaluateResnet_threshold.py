import torch
import numpy as np
from torch.utils.data import DataLoader
from train_dataset import FTIRDataset
from models.resnet_threshold import ResNetThreshold1D, apply_soft_threshold
from metrics import compute_psnr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNetThreshold1D().to(device)
model.load_state_dict(torch.load("models/resnet_threshold.pth", map_location=device))
model.eval()

dataset = FTIRDataset("data/dataset")
loader = DataLoader(dataset, batch_size=8)


def normalize_to_100(x):
    max_val = torch.max(x, dim=2, keepdim=True)[0] + 1e-8
    return (x / max_val) * 100


with torch.no_grad():
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)

        noisy_norm = normalize_to_100(noisy)
        threshold = model(noisy_norm)
        denoised = apply_soft_threshold(noisy_norm, threshold)

        print("Example PSNR:", compute_psnr(clean.cpu().numpy()[0][0],denoised.cpu().numpy()[0][0]))
        break
