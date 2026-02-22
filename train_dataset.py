import os
import numpy as np
import torch
from torch.utils.data import Dataset

class FTIRDataset(Dataset):
    def __init__(self, dataset_dir):
        self.X = np.load(os.path.join(dataset_dir, "X_train.npy"))
        self.Y = np.load(os.path.join(dataset_dir, "Y_train.npy"))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        noisy = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0)
        clean = torch.tensor(self.Y[idx], dtype=torch.float32).unsqueeze(0)
        return noisy, clean