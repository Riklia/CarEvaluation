import torch
from torch.utils.data import Dataset, DataLoader


class CarEvaluationDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.zeros(4, dtype=torch.float32)
        y_true_idx = self.y[idx]
        y[y_true_idx] = 1

        return X, y

