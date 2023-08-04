import torch
from torch import nn
from sklearn.metrics import f1_score, accuracy_score

metrics = {
    'f1_score_macro': lambda y_true, y_pred: f1_score(y_true, y_pred,
                                                 average='macro'),
}


class LinearModel(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128, dropout=0.5):
        super().__init__()

        self.features = torch.nn.Sequential(

            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.Dropout(dropout),
            torch.nn.ReLU(),

            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Dropout(dropout),
            torch.nn.ReLU(),

            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, out_dim),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.features(x)
        return output

    def compute_l1_loss(self, w):
        return torch.abs(w).sum()

    def compute_l2_loss(self, w):
        return torch.square(w).sum()
