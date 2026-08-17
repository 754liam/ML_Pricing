import numpy as np
import torch
import torch.nn as nn


class NeuralNetModel:
    def __init__(self, hidden=32, epochs=10, lr=1e-3):
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X = torch.tensor(np.asarray(X_train), dtype=torch.float32, device=self.device)
        y = torch.tensor(np.asarray(y_train), dtype=torch.float32, device=self.device)
        self.model = nn.Sequential(
            nn.Linear(X.shape[1], self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
        ).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss = nn.MSELoss()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss(self.model(X).squeeze(), y).backward()
            opt.step()

    def predict(self, X):
        X = torch.tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.model(X).squeeze().cpu().numpy()

# 32 formulas -> 32 scores -> 32 weights to be applied to the scores

# Adam is a nudger for the weights of the formulas for 10 epochs