import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from src.config import NUM_CLASSES


# dense pca benchmark network
class PCAMLPNet(nn.Module):
    def __init__(self, in_features: int, hidden1: int = 128, hidden2: int = 64, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# sparse group lasso network with column-wise input shrinkage
class SparseGroupLassoMLPNet(nn.Module):
    def __init__(self, in_features: int, hidden: int = 64, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        return self.fc2(self.drop(self.relu(self.bn1(self.fc1(x)))))

    # proximal step on the first layer weights to zero out entire feature columns
    def apply_proximal_step(self, lr: float, lambda_group: float):
        with torch.no_grad():
            w = self.fc1.weight.data
            col_norms = torch.norm(w, p=2, dim=0, keepdim=True)
            threshold = lr * lambda_group
            shrinkage = torch.clamp(1.0 - threshold / torch.clamp(col_norms, min=1e-8), min=0.0)
            self.fc1.weight.data.mul_(shrinkage)


# training loop for pytorch models
def train_pytorch_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 0.003,
    lambda_group: float = 0.0,
    device: str = "cpu"
):
    model.to(device)
    model.train()

    # create dataloader
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # adam optimizer and cross entropy loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            # apply group lasso shrinkage if penalty is set
            if lambda_group > 0 and hasattr(model, "apply_proximal_step"):
                model.apply_proximal_step(lr=lr, lambda_group=lambda_group)

    return model


# generate predictions and class probabilities
def predict_pytorch_model(model: nn.Module, X_test: np.ndarray, device: str = "cpu"):
    model.to(device)
    model.eval()

    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits = model(X_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

    return preds, probs


# identify input features that were not zeroed out by the proximal penalty
def extract_active_nn_features(model: SparseGroupLassoMLPNet, feature_names: np.ndarray, threshold: float = 1e-4):
    with torch.no_grad():
        w = model.fc1.weight.data
        col_norms = torch.norm(w, p=2, dim=0).cpu().numpy()
        active_idx = np.where(col_norms > threshold)[0]
        return set(feature_names[active_idx])
