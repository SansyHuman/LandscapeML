from sklearn.metrics import r2_score, root_mean_squared_error
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.nn as nn
import shap
import torch
import numpy as np


def train(loader: DataLoader, model: nn.Module, criterion: nn.Module, optimizer: Optimizer, device: torch.device, c: float=0.01):
    model.train()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        y_pred = model(x)
        loss = criterion(y_pred, y)

        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss += c * l1_norm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(loader: DataLoader, model: nn.Module, criterion: nn.Module, device: torch.device):
    model.eval()
    test_loss = 0.0
    error = 0.0
    test_cnt = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            loss = criterion(y_pred, y)

            test_loss += loss.item() * x.size(0)

            y_pred = y_pred.cpu().numpy()
            y = y.cpu().numpy()
            err = np.concatenate(np.abs((y_pred - y) / y))
            error += np.sum(err)
            test_cnt += len(err)

    return test_loss, error, test_cnt


def validate(X: torch.Tensor, y: torch.Tensor, model: nn.Module, device: torch.device, calculate_shap: bool):
    model.eval()
    y_real = None
    y_pred = None

    with torch.no_grad():
        X = X.to(device)
        y_real = y.cpu().numpy().ravel()

        outputs = model(X)
        y_pred = outputs.cpu().numpy().ravel()

    r2 = r2_score(y_real, y_pred)
    rmse = root_mean_squared_error(y_real, y_pred)
    residual = y_real - y_pred
    error = np.sum(np.abs(residual) / y_real)
    error /= len(y_real)

    print("Validation stats")
    print("=======================")
    print(f"    R2: {r2:.4f}")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    Error: {error:.4f}")

    stderr = np.std(residual)
    outlier_mask = np.abs(residual) > 3 * stderr

    shap_values = None

    if calculate_shap:
        print("Calculating SHAP values...")
        explainer = shap.DeepExplainer(model, X)
        shap_values = explainer(X)

    return y_real, y_pred, r2, rmse, float(error), shap_values, outlier_mask