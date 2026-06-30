from sklearn.metrics import r2_score, root_mean_squared_error
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.nn as nn
import shap
import torch
import numpy as np

from common.utils import FullyConnectedNetwork


class SCIAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super(SCIAutoencoder, self).__init__()

        self.encoder = FullyConnectedNetwork(
            input_dim, latent_dim,
            (input_dim, nn.GELU()),
            (input_dim // 2, nn.GELU()),
            (input_dim // 2, nn.GELU()),
            (input_dim // 4, nn.GELU()),
            (input_dim // 4, nn.GELU()),
            (input_dim // 8, nn.GELU()),
            (input_dim // 8, nn.GELU()),
            (input_dim // 16, nn.GELU()),
            (input_dim // 16, nn.GELU()),
            (input_dim // 16, nn.GELU()),
            (input_dim // 32, nn.GELU()),
            (input_dim // 32, nn.GELU()),
            (input_dim // 32, nn.GELU()),
            (input_dim // 32, nn.GELU())
        )

        self.regressor = nn.Linear(latent_dim, 2)

        self.decoder = FullyConnectedNetwork(
            latent_dim, input_dim,
            (input_dim // 32, nn.GELU()),
            (input_dim // 32, nn.GELU()),
            (input_dim // 32, nn.GELU()),
            (input_dim // 32, nn.GELU()),
            (input_dim // 16, nn.GELU()),
            (input_dim // 16, nn.GELU()),
            (input_dim // 16, nn.GELU()),
            (input_dim // 8, nn.GELU()),
            (input_dim // 8, nn.GELU()),
            (input_dim // 4, nn.GELU()),
            (input_dim // 4, nn.GELU()),
            (input_dim // 2, nn.GELU()),
            (input_dim // 2, nn.GELU()),
            (input_dim, nn.GELU()),
        )

    def forward_internal(self, x: torch.Tensor):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        charges = self.regressor(z)
        return x_recon, charges, z

    # For SHAP value calculation
    def forward(self, x: torch.Tensor):
        x_recon, charges, z = self.forward_internal(x)
        return charges


def train(loader: DataLoader, model: SCIAutoencoder, recon_loss_fn: nn.Module, charge_loss_fn: nn.Module,
          optimizer: Optimizer, device: torch.device,
          loss_charge_weight: float = 0.1, c: float=0.01):
    model.train()

    total_loss = 0
    train_cnt = 0
    for x, charge in loader:
        x = x.to(device)
        charge = charge.to(device)

        x_recon, charge_pred, _ = model.forward_internal(x)

        loss_recon = recon_loss_fn(x_recon, x)
        loss_charge = charge_loss_fn(charge_pred, charge)
        l1_norm = sum(p.abs().sum() for p in model.parameters())

        loss = loss_recon + loss_charge_weight * loss_charge + c * l1_norm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        train_cnt += x.size(0)

    print(f"Train loss: {total_loss / train_cnt:.4f}")


