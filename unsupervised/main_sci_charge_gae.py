import datetime
import sys
import os
import csv
import numpy as np
import shap
import torch
import torch_geometric.nn as pyg_nn
from torch_geometric.loader import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

from common.balanced_sample_tool import TheorySampler
from common.sci_parser import SuperConformalIndex
from common.utils import GenericDataset, FullyConnectedNetwork, MRSELoss
from torch.optim import Optimizer
from regression.main_sci_regression_gnn import build_data_sci


class GAEEncoder(nn.Module):
    def __init__(self, in_channels: int, edge_channels: int, hidden_channels: int,
                 latent_dim: int, heads: int=2):
        super(GAEEncoder, self).__init__()
        self.conv1 = pyg_nn.GATConv(in_channels, hidden_channels, heads=heads, concat=True, edge_dim=edge_channels)
        self.elu1 = nn.GELU()

        self.conv_mu = pyg_nn.GATConv(hidden_channels * heads, latent_dim, heads=1, concat=False, edge_dim=edge_channels)
        self.conv_logvar = pyg_nn.GATConv(hidden_channels * heads, latent_dim, heads=1, concat=False, edge_dim=edge_channels)

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.elu1(self.conv1(x, edge_index, edge_attr))
        mu = self.conv_mu(h, edge_index, edge_attr)
        logvar = self.conv_logvar(h, edge_index, edge_attr)
        return mu, logvar


class ChargeVGAE(pyg_nn.VGAE):
    def __init__(self, encoder: nn.Module, latent_dim: int, edge_channels: int):
        super(ChargeVGAE, self).__init__(encoder)
        self.regressor = nn.Linear(latent_dim, 1)
        self.edge_decoder = nn.Sequential(
            nn.Linear(2 * latent_dim, edge_channels),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_attr, batch):
        z = self.encode(x, edge_index, edge_attr, batch)
        adj_recon = torch.sigmoid(torch.matmul(z, z.t()))

        row, col = edge_index
        edge_inputs = torch.cat([z[row], z[col]], dim=-1)
        edge_recon = self.edge_decoder(edge_inputs)

        z_pooled = pyg_nn.global_mean_pool(z, batch)
        charges = self.regressor(z_pooled)

        return adj_recon, edge_recon, charges, z


def train(loader: DataLoader, model: nn.Module, optimizer: Optimizer,
          recon_loss_fn: nn.Module, edge_loss_fn: nn.Module, charge_loss_fn: nn.Module,
          device: torch.device, c: float=0.01):
    model.train()
    for _, data in enumerate(loader):
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(device)
        batch = data.batch.to(device)
        y = data.y.to(device)

        adj_recon, edge_recon, y_pred, _ = model(x, edge_index, edge_attr, batch)

        adj_true = torch.zeros_like(adj_recon, dtype=torch.float32)
        adj_true[data.edge_index[0], data.edge_index[1]] = 1.0
        adj_true = adj_true.to(device)

        loss_recon = recon_loss_fn(adj_recon, adj_true)
        loss_edge = edge_loss_fn(edge_recon, edge_attr)
        loss_charges = charge_loss_fn(y_pred, y)
        kl_loss = (1 / data.num_nodes) * model.kl_loss()
        l1_norm = sum(p.abs().sum() for p in model.parameters())

        loss = loss_recon + loss_edge + 0.5 * loss_charges + kl_loss + c * l1_norm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item():.4f}")


if __name__ == "__main__":
    print(sys.version)
    print("GIL enabled:", sys._is_gil_enabled())

    os.makedirs('../data/regression', exist_ok=True)
    csv.field_size_limit(np.iinfo(np.int32).max)

    filename = input("Enter file name to load: ")

    theory_sampler = TheorySampler(filename)
    stats = theory_sampler.get_theory_stats()
    for row in stats.collect():
        print(row.asDict())

    DIM_LO = float(input("Enter lower bound of dimension of operator to use: "))
    DIM_HI = float(input("Enter upper bound of dimension of operator to use: "))

    central_charge = ""
    tmp = int(input("Enter which charge to use; 1. a, 2. c\n>>>"))
    if tmp == 1:
        central_charge = "CentralChargeA"
    else:
        central_charge = "CentralChargeC"
    min_charge = float(input("Enter minimum charge: "))
    max_charge = float(input("Enter maximum charge: "))
    n_bins = int(input("Enter number of bins: "))

    theory_sampler.get_bins_stats(central_charge, min_charge, max_charge, n_bins).show(n=n_bins, truncate=False)
    n_per_bins = int(input("Enter number of theories per bin: "))

    n_train = int(input("Enter number of training samples: "))
    n_test = int(input("Enter number of testing samples: "))
    n_validate = int(input("Enter number of validation samples: "))

    data_train, data_test, data_validate = build_data_sci(
        theory_sampler, central_charge, min_charge, max_charge, n_bins, n_per_bins,
        n_train, n_test, n_validate, DIM_LO, DIM_HI
    )

    dataloader_train = [DataLoader(data_train[i], batch_size=64, shuffle=True) for i in range(len(data_train))]
    dataloader_test = [DataLoader(data_test[i], batch_size=64, shuffle=False) for i in range(len(data_test))]
    dataloader_validate = [DataLoader(data_validate[i], batch_size=64, shuffle=False) for i in range(len(data_validate))]

    for step, data in enumerate(dataloader_train[0]):
        print(f'Step {step + 1}:')
        print('========')
        print(f'Number of graphs in the current batch: {data.num_graphs}')
        print(data)
        print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latent_dim = 32
    encoder = GAEEncoder(2, 1, 8, latent_dim).to(device)
    model = ChargeVGAE(encoder, latent_dim, 1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    recon_loss_fn = nn.BCELoss()  # adjacency reconstruction
    edge_loss_fn = nn.MSELoss()  # edge attribute reconstruction
    charge_loss_fn = MRSELoss()

    for epoch in range(100):
        train(dataloader_train[0], model, optimizer,
              recon_loss_fn, edge_loss_fn, charge_loss_fn,
              device, 0)
