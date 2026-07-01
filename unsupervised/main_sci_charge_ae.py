from sklearn.metrics import r2_score, root_mean_squared_error
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.nn as nn
import shap
import torch
import numpy as np
import sys
import os
import csv

from common.balanced_sample_tool import TheorySampler
from common.sci_parser import SuperConformalIndex
from common.utils import FullyConnectedNetwork, GenericDataset


def build_data_sci(sampler: TheorySampler, charge_col: str, min_charge: float, max_charge: float,
                   n_bins: int, n_per_bins: int, n_train: int, n_test: int, n_validate: int,
                   grid: np.ndarray, kde_bandwidth: float):
    input_train = []
    output_train = []
    input_test = []
    output_test = []
    id_validate = []
    input_validate = []
    output_validate = []

    def build_dataset(data: TheorySampler, input_set, output_set, id_set=None):
        rows = data.df.collect()

        input_data = []
        output_data = []
        id_data = []
        for row in rows:
            input_data.append(SuperConformalIndex(row["SCI"]).featurize_sci(grid, kde_bandwidth))
            output_data.append([float(row["CentralChargeA"]), float(row["CentralChargeC"])])
            if id_set is not None:
                id_data.append(int(row["id"]))

        input_data = np.stack(input_data)
        output_data = np.stack(output_data)

        input_set.append(input_data)
        output_set.append(output_data)
        if id_set is not None:
            id_set.append(id_data)

    for i in range(n_train):
        build_dataset(sampler.get_balanced_bins_sample(charge_col, min_charge, max_charge, n_bins, n_per_bins),
                      input_train, output_train)
        print(f"Train data {i + 1} built.")

    for i in range(n_test):
        build_dataset(sampler.get_balanced_bins_sample(charge_col, min_charge, max_charge, n_bins, n_per_bins),
                      input_test, output_test)
        print(f"Test data {i + 1} built.")

    for i in range(n_validate):
        build_dataset(sampler.get_balanced_bins_sample(charge_col, min_charge, max_charge, n_bins, n_per_bins),
                      input_validate, output_validate, id_set=id_validate)
        print(f"Validation data {i + 1} built.")

    return input_train, output_train, input_test, output_test, id_validate, input_validate, output_validate


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

def test(loader: DataLoader, model: SCIAutoencoder, recon_loss_fn: nn.Module, charge_loss_fn: nn.Module,
         device: torch.device, loss_charge_weight: float = 0.1):
    model.eval()
    test_loss = 0.0
    recon_error = 0.0
    charge_error = 0.0
    test_cnt = 0
    recon_cnt = 0
    charge_cnt = 0

    with torch.no_grad():
        for x, charge in loader:
            x = x.to(device)
            charge = charge.to(device)

            x_recon, charge_pred, _ = model.forward_internal(x)

            loss_recon = recon_loss_fn(x_recon, x)
            loss_charge = charge_loss_fn(charge_pred, charge)

            loss = loss_recon + loss_charge_weight * loss_charge

            test_loss += loss.item() * x.size(0)
            test_cnt += x.size(0)

            x_recon = x_recon.cpu().numpy()
            charge_pred = charge_pred.cpu().numpy()
            x = x.cpu().numpy()
            charge = charge.cpu().numpy()

            err = np.concatenate(np.abs((x_recon - x) / (x + 1e-12)))
            recon_error += np.sum(err)
            recon_cnt += len(err)

            err = np.concatenate(np.abs((charge_pred - charge) / charge))
            charge_error += np.sum(err)
            charge_cnt += len(err)

    return test_loss, recon_error, charge_error, test_cnt, recon_cnt, charge_cnt


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

    GRID_LO = float(input("Enter lower bound of feature grid: "))
    GRID_HI = float(input("Enter upper bound of feature grid: "))
    GRID_STEP = float(input("Enter step size of feature grid: "))

    GRID = np.arange(GRID_LO, GRID_HI + GRID_STEP, GRID_STEP)
    KDE_BANDWIDTH = float(input("Enter bandwidth of feature grid: "))

    central_charge = ""
    tmp = int(input("Enter which charge to use for sampling; 1. a, 2. c\n>>>"))
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

    input_train, output_train, input_test, output_test, id_validate, input_validate, output_validate = build_data_sci(
        theory_sampler, central_charge, min_charge, max_charge, n_bins, n_per_bins,
        n_train, n_test, n_validate, GRID, KDE_BANDWIDTH
    )

    dataset_train = [GenericDataset(input_train[i], output_train[i]) for i in range(n_train)]
    dataset_test = [GenericDataset(input_test[i], output_test[i]) for i in range(n_test)]

    dataloader_train = [DataLoader(dataset_train[i], batch_size=32, shuffle=True) for i in range(n_train)]
    dataloader_test = [DataLoader(dataset_test[i], batch_size=32, shuffle=False) for i in range(n_test)]

    for index, (x, y) in enumerate(dataloader_train[0]):
        print(f'{index}/{len(dataloader_train)}', end=' ')
        print('x shape: ', x.shape, end=' ')
        print('y shape: ', y.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    input_num = input_train[0].shape[1]
    model = SCIAutoencoder(input_num, 16).to(device)

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_loss = 1e10

    n_epochs = int(input("Enter number of epochs: "))
    checkpoint_path = input("Enter the name of the checkpoint file: ")

    file_suffix = f"{central_charge[-1].lower()}_{min_charge}_{max_charge}_{GRID_LO}_{GRID_HI}_{GRID_STEP}_{KDE_BANDWIDTH}"
    if os.path.isfile(checkpoint_path):
        print('Checkpoint available. Loads checkpoint...')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_loss = checkpoint['best_loss']

    for epoch in range(n_epochs):
        print(f"Train epoch {epoch + 1}...")
        for i in range(n_train):
            train(dataloader_train[i], model, criterion, criterion, optimizer, device,
                  loss_charge_weight=0.5, c=0)
            print(f"Training set {i + 1}/{n_train} complete.")

        print(f"Test epoch {epoch + 1}...")
        total_loss = 0.0
        total_recon_error = 0.0
        total_charge_error = 0.0
        total_cnt = 0
        total_recon_cnt = 0
        total_charge_cnt = 0

        for i in range(n_test):
            loss, recon_error, charge_error, cnt, recon_cnt, charge_cnt = test(
                dataloader_test[i], model, criterion, criterion, device,
                loss_charge_weight=0.5
            )
            total_loss += loss
            total_recon_error += recon_error
            total_charge_error += charge_error
            total_cnt += cnt
            total_recon_cnt += recon_cnt
            total_charge_cnt += charge_cnt
            print(f"Test set {i + 1}/{n_test} complete.")

        total_loss /= total_cnt
        total_recon_error /= total_recon_cnt
        total_charge_error /= total_charge_cnt
        print(f"Epoch {epoch + 1}/{n_epochs} test loss: {total_loss}")
        print(f"reconstruction error: {total_recon_error * 100:.2f} % charge error: {total_charge_error * 100:.2f} %")
        if total_loss < best_loss:
            best_loss = total_loss
            print('New best loss obtained. Saving model...')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss
            }, checkpoint_path)
