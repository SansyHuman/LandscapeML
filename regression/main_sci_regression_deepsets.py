import datetime
import sys
import os
import csv
from typing import Union

import numpy as np
import shap
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import matplotlib.pyplot as plt

from common.balanced_sample_tool import TheorySampler
from common.sci_parser import SuperConformalIndex
from common.utils import GenericDataset, FullyConnectedNetwork, MRSELoss, DeepSetsInvariant, GenericDeepSetsDataset


class DeepSetsDataset:
    def __init__(self, input: np.ndarray, mask: np.ndarray, output: np.ndarray, id: Union[list[int], None]=None):
        self.input = input
        self.mask = mask
        self.output = output
        self.id = id


class DeepSetsData:
    def __init__(self, train_set: list[DeepSetsDataset], test_set: list[DeepSetsDataset], validation_set: list[DeepSetsDataset]):
        self.train_set = train_set
        self.test_set = test_set
        self.validation_set = validation_set


class DeepSetsRegressionModel(nn.Module):
    def __init__(self, deepsets: DeepSetsInvariant, hidden_dims: list[int]):
        super().__init__()
        self.deepsets = deepsets
        self.fcn = FullyConnectedNetwork(deepsets.output_dim, 1,
                                         *list(zip(hidden_dims, [nn.GELU()] * len(hidden_dims))))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = self.get_feature_vector(x, mask)
        return self.fcn(z)

    def get_feature_vector(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.deepsets(x, mask)


def build_data_sci(sampler: TheorySampler, train_sampler: TheorySampler, test_sampler: TheorySampler,
                        charge_col: str, min_charge: float, max_charge: float,
                        n_bins: int, n_per_bins_train: int, n_per_bins_test: int, n_per_bins_validate: int,
                        n_train: int, n_test: int, n_validate: int):
    train_set = []
    test_set = []
    validation_set = []

    def build_dataset(data: TheorySampler, build_id_set=False):
        rows = data.df.collect()

        input_data = []
        output_data = []
        id_data = [] if build_id_set else None
        for row in rows:
            input_data.append(SuperConformalIndex(row["SCI"]).terms_list)
            output_data.append([float(row[charge_col])])
            if build_id_set:
                id_data.append(int(row["id"]))

        max_len = max(len(t) for t in input_data)
        padded = np.stack([
            np.concatenate((t, np.zeros((max_len - len(t), 3)))) for t in input_data
        ])
        mask = np.stack([
            np.concatenate((np.ones(len(t)), np.zeros(max_len - len(t)))) for t in input_data
        ])
        output_data = np.stack(output_data)

        return DeepSetsDataset(padded, mask, output_data, id_data)

    for i in range(n_train):
        train_set.append(build_dataset(train_sampler.get_balanced_bins_sample(charge_col, min_charge, max_charge, n_bins, n_per_bins_train)))
        print(f"Train data {i + 1} built.")

    for i in range(n_test):
        test_set.append(build_dataset(test_sampler.get_balanced_bins_sample(charge_col, min_charge, max_charge, n_bins, n_per_bins_test)))
        print(f"Test data {i + 1} built.")

    for i in range(n_validate):
        validation_set.append(build_dataset(sampler.get_balanced_bins_sample(charge_col, min_charge, max_charge, n_bins, n_per_bins_validate), build_id_set=True))
        print(f"Validation data {i + 1} built.")

    return DeepSetsData(train_set, test_set, validation_set)


def train(loader: DataLoader, model: nn.Module, criterion: nn.Module, optimizer: Optimizer, device: torch.device, c: float=0.01):
    model.train()
    for x, mask, y in loader:
        x = x.to(device)
        mask = mask.to(device)
        y = y.to(device)

        y_pred = model(x, mask)
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
        for x, mask, y in loader:
            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device)

            y_pred = model(x, mask)
            loss = criterion(y_pred, y)

            test_loss += loss.item() * x.size(0)

            y_pred = y_pred.cpu().numpy()
            y = y.cpu().numpy()
            err = np.concatenate(np.abs((y_pred - y) / y))
            error += np.sum(err)
            test_cnt += len(err)

    return test_loss, error, test_cnt


if __name__ == "__main__":
    print(sys.version)
    print("GIL enabled:", sys._is_gil_enabled())

    os.makedirs('../data/regression', exist_ok=True)
    csv.field_size_limit(np.iinfo(np.int32).max)

    filename = input("Enter file name to load: ")

    theory_sampler = TheorySampler(filename)
    stats = theory_sampler.get_theory_stats()
    seeds = stats.select("Name").rdd.map(lambda row: row[0]).collect()
    for row in stats.collect():
        print(row.asDict())

    train_seeds = [seeds[i] for i in range(len(seeds)) if (i + 1) % 5 != 0]
    test_seeds = [seeds[i] for i in range(len(seeds)) if (i + 1) % 5 == 0]

    central_charge = ""
    tmp = int(input("Enter which charge to use; 1. a, 2. c\n>>>"))
    if tmp == 1:
        central_charge = "CentralChargeA"
    else:
        central_charge = "CentralChargeC"
    min_charge = float(input("Enter minimum charge: "))
    max_charge = float(input("Enter maximum charge: "))
    n_bins = int(input("Enter number of bins: "))

    print("Whole data stats")
    theory_sampler.get_bins_stats(central_charge, min_charge, max_charge, n_bins).show(n=n_bins, truncate=False)

    print(f"Train seeds: {train_seeds}")
    train_sampler = theory_sampler.get_selected_theories(train_seeds)
    train_sampler.get_bins_stats(central_charge, min_charge, max_charge, n_bins).show(n=n_bins, truncate=False)

    print(f"Test seeds: {test_seeds}")
    test_sampler = theory_sampler.get_selected_theories(test_seeds)
    test_sampler.get_bins_stats(central_charge, min_charge, max_charge, n_bins).show(n=n_bins, truncate=False)

    n_train = int(input("Enter number of training samples: "))
    n_per_bins_train = int(input("Enter number of training samples per bin: "))
    n_test = int(input("Enter number of testing samples: "))
    n_per_bins_test = int(input("Enter number of testing samples per bin: "))
    n_validate = int(input("Enter number of validation samples: "))
    n_per_bins_validate = int(input("Enter number of validation samples per bin(validation data are chosen from the whole dataset): "))

    data = build_data_sci(theory_sampler, train_sampler, test_sampler,
            central_charge, min_charge, max_charge,
            n_bins, n_per_bins_train, n_per_bins_test, n_per_bins_validate,
            n_train, n_test, n_validate)

    dataset_train = [GenericDeepSetsDataset(data.train_set[i].input, data.train_set[i].mask, data.train_set[i].output) for i in range(n_train)]
    dataset_test = [GenericDeepSetsDataset(data.test_set[i].input, data.test_set[i].mask, data.test_set[i].output) for i in range(n_test)]

    dataloader_train = [DataLoader(dataset_train[i], batch_size=32, shuffle=True) for i in range(n_train)]
    dataloader_test = [DataLoader(dataset_test[i], batch_size=32, shuffle=False) for i in range(n_test)]

    for index, (x, mask, y) in enumerate(dataloader_train[0]):
        print(f'{index}/{len(dataloader_train)}', end=' ')
        print('x shape: ', x.shape, end=' ')
        print('mask shape: ', mask.shape, end=' ')
        print('y shape: ', y.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    deepsets_output_dim = int(input("Enter deepsets output dimension: "))
    deepsets = DeepSetsInvariant(
        input_dim=3,
        phi_hidden_dims=[deepsets_output_dim // 3, deepsets_output_dim],
        rho_hidden_dims=[deepsets_output_dim, deepsets_output_dim],
        output_dim=deepsets_output_dim,
        pool_type="sum"
    )
    model = DeepSetsRegressionModel(deepsets, [
        deepsets_output_dim // 2,
        deepsets_output_dim // 4,
        deepsets_output_dim // 8,
        deepsets_output_dim // 16
    ]).to(device)

    criterion = MRSELoss(multiplier=max_charge)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_loss = 1e10

    n_epochs = int(input("Enter number of epochs: "))
    checkpoint_path = input("Enter the name of the checkpoint file: ")

    file_suffix = f"{central_charge[-1].lower()}_{min_charge}_{max_charge}_{deepsets_output_dim}"
    if os.path.isfile(checkpoint_path):
        print('Checkpoint available. Loads checkpoint...')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_loss = checkpoint['best_loss']

    for epoch in range(n_epochs):
        print(f"Train epoch {epoch + 1}...")
        for i in range(n_train):
            train(dataloader_train[i], model, criterion, optimizer, device, 0.0001)
            print(f"Training set {i + 1}/{n_train} complete.")

        print(f"Test epoch {epoch + 1}...")
        total_loss = 0.0
        total_error = 0.0
        total_cnt = 0
        for i in range(n_test):
            loss, error, cnt = test(dataloader_test[i], model, criterion, device)
            total_loss += loss
            total_error += error
            total_cnt += cnt
            print(f"Test set {i + 1}/{n_test} complete.")

        total_loss /= total_cnt
        total_error /= total_cnt
        print(f"Epoch {epoch + 1}/{n_epochs} test loss: {total_loss} error: {total_error * 100} %")
        if total_loss < best_loss:
            best_loss = total_loss
            print('New best loss obtained. Saving model...')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss
            }, checkpoint_path)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_loss = checkpoint['best_loss']
