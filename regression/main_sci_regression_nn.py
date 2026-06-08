import datetime
import sys
import os
import csv
import numpy as np
import torch
from sklearn.metrics import r2_score, root_mean_squared_error
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

from common.balanced_sample_tool import TheorySampler
from common.sci_parser import SuperConformalIndex
from common.utils import GenericDataset, FullyConnectedNetwork


def build_data(sampler: TheorySampler, charge_col: str, min_charge: float, max_charge: float,
               n_bins: int, n_per_bins: int, n_train: int, n_test: int, n_validate: int,
               grid: np.ndarray, kde_bandwidth: float):
    input_train = []
    output_train = []
    input_test = []
    output_test = []
    input_validate = []
    output_validate = []

    def build_dataset(data: TheorySampler, input_set, output_set):
        rows = data.df.collect()

        input_data = []
        output_data = []
        for row in rows:
            input_data.append(SuperConformalIndex(row["SCI"]).featurize_sci(grid, kde_bandwidth))
            output_data.append([float(row[charge_col])])

        input_data = np.stack(input_data)
        output_data = np.stack(output_data)

        input_set.append(input_data)
        output_set.append(output_data)

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
                      input_validate, output_validate)
        print(f"Validation data {i + 1} built.")

    return input_train, output_train, input_test, output_test, input_validate, output_validate


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

def validate(X: torch.Tensor, y: torch.Tensor, model: nn.Module, device: torch.device):
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
    error = np.sum(np.abs(y_real - y_pred) / y_real)
    error /= len(y_real)

    print("Validation stats")
    print("=======================")
    print(f"    R2: {r2:.4f}")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    Error: {error:.4f}")

    return y_real, y_pred, r2, rmse, float(error)


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

    input_train, output_train, input_test, output_test, input_validate, output_validate = build_data(
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

    input_num = input_train[0].shape[1]
    model = FullyConnectedNetwork(
        input_num, 1,
        (input_num * 4, nn.GELU()),
        (input_num * 4, nn.GELU()),
        (input_num * 16, nn.GELU()),
        (input_num * 16, nn.GELU()),
        (input_num * 32, nn.GELU()),
        (input_num * 32, nn.GELU()),
        (input_num * 32, nn.GELU()),
        (input_num * 32, nn.GELU()),
        (input_num * 8, nn.GELU()),
        (input_num * 8, nn.GELU()),
        (input_num * 2, nn.GELU()),
        (input_num * 1, nn.GELU()),
    ).to(device)

    print("Charge calculation model shape: ", model(
        torch.randn(32, input_num).to(device)
    ).shape)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_loss = 1e10

    n_epochs = int(input("Enter number of epochs: "))
    checkpoint_path = input("Enter the name of the checkpoint file: ")

    file_suffix = f"{central_charge[-1].lower()}_{GRID_HI}_{GRID_STEP}_{KDE_BANDWIDTH}"
    if os.path.isfile(checkpoint_path):
        print('Checkpoint available. Loads checkpoint...')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_loss = checkpoint['best_loss']

    for epoch in range(n_epochs):
        print(f"Train epoch {epoch + 1}...")
        for i in range(n_train):
            train(dataloader_train[i], model, criterion, optimizer, device, 0.00001)
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

    r2_scores = []
    rmse_scores = []
    errors = []
    save_dir = f"../data/regression/{datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}"
    os.makedirs(save_dir, exist_ok=True)

    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 15

    for i in range(n_validate):
        y_real, y_pred, r2, rmse, error = validate(
            torch.tensor(input_validate[i], dtype=torch.float32),
            torch.tensor(output_validate[i], dtype=torch.float32),
            model, device)
        r2_scores.append(r2)
        rmse_scores.append(rmse)
        errors.append(f"{error * 100.0}%")

        plt.close('all')

        fig, ax = plt.subplots()
        fig.suptitle(f"Neural Network Regression of central charge {central_charge[-1].lower()}")

        ax.set_title(f"R2 = {r2:.3f}, RMSE = {rmse:.3f}")
        ax.scatter(y_real, y_pred)
        y_range = [np.min(y_real), np.max(y_real)]
        ax.plot(y_range, y_range, linestyle='--', color='red')
        ax.set_xlabel(f"Real {central_charge[-1].lower()}")
        ax.set_ylabel(f"Predicted {central_charge[-1].lower()}")

        plt.savefig(f"{save_dir}/sci_regression_nn_{central_charge[-1].lower()}_{i + 1}.png")

    with open(f"{save_dir}/sci_regression_nn_{file_suffix}.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Score"] + [i + 1 for i in range(n_validate)])
        writer.writerow(["R2"] + r2_scores)
        writer.writerow(["RMSE"] + rmse_scores)
        writer.writerow(["Error"] + errors)
