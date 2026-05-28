import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, root_mean_squared_error
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import numpy as np
import sys
import os
import csv
import matplotlib.pyplot as plt

from common.balanced_sample_tool import TheorySampler
from common.sci_parser import SuperConformalIndex
from common.utils import FullyConnectedNetwork


def build_data(sampler: TheorySampler, charge_col: str, min_charge: float, max_charge: float,
               n_bins: int, n_per_bins: int, n_train: int, n_test: int, n_validate: int):
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
            sci = SuperConformalIndex(row["SCI"])
            input_data.append(torch.tensor([[sci.relevant_dims[i], float(sci.relevant_spectrum[sci.relevant_dims[i]])] for i in range(len(sci.relevant_dims))]))
            output_data.append([float(row[charge_col])])

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


# Dataset with variable-length float sequences
class SequenceDataset(Dataset):
    def __init__(self, sequences, outputs):
        self.sequences = sequences
        self.outputs = outputs

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index], self.outputs[index]


# Custom collate function to pad sequences
def collate_fn(batch):
    sequences, outputs = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded = pad_sequence(sequences, batch_first=True)
    outputs = torch.tensor(np.array(outputs))
    return padded, lengths, outputs


class GRURegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, *args):
        super(GRURegressionModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = FullyConnectedNetwork(hidden_dim, 1, *args)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.gru(packed)
        last_hidden = hidden[-1]
        return self.fc(last_hidden)


def train(loader: DataLoader, model: nn.Module, criterion: nn.Module, optimizer: Optimizer, device: torch.device):
    model.train()
    for x, lengths, y in loader:
        x = x.to(device)
        y = y.to(device)

        y_pred = model(x, lengths)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(loader: DataLoader, model: nn.Module, criterion: nn.Module, device: torch.device):
    model.eval()
    test_loss = 0.0
    error = 0.0
    test_cnt = 0

    with torch.no_grad():
        for x, lengths, y in loader:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x, lengths)
            loss = criterion(y_pred, y)

            test_loss += loss.item()

            y_pred = y_pred.cpu().numpy()
            y = y.cpu().numpy()
            err = np.concatenate(np.abs((y_pred - y) / y))
            error += np.sum(err)
            test_cnt += len(err)

    return test_loss, error, test_cnt

def validate(loader: DataLoader, model: nn.Module, device: torch.device):
    model.eval()
    y_real = []
    y_pred = []

    with torch.no_grad():
        for x, lengths, y in loader:
            x = x.to(device)
            y_real += y.cpu().numpy().ravel().tolist()

            outputs = model(x, lengths)
            y_pred += outputs.cpu().numpy().ravel().tolist()

    y_real = np.asarray(y_real)
    y_pred = np.asarray(y_pred)
    r2 = r2_score(y_real, y_pred)
    rmse = root_mean_squared_error(y_real, y_pred)
    error = np.sum(np.abs(y_real - y_pred) / y_real)
    error /= len(y_real)

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
        n_train, n_test, n_validate
    )

    dataset_train = [SequenceDataset(input_train[i], output_train[i]) for i in range(n_train)]
    dataset_test = [SequenceDataset(input_test[i], output_test[i]) for i in range(n_test)]
    dataset_validate = [SequenceDataset(input_validate[i], output_validate[i]) for i in range(n_validate)]

    dataloader_train = [DataLoader(dataset_train[i], batch_size=32, collate_fn=collate_fn, shuffle=True) for i in range(n_train)]
    dataloader_test = [DataLoader(dataset_test[i], batch_size=32, collate_fn=collate_fn, shuffle=False) for i in range(n_test)]
    dataloader_validate = [DataLoader(dataset_validate[i], batch_size=32, collate_fn=collate_fn, shuffle=False) for i in range(n_validate)]
    for index, (padded, lengths, outputs) in enumerate(dataloader_train[0]):
        print(f'{index}/{len(dataloader_train)}', end=' ')
        print('x shape: ', padded.shape, end=' ')
        print('y shape: ', outputs.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_dim = 16
    model = GRURegressionModel(
        2, hidden_dim, 4,
        (hidden_dim * 4, nn.GELU()),
        (hidden_dim * 4, nn.GELU()),
        (hidden_dim * 16, nn.GELU()),
        (hidden_dim * 16, nn.GELU()),
        (hidden_dim * 16, nn.GELU()),
        (hidden_dim * 16, nn.GELU()),
        (hidden_dim * 8, nn.GELU()),
        (hidden_dim * 8, nn.GELU()),
        (hidden_dim * 2, nn.GELU()),
        (hidden_dim * 1, nn.GELU())
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_loss = 1e10

    n_epochs = int(input("Enter number of epochs: "))

    file_suffix = f"{central_charge[-1].lower()}"
    checkpoint_path = f"../data/regression/checkpoint_spectrum_regression_rnn_{file_suffix}.tar"
    if os.path.isfile(checkpoint_path):
        print('Checkpoint available. Loads checkpoint...')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_loss = checkpoint['best_loss']

    for epoch in range(n_epochs):
        print(f"Train epoch {epoch + 1}...")
        for i in range(n_train):
            train(dataloader_train[i], model, criterion, optimizer, device)
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
        print(
            f"Epoch {epoch + 1}/{n_epochs} test loss: {total_loss} error: {total_error * 100} %")
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
        y_real, y_pred, r2, rmse, error = validate(dataloader_validate[i], model, device)
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

        plt.savefig(f"{save_dir}/spectrum_regression_rnn_{central_charge[-1].lower()}_{i + 1}.png")

    with open(f"{save_dir}/spectrum_regression_rnn_{file_suffix}.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Score"] + [i + 1 for i in range(n_validate)])
        writer.writerow(["R2"] + r2_scores)
        writer.writerow(["RMSE"] + rmse_scores)
        writer.writerow(["Error"] + errors)
