import datetime
import sys
import os
import csv
import numpy as np
import shap
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

from common.balanced_sample_tool import TheorySampler
from common.sci_parser import SuperConformalIndex
from common.utils import GenericDataset, FullyConnectedNetwork, MRSELoss
from regression.regression_common import train, test, validate


def build_data_spectrum(sampler: TheorySampler, train_sampler: TheorySampler, test_sampler: TheorySampler,
                        charge_col: str, min_charge: float, max_charge: float,
                        n_bins: int, n_per_bins_train: int, n_per_bins_test: int, n_per_bins_validate: int,
                        n_train: int, n_test: int, n_validate: int,
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
            input_data.append(SuperConformalIndex(row["SCI"]).featurize_relevant_spectrum(grid, kde_bandwidth))
            output_data.append([float(row[charge_col])])
            if id_set is not None:
                id_data.append(int(row["id"]))

        input_data = np.stack(input_data)
        output_data = np.stack(output_data)

        input_set.append(input_data)
        output_set.append(output_data)
        if id_set is not None:
            id_set.append(id_data)

    for i in range(n_train):
        build_dataset(train_sampler.get_balanced_bins_sample(charge_col, min_charge, max_charge, n_bins, n_per_bins_train),
                      input_train, output_train)
        print(f"Train data {i + 1} built.")

    for i in range(n_test):
        build_dataset(test_sampler.get_balanced_bins_sample(charge_col, min_charge, max_charge, n_bins, n_per_bins_test),
                      input_test, output_test)
        print(f"Test data {i + 1} built.")

    for i in range(n_validate):
        build_dataset(sampler.get_balanced_bins_sample(charge_col, min_charge, max_charge, n_bins, n_per_bins_validate),
                      input_validate, output_validate, id_set=id_validate)
        print(f"Validation data {i + 1} built.")

    return input_train, output_train, input_test, output_test, id_validate, input_validate, output_validate


def build_data_sci(sampler: TheorySampler, train_sampler: TheorySampler, test_sampler: TheorySampler,
                        charge_col: str, min_charge: float, max_charge: float,
                        n_bins: int, n_per_bins_train: int, n_per_bins_test: int, n_per_bins_validate: int,
                        n_train: int, n_test: int, n_validate: int,
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
            output_data.append([float(row[charge_col])])
            if id_set is not None:
                id_data.append(int(row["id"]))

        input_data = np.stack(input_data)
        output_data = np.stack(output_data)

        input_set.append(input_data)
        output_set.append(output_data)
        if id_set is not None:
            id_set.append(id_data)

    for i in range(n_train):
        build_dataset(train_sampler.get_balanced_bins_sample(charge_col, min_charge, max_charge, n_bins, n_per_bins_train),
                      input_train, output_train)
        print(f"Train data {i + 1} built.")

    for i in range(n_test):
        build_dataset(test_sampler.get_balanced_bins_sample(charge_col, min_charge, max_charge, n_bins, n_per_bins_test),
                      input_test, output_test)
        print(f"Test data {i + 1} built.")

    for i in range(n_validate):
        build_dataset(sampler.get_balanced_bins_sample(charge_col, min_charge, max_charge, n_bins, n_per_bins_validate),
                      input_validate, output_validate, id_set=id_validate)
        print(f"Validation data {i + 1} built.")

    return input_train, output_train, input_test, output_test, id_validate, input_validate, output_validate


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

    print("Enter the program:")
    print("1. Relevant operator spectrum")
    print("2. Full SCI")
    program = int(input(">>"))

    program_name = ""
    number_of_ops_name = ""
    if program == 1:
        program_name = "spectrum"
        number_of_ops_name = "Number of relevant operators"
    elif program == 2:
        program_name = "sci"
        number_of_ops_name = "Sum of absolute values of coefficients"
    else:
        print("Invalid program number.")
        exit(1)

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

    print("Whole data stats")
    theory_sampler.get_bins_stats(central_charge, min_charge, max_charge, n_bins).show(n=n_bins, truncate=False)

    train_ratio = float(input("Enter train ratio: "))
    train_sampler, test_sampler = theory_sampler.get_train_test_sets_bins(central_charge, min_charge, max_charge, n_bins, train_ratio)

    print("Train data stats")
    train_sampler.get_bins_stats(central_charge, min_charge, max_charge, n_bins).show(n=n_bins, truncate=False)
    print("Test data stats")
    test_sampler.get_bins_stats(central_charge, min_charge, max_charge, n_bins).show(n=n_bins, truncate=False)

    n_train = int(input("Enter number of training samples: "))
    n_per_bins_train = int(input("Enter number of training samples per bin: "))
    n_test = int(input("Enter number of testing samples: "))
    n_per_bins_test = int(input("Enter number of testing samples per bin: "))
    n_validate = int(input("Enter number of validation samples: "))
    n_per_bins_validate = int(input("Enter number of validation samples per bin(validation data are chosen from the whole dataset): "))

    input_train, output_train, input_test, output_test, id_validate, input_validate, output_validate = None, None, None, None, None, None, None
    if program == 1:
        input_train, output_train, input_test, output_test, id_validate, input_validate, output_validate = build_data_spectrum(
            theory_sampler, train_sampler, test_sampler,
            central_charge, min_charge, max_charge,
            n_bins, n_per_bins_train, n_per_bins_test, n_per_bins_validate,
            n_train, n_test, n_validate, GRID, KDE_BANDWIDTH
        )
    if program == 2:
        input_train, output_train, input_test, output_test, id_validate, input_validate, output_validate = build_data_sci(
            theory_sampler, train_sampler, test_sampler,
            central_charge, min_charge, max_charge,
            n_bins, n_per_bins_train, n_per_bins_test, n_per_bins_validate,
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
        (input_num * 2, nn.GELU()),
        (input_num, nn.GELU()),
        (input_num // 2, nn.GELU()),
        (input_num // 4, nn.GELU()),
        (input_num // 8, nn.GELU()),
        (input_num // 16, nn.GELU()),
    ).to(device)

    print("Charge calculation model shape: ", model(
        torch.randn(32, input_num).to(device)
    ).shape)

    criterion = MRSELoss(multiplier=max_charge)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_loss = 1e10

    n_epochs = int(input("Enter number of epochs: "))
    checkpoint_path = input("Enter the name of the checkpoint file: ")

    file_suffix = f"{central_charge[-1].lower()}_{min_charge}_{max_charge}_{GRID_HI}_{GRID_STEP}_{KDE_BANDWIDTH}"
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

    r2_scores = []
    rmse_scores = []
    errors = []
    outliers = set()
    save_dir = f"../data/regression/{datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}"
    os.makedirs(save_dir, exist_ok=True)

    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 15

    for i in range(n_validate):
        y_real, y_pred, r2, rmse, error, shap_value, outlier_mask = validate(
            torch.tensor(input_validate[i], dtype=torch.float32),
            torch.tensor(output_validate[i], dtype=torch.float32),
            model, device, i == 0)
        r2_scores.append(r2)
        rmse_scores.append(rmse)
        errors.append(error * 100.0)
        ids = np.asarray(id_validate[i], dtype=int)
        outliers.update(ids[outlier_mask].tolist())

        plt.close('all')

        fig, ax = plt.subplots()
        fig.suptitle(f"Neural Network Regression of central charge {central_charge[-1].lower()}")

        ax.set_title(f"R2 = {r2:.3f}, RMSE = {rmse:.3f}")
        ax.scatter(y_real[~outlier_mask], y_pred[~outlier_mask], color='blue', label='Inliers')
        ax.scatter(y_real[outlier_mask], y_pred[outlier_mask], color='red', label='Outliers')
        y_range = [np.min(y_real), np.max(y_real)]
        ax.plot(y_range, y_range, linestyle='--', color='green')
        ax.set_xlabel(f"Real {central_charge[-1].lower()}")
        ax.set_ylabel(f"Predicted {central_charge[-1].lower()}")
        ax.legend()

        plt.savefig(f"{save_dir}/{program_name}_regression_nn_{central_charge[-1].lower()}_{min_charge}_{max_charge}_{i + 1}.png")

        if shap_value is not None:
            shap_value.feature_names = [f"{GRID[i]:.4f}" for i in range(len(GRID))] + [
                "Minimal gap",
                "Maximal gap",
                "Mean gap",
                "Stdev gap",
                "Median gap",
                "1st quartile gap",
                "3rd quartile gap",
                number_of_ops_name,
                f"Log of {number_of_ops_name}",
                "Mean dimension",
                "Stdev dimension",
                "Minimal dimension",
                "Maximal dimension",
                "Median dimension",
                "1st quartile dimension",
                "3rd quartile dimension"
            ]
            shap_value.values = shap_value.values.squeeze(2)
            shap_value.data = shap_value.data.detach().cpu().numpy()
            plt.close("all")

            fig, ax = plt.subplots(figsize=(24, 16))
            ax.set_title(f"SHAP values for central charge {central_charge[-1].lower()}")
            shap.plots.beeswarm(shap_value, max_display=15, show=False, color_bar=True, ax=ax, group_remaining_features=True, plot_size=None)

            plt.savefig(f"{save_dir}/{program_name}_regression_nn_{central_charge[-1].lower()}_{min_charge}_{max_charge}_shap.png")

            plt.close("all")

            fig, ax = plt.subplots(figsize=(24, 16))
            ax.set_title(f"Absolute average SHAP values for central charge {central_charge[-1].lower()}")
            shap.plots.bar(shap_value, max_display=15, show=False, ax=ax)

            plt.savefig(f"{save_dir}/{program_name}_regression_nn_{central_charge[-1].lower()}_{min_charge}_{max_charge}_shap_abs_avg.png")

            with open(f"{save_dir}/{program_name}_regression_nn_{file_suffix}_shap.csv", 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Theory"] + shap_value.feature_names)
                for i in range(len(shap_value.values)):
                    writer.writerow([f"{i + 1}"] + shap_value.values[i].tolist())

    with open(f"{save_dir}/{program_name}_regression_nn_{file_suffix}.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Score"] + [i + 1 for i in range(n_validate)])
        writer.writerow(["R2"] + r2_scores)
        writer.writerow(["RMSE"] + rmse_scores)
        writer.writerow(["Error (%)"] + errors)

    with open(f"{save_dir}/{program_name}_regression_nn_{file_suffix}_outliers.txt", "w") as f:
        f.writelines([f"{i}\n" for i in outliers])
