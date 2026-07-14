import datetime
import random

from sklearn.metrics import r2_score, root_mean_squared_error
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import numpy as np
import sys
import os
import csv
import math

from common.balanced_sample_tool import TheorySampler
from common.sci_parser import SuperConformalIndex
from common.utils import FullyConnectedNetwork, GenericDataset, CorrelationLoss


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
            input_data.append(SuperConformalIndex(row["SCI"]).featurize_sci(grid, kde_bandwidth)[:-16])
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


class SCIAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super(SCIAutoencoder, self).__init__()

        self.encoder = FullyConnectedNetwork(
            input_dim, latent_dim,
            (input_dim // 2, nn.GELU()),
            (input_dim // 3, nn.GELU()),
            (input_dim // 4, nn.GELU()),
            (input_dim // 5, nn.GELU()),
            (input_dim // 6, nn.GELU()),
        )

        self.regressor = nn.Linear(latent_dim, 2)

        self.decoder = FullyConnectedNetwork(
            latent_dim, input_dim,
            (input_dim // 6, nn.GELU()),
            (input_dim // 5, nn.GELU()),
            (input_dim // 4, nn.GELU()),
            (input_dim // 3, nn.GELU()),
            (input_dim // 2, nn.GELU()),
        )

    def forward_internal(self, x: torch.Tensor):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        charges = self.regressor(z)
        return x_recon, charges, z

    # For SHAP value calculation
    def forward(self, x: torch.Tensor):
        x_recon, charges, z = self.forward_internal(x)
        return z


class SCIVariationalAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super(SCIVariationalAutoencoder, self).__init__()

        self.encoder = FullyConnectedNetwork(
            input_dim, input_dim // 6,
            (input_dim // 2, nn.GELU()),
            (input_dim // 3, nn.GELU()),
            (input_dim // 4, nn.GELU()),
            (input_dim // 5, nn.GELU()),
        )
        self.fc_elu = nn.GELU()
        self.fc_mu = nn.Linear(input_dim // 6, latent_dim)
        self.fc_logvar = nn.Linear(input_dim // 6, latent_dim)

        self.regressor = nn.Linear(latent_dim, 2)

        self.decoder = FullyConnectedNetwork(
            latent_dim, input_dim,
            (input_dim // 6, nn.GELU()),
            (input_dim // 5, nn.GELU()),
            (input_dim // 4, nn.GELU()),
            (input_dim // 3, nn.GELU()),
            (input_dim // 2, nn.GELU()),
        )

    def encode(self, x: torch.Tensor):
        h = self.fc_elu(self.encoder(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        return self.decoder(z)

    def forward_internal(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        charges = self.regressor(z)
        return x_recon, charges, mu, logvar

    # For SHAP value calculation
    def forward(self, x: torch.Tensor):
        x_recon, charges, mu, logvar = self.forward_internal(x)
        return self.reparameterize(mu, logvar)


class AEReconLoss(nn.Module):
    def __init__(self, kde_shape_weight: float=1.0):
        super(AEReconLoss, self).__init__()
        self.kde_shape_criterion = CorrelationLoss()
        self.kde_value_criterion = nn.MSELoss()

        self.kde_shape_weight = kde_shape_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor):

        loss_kde_shape = self.kde_shape_criterion(pred, target)
        loss_kde_value = self.kde_value_criterion(pred, target)

        return self.kde_shape_weight * loss_kde_shape + loss_kde_value


def train_ae(loader: DataLoader, model: SCIAutoencoder, recon_loss_fn: nn.Module, charge_loss_fn: nn.Module,
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


def train_vae(loader: DataLoader, model: SCIVariationalAutoencoder, recon_loss_fn: nn.Module, charge_loss_fn: nn.Module,
             optimizer: Optimizer, device: torch.device,
             loss_charge_weight: float = 0.1, c: float=0.01):
    model.train()

    total_loss = 0
    train_cnt = 0
    for x, charge in loader:
        x = x.to(device)
        charge = charge.to(device)

        x_recon, charge_pred, _, _ = model.forward_internal(x)

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


def test_ae(loader: DataLoader, model: SCIAutoencoder, recon_loss_fn: nn.Module, charge_loss_fn: nn.Module,
            device: torch.device, loss_charge_weight: float = 0.1):
    model.eval()
    test_loss = 0.0
    recon_corr = 0.0
    recon_mse = 0.0
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

            corr = CorrelationLoss()

            recon_corr += (1 - corr(x_recon, x).item()) * x.size(0)
            recon_mse += torch.mean((x_recon - x) ** 2).item() * x.size(0)
            recon_cnt += x.size(0)

            charge_pred = charge_pred.cpu().numpy()
            charge = charge.cpu().numpy()

            err = np.concatenate(np.abs((charge_pred - charge) / charge))
            charge_error += np.sum(err)
            charge_cnt += len(err)

    return test_loss, recon_corr, recon_mse, charge_error, test_cnt, recon_cnt, charge_cnt


def test_vae(loader: DataLoader, model: SCIVariationalAutoencoder, recon_loss_fn: nn.Module, charge_loss_fn: nn.Module,
            device: torch.device, loss_charge_weight: float = 0.1):
    model.eval()
    test_loss = 0.0
    recon_corr = 0.0
    recon_mse = 0.0
    charge_error = 0.0
    test_cnt = 0
    recon_cnt = 0
    charge_cnt = 0

    with torch.no_grad():
        for x, charge in loader:
            x = x.to(device)
            charge = charge.to(device)

            x_recon, charge_pred, _, _ = model.forward_internal(x)

            loss_recon = recon_loss_fn(x_recon, x)
            loss_charge = charge_loss_fn(charge_pred, charge)

            loss = loss_recon + loss_charge_weight * loss_charge

            test_loss += loss.item() * x.size(0)
            test_cnt += x.size(0)

            corr = CorrelationLoss()

            recon_corr += (1 - corr(x_recon, x).item()) * x.size(0)
            recon_mse += torch.mean((x_recon - x) ** 2).item() * x.size(0)
            recon_cnt += x.size(0)

            charge_pred = charge_pred.cpu().numpy()
            charge = charge.cpu().numpy()

            err = np.concatenate(np.abs((charge_pred - charge) / charge))
            charge_error += np.sum(err)
            charge_cnt += len(err)

    return test_loss, recon_corr, recon_mse, charge_error, test_cnt, recon_cnt, charge_cnt


class AEValidationData():
    def __init__(self, a_real: np.ndarray, a_pred: np.ndarray,
                 c_real: np.ndarray, c_pred: np.ndarray,
                 x: np.ndarray, x_recon: np.ndarray,
                 r2_a: float, r2_c: float,
                 rmse_a: float, rmse_c: float,
                 error_a: float, error_c: float,
                 recon_corr: float, recon_rmse: float):
        import shap
        self.shap_values: shap.Explanation = None
        self.a_real = a_real
        self.a_pred = a_pred
        self.c_real = c_real
        self.c_pred = c_pred
        self.x = x
        self.x_recon = x_recon
        self.r2_a = r2_a
        self.r2_c = r2_c
        self.rmse_a = rmse_a
        self.rmse_c = rmse_c
        self.error_a = error_a
        self.error_c = error_c
        self.recon_corr = recon_corr
        self.recon_rmse = recon_rmse


def validate_ae(x: torch.Tensor, charge: torch.Tensor, model: SCIAutoencoder, device: torch.device, calculate_shap: bool=False):
    import shap
    model.eval()
    charge_real = None
    charge_pred = None

    with torch.no_grad():
        x = x.to(device)
        charge_real = charge.cpu().numpy()

        x_recon, charge_pred, _ = model.forward_internal(x)
        charge_pred = charge_pred.cpu().numpy()

    a_real = charge_real[:, 0].ravel()
    a_pred = charge_pred[:, 0].ravel()

    c_real = charge_real[:, 1].ravel()
    c_pred = charge_pred[:, 1].ravel()

    r2_a = r2_score(a_real, a_pred)
    r2_c = r2_score(c_real, c_pred)

    rmse_a = root_mean_squared_error(a_real, a_pred)
    rmse_c = root_mean_squared_error(c_real, c_pred)

    error_a = float(np.sum(np.abs((a_real - a_pred) / a_real)) / len(a_real))
    error_c = float(np.sum(np.abs((c_real - c_pred) / c_real)) / len(c_real))

    corr = CorrelationLoss()

    recon_corr = (1 - corr(x_recon, x).item())
    recon_rmse = torch.sqrt(torch.mean((x_recon - x) ** 2)).item()

    print("Validation stats")
    print("=======================")
    print(f"    R2 of a: {r2_a:.4f}")
    print(f"    RMSE of a: {rmse_a:.4f}")
    print(f"    Error of a: {error_a:.4f}")
    print(f"    R2 of c: {r2_c:.4f}")
    print(f"    RMSE of c: {rmse_c:.4f}")
    print(f"    Error of c: {error_c:.4f}")
    print(f"    Correlation of reconstruction: {recon_corr:.4f}")
    print(f"    RMSE of reconstruction: {recon_rmse:.4f}")

    shap_values = None

    if calculate_shap:
        print("Calculating SHAP values...")
        explainer = shap.DeepExplainer(model, x)
        shap_values = explainer(x)
        shap_values.data = shap_values.data.detach().cpu().numpy()

    x_index = random.randint(0, x.size(0) - 1)
    x = x[x_index, :].cpu().numpy().ravel()
    x_recon = x_recon[x_index, :].cpu().numpy().ravel()

    validation_data = AEValidationData(a_real, a_pred, c_real, c_pred, x, x_recon,
                            r2_a, r2_c, rmse_a, rmse_c, error_a, error_c,
                            recon_corr, recon_rmse)
    if calculate_shap:
        validation_data.shap_values = shap_values

    return validation_data


def validate_vae(x: torch.Tensor, charge: torch.Tensor, model: SCIAutoencoder, device: torch.device, calculate_shap: bool=False):
    import shap
    model.eval()
    charge_real = None
    charge_pred = None

    with torch.no_grad():
        x = x.to(device)
        charge_real = charge.cpu().numpy()

        x_recon, charge_pred, _ = model.forward_internal(x)
        charge_pred = charge_pred.cpu().numpy()

    a_real = charge_real[:, 0].ravel()
    a_pred = charge_pred[:, 0].ravel()

    c_real = charge_real[:, 1].ravel()
    c_pred = charge_pred[:, 1].ravel()

    r2_a = r2_score(a_real, a_pred)
    r2_c = r2_score(c_real, c_pred)

    rmse_a = root_mean_squared_error(a_real, a_pred)
    rmse_c = root_mean_squared_error(c_real, c_pred)

    error_a = float(np.sum(np.abs((a_real - a_pred) / a_real)) / len(a_real))
    error_c = float(np.sum(np.abs((c_real - c_pred) / c_real)) / len(c_real))

    corr = CorrelationLoss()

    recon_corr = (1 - corr(x_recon, x).item())
    recon_rmse = torch.sqrt(torch.mean((x_recon - x) ** 2)).item()

    print("Validation stats")
    print("=======================")
    print(f"    R2 of a: {r2_a:.4f}")
    print(f"    RMSE of a: {rmse_a:.4f}")
    print(f"    Error of a: {error_a:.4f}")
    print(f"    R2 of c: {r2_c:.4f}")
    print(f"    RMSE of c: {rmse_c:.4f}")
    print(f"    Error of c: {error_c:.4f}")
    print(f"    Correlation of reconstruction: {recon_corr:.4f}")
    print(f"    RMSE of reconstruction: {recon_rmse:.4f}")

    shap_values = None

    if calculate_shap:
        print("Calculating SHAP values...")
        explainer = shap.DeepExplainer(model, x)
        shap_values = explainer(x)

    x_index = random.randint(0, x.size(0) - 1)
    x = x[x_index, :].cpu().numpy().ravel()
    x_recon = x_recon[x_index, :].cpu().numpy().ravel()

    validation_data = AEValidationData(a_real, a_pred, c_real, c_pred, x, x_recon,
                                       r2_a, r2_c, rmse_a, rmse_c, error_a, error_c,
                                       recon_corr, recon_rmse)
    if calculate_shap:
        validation_data.shap_values = shap_values

    return validation_data


if __name__ == "__main__":
    import shap
    print(sys.version)
    print("GIL enabled:", sys._is_gil_enabled())

    os.makedirs('../data/unsupervised', exist_ok=True)
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

    print("Whole data stats")
    theory_sampler.get_bins_stats(central_charge, min_charge, max_charge, n_bins).show(n=n_bins, truncate=False)

    train_ratio = float(input("Enter train ratio: "))
    train_sampler, test_sampler = theory_sampler.get_train_test_sets_bins(central_charge, min_charge, max_charge,
                                                                          n_bins, train_ratio)
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
    print(f"Device: {device}")

    input_num = input_train[0].shape[1]
    latent_dim = int(input("Enter the latent dimension: "))
    model = None

    train = None
    test = None
    validate = None

    model_name = int(input("Enter which model to use; 1. autoencoder, 2. variational autoencoder\n>>>"))
    if model_name == 1:
        model = SCIAutoencoder(input_num, latent_dim).to(device)
        train = train_ae
        test = test_ae
        validate = validate_ae
        model_name = "Autoencoder"
    elif model_name == 2:
        model = SCIVariationalAutoencoder(input_num, latent_dim).to(device)
        train = train_vae
        test = test_vae
        validate = validate_vae
        model_name = "Variational Autoencoder"
    else:
        print("Invalid model")
        exit(-1)

    recon_criterion = AEReconLoss(kde_shape_weight=50.0)
    charge_criterion = nn.MSELoss()
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

    loss_charge_weight = 50.0

    for epoch in range(n_epochs):
        print(f"Train epoch {epoch + 1}...")
        for i in range(n_train):
            train(dataloader_train[i], model, recon_criterion, charge_criterion, optimizer, device,
                     loss_charge_weight=loss_charge_weight, c=0.00001)
            print(f"Training set {i + 1}/{n_train} complete.")

        print(f"Test epoch {epoch + 1}...")
        total_loss = 0.0
        total_recon_corr = 0.0
        total_recon_mse = 0.0
        total_charge_error = 0.0
        total_cnt = 0
        total_recon_cnt = 0
        total_charge_cnt = 0

        for i in range(n_test):
            loss, recon_corr, recon_mse, charge_error, cnt, recon_cnt, charge_cnt = test(
                dataloader_test[i], model, recon_criterion, charge_criterion, device,
                loss_charge_weight=loss_charge_weight
            )
            total_loss += loss
            total_recon_corr += recon_corr
            total_recon_mse += recon_mse
            total_charge_error += charge_error
            total_cnt += cnt
            total_recon_cnt += recon_cnt
            total_charge_cnt += charge_cnt
            print(f"Test set {i + 1}/{n_test} complete.")

        total_loss /= total_cnt
        total_recon_corr /= total_recon_cnt
        total_recon_mse /= total_recon_cnt
        total_charge_error /= total_charge_cnt
        print(f"Epoch {epoch + 1}/{n_epochs} test loss: {total_loss}")
        print(f"reconstruction correlation: {total_recon_corr:.4f} reconstruction rmse: {math.sqrt(total_recon_mse):.4f} charge error: {total_charge_error * 100:.2f} %")
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

    r2_scores = [[], []]
    rmse_scores = [[], []]
    errors = [[], []]
    recon_corrs = []
    recon_rmse_scores = []
    save_dir = f"../data/unsupervised/{datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}"
    os.makedirs(save_dir, exist_ok=True)

    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 15

    for i in range(n_validate):
        validation_data = validate(
            torch.tensor(input_validate[i], dtype=torch.float32),
            torch.tensor(output_validate[i], dtype=torch.float32),
            model, device, i == 0
        )

        if i == 0:
            validation_data.shap_values.feature_names = [f"{GRID[i]:.4f}" for i in range(len(GRID))]
            for j in range(latent_dim):
                shap_latent = validation_data.shap_values[:,:,j]

                plt.close("all")
                fig, ax = plt.subplots(figsize=(24, 16))
                ax.set_title(f"SHAP values for latent vector element {j + 1}")
                shap.plots.beeswarm(shap_latent, max_display=15, show=False, color_bar=True,
                                    ax=ax, group_remaining_features=True, plot_size=None)

                plt.savefig(f"{save_dir}/sci_charge_ae_{model_name}_latent{j + 1}_shap.png")

                plt.close("all")
                fig, ax = plt.subplots(figsize=(24, 16))
                ax.set_title(f"Absolute average SHAP values for latent vector element {i + 1}")
                shap.plots.bar(shap_latent, max_display=15, show=False, ax=ax)

                plt.savefig(f"{save_dir}/sci_charge_ae_{model_name}_latent{j + 1}_shap_abs_avg.png")


        r2_scores[0].append(validation_data.r2_a)
        r2_scores[1].append(validation_data.r2_c)
        rmse_scores[0].append(validation_data.rmse_a)
        rmse_scores[1].append(validation_data.rmse_c)
        errors[0].append(validation_data.error_a * 100.0)
        errors[1].append(validation_data.error_c * 100.0)
        recon_corrs.append(validation_data.recon_corr)
        recon_rmse_scores.append(validation_data.recon_rmse)

        plt.close("all")

        fig, ax = plt.subplots(1, 2, sharey=True, squeeze=True)
        fig.suptitle(f"SCI {model_name} central charge regression")

        ax[0].set_title(f"a R2 = {validation_data.r2_a:.3f}, RMSE = {validation_data.rmse_a:.3f}")
        ax[0].scatter(validation_data.a_real, validation_data.a_pred, color="blue")
        y_range = [np.min(validation_data.a_real), np.max(validation_data.a_real)]
        ax[0].plot(y_range, y_range, linestyle="--", color="red")
        ax[0].set_xlabel("Real a")
        ax[0].set_ylabel("Predicted a")

        ax[1].set_title(f"c R2 = {validation_data.r2_c:.3f}, RMSE = {validation_data.rmse_c:.3f}")
        ax[1].scatter(validation_data.c_real, validation_data.c_pred, color="blue")
        y_range = [np.min(validation_data.c_real), np.max(validation_data.c_real)]
        ax[1].plot(y_range, y_range, linestyle="--", color="red")
        ax[1].set_xlabel("Real c")
        ax[1].set_ylabel("Predicted c")

        plt.savefig(f"{save_dir}/sci_charge_ae_{model_name}_charge_regression_{i + 1}.png")

        plt.close("all")

        fig, ax = plt.subplots()
        fig.suptitle(f"SCI {model_name} KDE data reconstruction")
        ax.plot(GRID, validation_data.x, color="blue", label="Real")
        ax.plot(GRID, validation_data.x_recon, color="red", label="Reconstructed")
        ax.fill_between(GRID, validation_data.x, validation_data.x_recon, color="lightcoral", label="Error")
        ax.set_xlabel("Operator dimension")
        ax.set_ylabel("Approximate operator number")
        ax.legend()

        plt.savefig(f"{save_dir}/sci_charge_ae_{model_name}_kde_reconstruction_{i + 1}.png")

    with open(f"{save_dir}/sci_charge_ae_{model_name}_{file_suffix}.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Score"] + [i + 1 for i in range(n_validate)])
        writer.writerow(["R2 of a"] + r2_scores[0])
        writer.writerow(["R2 of c"] + r2_scores[1])
        writer.writerow(["RMSE of a"] + rmse_scores[0])
        writer.writerow(["RMSE of c"] + rmse_scores[1])
        writer.writerow(["Error of a (%)"] + errors[0])
        writer.writerow(["Error of c (%)"] + errors[1])
        writer.writerow(["Reconstruction correlation"] + recon_corrs)
        writer.writerow(["Reconstruction RMSE"] + recon_rmse_scores)

    with open(f"{save_dir}/sci_charge_ae_{model_name}_{file_suffix}_charge_eq.csv", 'w', newline='') as f:
        weight = model.regressor.weight.detach().cpu().numpy().tolist()
        bias = model.regressor.bias.detach().cpu().numpy().tolist()

        writer = csv.writer(f)
        writer.writerow(["Charge"] + [f"x{i + 1}" for i in range(latent_dim)] + ["bias"])
        writer.writerow(["a"] + weight[0] + [bias[0]])
        writer.writerow(["c"] + weight[1] + [bias[1]])
