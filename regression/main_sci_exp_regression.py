import datetime
import sys
import os
import csv
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
import matplotlib.pyplot as plt
import joblib

from common.balanced_sample_tool import TheorySampler
from common.sci_parser import SuperConformalIndex


def build_data(sampler: TheorySampler, charge_col: str, min_charge: float, max_charge: float,
               n_bins: int, n_per_bins: int, n_validate: int,
               grid: np.ndarray, kde_bandwidth: float):
    input_train = []
    output_train = []
    input_validate = []
    output_validate = []

    def build_dataset(data: TheorySampler, input_set, output_set):
        rows = data.df.collect()

        input_data = []
        output_data = []
        for row in rows:
            input_data.append(SuperConformalIndex(row["SCI"]).featurize_dimensions(grid, kde_bandwidth))
            output_data.append(float(row[charge_col]))

        input_data = np.stack(input_data)
        output_data = np.asarray(output_data)

        input_set.append(input_data)
        output_set.append(output_data)

    build_dataset(sampler.get_balanced_bins_sample(charge_col, min_charge, max_charge, n_bins, n_per_bins),
                  input_train, output_train)
    print(f"Train data built.")

    for i in range(n_validate):
        build_dataset(sampler.get_balanced_bins_sample(charge_col, min_charge, max_charge, n_bins, n_per_bins),
                      input_validate, output_validate)
        print(f"Validation data {i + 1} built.")

    return input_train, output_train, input_validate, output_validate


def fit_data(x: np.ndarray, y: np.ndarray, model: GridSearchCV):
    model.fit(x, y)

    print("Regression of the data")
    print("=============================")
    print(f"    best params: {model.best_params_}")
    print(f"    best score: {model.best_score_:.4f}")
    print()

def validate(x: np.ndarray, y: np.ndarray, model: GridSearchCV):
    y_pred = model.predict(x)

    r2 = r2_score(y, y_pred)
    error = np.sum(np.abs(y - y_pred) / y)
    error /= len(y)
    rmse = root_mean_squared_error(y, y_pred)

    return y_pred, r2, rmse, error


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

    n_validate = int(input("Enter number of validation samples: "))

    input_train, output_train, input_validate, output_validate = build_data(
        theory_sampler, central_charge, min_charge, max_charge, n_bins, n_per_bins,
        n_validate, GRID, KDE_BANDWIDTH
    )

    krr = KernelRidge(kernel="rbf")
    krr_grid = {
        "alpha": np.logspace(-4, 0, 9),
        "gamma": np.logspace(-5, -1, 9),
    }

    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    krr_search = GridSearchCV(
        krr,
        krr_grid,
        cv=cv,
        scoring="r2",
        n_jobs=-1
    )

    file_suffix = f"{central_charge[-1].lower()}_{GRID_LO}_{GRID_HI}_{GRID_STEP}_{KDE_BANDWIDTH}"

    r2_scores = []
    rmse_scores = []
    errors = []
    save_dir = f"../data/regression/{datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}"
    os.makedirs(save_dir, exist_ok=True)

    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 15

    fit_data(input_train[0], output_train[0], krr_search)

    for i in range(n_validate):
        x = input_validate[i]
        y_real = output_validate[i]

        y_pred, r2, rmse, error = validate(x, y_real, krr_search)
        r2_scores.append(r2)
        rmse_scores.append(rmse)
        errors.append(f"{error * 100.0:.2f}%")

        print(f"Validation set {i + 1} result")
        print("=======================================")
        print(f"    R2: {r2:.4f}")
        print(f"    RMSE: {rmse:.4f}")
        print(f"    Error: {error * 100.0:.2f}%")
        print()

        plt.close('all')

        fig, ax = plt.subplots()
        fig.suptitle(f"Kernel Ridge Regression of central charge {central_charge[-1].lower()}")

        ax.set_title(f"R2 = {r2:.3f}, RMSE = {rmse:.3f}")
        ax.scatter(y_real, y_pred)
        y_range = [np.min(y_real), np.max(y_real)]
        ax.plot(y_range, y_range, linestyle='--', color='red')
        ax.set_xlabel(f"Real {central_charge[-1].lower()}")
        ax.set_ylabel(f"Predicted {central_charge[-1].lower()}")

        plt.savefig(f"{save_dir}/sci_exp_regression_{central_charge[-1].lower()}_{i + 1}.png")

    with open(f"{save_dir}/sci_exp_regression_{file_suffix}.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Score"] + [i + 1 for i in range(n_validate)])
        writer.writerow(["R2"] + r2_scores)
        writer.writerow(["RMSE"] + rmse_scores)
        writer.writerow(["Error"] + errors)

    joblib.dump(krr_search.best_estimator_, f"{save_dir}/best_krr_model.pkl")
