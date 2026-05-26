import datetime

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from common.balanced_sample_tool import TheorySampler
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import csv
import sys
import os.path
import os
import math
import json
import pathlib
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from common.sci_parser import SuperConformalIndex


def build_data(sampler: TheorySampler, n_per_theory: int, n_iter: int, grid: np.ndarray, kde_bandwidth: float):
    exp_data_per_theories = dict()
    ac_data_per_theories = dict()

    stats = sampler.get_theory_stats()
    theories = stats.select("Name").rdd.flatMap(lambda x: x).collect()
    for theory in theories:
        exp_data_per_theories[theory] = [[] for _ in range(n_iter)]
        ac_data_per_theories[theory] = [[] for _ in range(n_iter)]

    for i in range(n_iter):
        sampled_data = sampler.get_manual_sample(theories, n_per_theory)
        rows = sampled_data.df.collect()

        for row in rows:
            exp_data_per_theories[row["Name"]][i].append(SuperConformalIndex(row["SCI"]).featurize_dimensions(grid, kde_bandwidth))
            ac_data_per_theories[row["Name"]][i].append([float(row["CentralChargeA"]), float(row["CentralChargeC"])])

        print(f"Data generation iteration {i + 1} completed.")

    for theory in theories:
        for i in range(n_iter):
            exp_data_per_theories[theory][i] = np.stack(exp_data_per_theories[theory][i])
            ac_data_per_theories[theory][i] = np.stack(ac_data_per_theories[theory][i])

    return exp_data_per_theories, ac_data_per_theories


def cluster_pair(theory1_input: np.ndarray, theory2_input: np.ndarray):
    data_num = theory1_input.shape[0]
    X = np.vstack((theory1_input, theory2_input))
    y_true = np.hstack((np.zeros(data_num, dtype=int), np.ones(data_num, dtype=int)), dtype=int)

    Xs = StandardScaler().fit_transform(X)

    reduction_model = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        random_state=42
    )
    X_tsne = reduction_model.fit_transform(Xs)

    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    kmeans.fit(X_tsne)
    y_pred = kmeans.labels_

    acc_direct = np.mean(y_pred == y_true)
    acc_flipped = np.mean(y_pred == (1 - y_true))

    return max(acc_direct, acc_flipped)


def cluster_pair_no_tsne(theory1_input: np.ndarray, theory2_input: np.ndarray):
    data_num = theory1_input.shape[0]
    X = np.vstack((theory1_input, theory2_input))
    y_true = np.hstack((np.zeros(data_num, dtype=int), np.ones(data_num, dtype=int)), dtype=int)

    Xs = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    kmeans.fit(Xs)
    y_pred = kmeans.labels_

    acc_direct = np.mean(y_pred == y_true)
    acc_flipped = np.mean(y_pred == (1 - y_true))

    return max(acc_direct, acc_flipped)


if __name__ == '__main__':
    # os.environ["PYSPARK_PYTHON"] = "/home/subo-lee/PycharmProjects/LandscapeML/.venv/bin/python3.14t"
    # os.environ["PYSPARK_DRIVER_PYTHON"] = "/home/subo-lee/PycharmProjects/LandscapeML/.venv/bin/python3.14t"

    print(sys.version)
    print("GIL enabled:", sys._is_gil_enabled())

    os.makedirs('../data/clustering', exist_ok=True)
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

    min_count = int(input("Enter the minimum number of fixed points of theories to choose: "))
    n_sample = int(input("Enter the number of samples from each theories: "))
    n_iter = int(input("Enter the number of iterations for each pairs: "))

    cutoff_sampler = theory_sampler.get_theories_above_count(min_count)
    theories = cutoff_sampler.get_theory_stats().select("Name").rdd.flatMap(lambda x: x).collect()
    n_theory = len(theories)
    print(f"Selected theories: {n_theory}")
    print(theories)

    theories_dict = dict()
    for i in range(n_theory):
        theories_dict[theories[i]] = i

    accuracy_exp = [[0.0 for _ in range(n_theory)] for _ in range(n_theory)]
    accuracy_ac = [[0.0 for _ in range(n_theory)] for _ in range(n_theory)]

    pairs = [(theories[i], theories[j]) for i in range(n_theory) for j in range(i + 1, n_theory)]
    print(f"Number of pairs: {len(pairs)}")

    exp_data_per_theory, ac_data_per_theory = build_data(cutoff_sampler, n_sample, n_iter, GRID, KDE_BANDWIDTH)

    def pair_calculation(theory1: str, theory2: str):
        print(f"Calculating accuracy for {theory1} and {theory2}")
        acc_exp = []
        acc_ac = []

        for i in range(n_iter):
            acc_exp.append(cluster_pair(exp_data_per_theory[theory1][i], exp_data_per_theory[theory2][i]))
            acc_ac.append(cluster_pair_no_tsne(ac_data_per_theory[theory1][i], ac_data_per_theory[theory2][i]))

        mean_acc_exp = float(np.mean(acc_exp))
        accuracy_exp[theories_dict[theory1]][theories_dict[theory2]] = mean_acc_exp
        accuracy_exp[theories_dict[theory2]][theories_dict[theory1]] = mean_acc_exp
        print(f"Accuracy for {theory1} and {theory2} with exponent clustering: {mean_acc_exp}")

        mean_acc_ac = float(np.mean(acc_ac))
        accuracy_ac[theories_dict[theory1]][theories_dict[theory2]] = mean_acc_ac
        accuracy_ac[theories_dict[theory2]][theories_dict[theory1]] = mean_acc_ac
        print(f"Accuracy for {theory1} and {theory2} with a/c clustering: {mean_acc_ac}")

    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        executor.map(lambda arg: pair_calculation(*arg), pairs)

    save_dir = f"../data/clustering/{datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}"
    os.makedirs(save_dir, exist_ok=True)

    with open(f"{save_dir}/sci_exp_cluster_pairwise_exponents.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Theories"] + theories)
        for i in range(n_theory):
            writer.writerow([theories[i]] + accuracy_exp[i])

    with open(f"{save_dir}/sci_exp_cluster_pairwise_ac.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Theories"] + theories)
        for i in range(n_theory):
            writer.writerow([theories[i]] + accuracy_ac[i])

    accuracy_diff = np.abs(np.asarray(accuracy_exp) - np.asarray(accuracy_ac))

    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (24, 12)
    plt.rcParams['font.size'] = 12

    fig, ax = plt.subplots(nrows=1, ncols=3, squeeze=True)
    fig.suptitle("Pairwise clustering accuracies")

    vmin, vmax = 0.5, 1.0

    ax[0].set_title("Exponent clustering")
    cmap_plot = ax[0].matshow(accuracy_exp, vmin=vmin, vmax=vmax, cmap='gray')
    fig.colorbar(cmap_plot, ax=ax[0], shrink=0.7)

    ax[1].set_title("a/c clustering")
    cmap_plot = ax[1].matshow(accuracy_ac, vmin=vmin, vmax=vmax, cmap='gray')
    fig.colorbar(cmap_plot, ax=ax[1], shrink=0.7)

    ax[2].set_title("Accuracy difference")
    cmap_plot = ax[2].matshow(accuracy_diff, cmap='gray')
    fig.colorbar(cmap_plot, ax=ax[2], shrink=0.7)

    plt.savefig(f'{save_dir}/sci_exp_cluster_pairwise.png')
    plt.show()
