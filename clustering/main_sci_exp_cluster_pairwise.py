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


data_per_theories = dict()


def build_data(sampler: TheorySampler, n_per_theory: int, n_iter: int, grid: np.ndarray, kde_bandwidth: float):
    stats = sampler.get_theory_stats()
    theories = stats.select("Name").rdd.flatMap(lambda x: x).collect()
    for theory in theories:
        data_per_theories[theory] = [[] for _ in range(n_iter)]

    for i in range(n_iter):
        sampled_data = sampler.get_manual_sample(theories, n_per_theory)
        rows = sampled_data.df.collect()

        for row in rows:
            data_per_theories[row["Name"]][i].append(SuperConformalIndex(row["SCI"]).featurize_dimensions(grid, kde_bandwidth))

        print(f"Data generation iteration {i + 1} completed.")

    for theory in theories:
        for i in range(n_iter):
            data_per_theories[theory][i] = np.stack(data_per_theories[theory][i])


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

if __name__ == '__main__':
    os.environ["PYSPARK_PYTHON"] = "/home/subo-lee/PycharmProjects/LandscapeML/.venv/bin/python3.14t"
    os.environ["PYSPARK_DRIVER_PYTHON"] = "/home/subo-lee/PycharmProjects/LandscapeML/.venv/bin/python3.14t"

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

    accuracy = [[0.0 for _ in range(n_theory)] for _ in range(n_theory)]

    pairs = [(theories[i], theories[j]) for i in range(n_theory) for j in range(i + 1, n_theory)]
    print(f"Number of pairs: {len(pairs)}")

    build_data(cutoff_sampler, n_sample, n_iter, GRID, KDE_BANDWIDTH)

    def pair_calculation(theory1: str, theory2: str):
        print(f"Calculating accuracy for {theory1} and {theory2}")
        acc = []

        for i in range(n_iter):
            acc.append(cluster_pair(data_per_theories[theory1][i], data_per_theories[theory2][i]))

        mean_acc = float(np.mean(acc))
        accuracy[theories_dict[theory1]][theories_dict[theory2]] = mean_acc
        accuracy[theories_dict[theory2]][theories_dict[theory1]] = mean_acc
        print(f"Accuracy for {theory1} and {theory2}: {mean_acc}")

    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        executor.map(lambda arg: pair_calculation(*arg), pairs)

    save_dir = f"../data/clustering/{datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}"
    os.makedirs(save_dir, exist_ok=True)

    with open(f"{save_dir}/sci_exp_cluster_pairwise.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Theories"] + theories)
        for i in range(n_theory):
            writer.writerow([theories[i]] + accuracy[i])

    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 15

    fig, ax = plt.subplots()
    fig.suptitle("Pairwise clustering accuracies")

    cmap_plot = ax.matshow(accuracy, cmap='gray')
    fig.colorbar(cmap_plot, ax=ax)

    plt.savefig(f'{save_dir}/sci_exp_cluster_pairwise.png')
