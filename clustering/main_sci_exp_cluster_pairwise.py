import datetime
import random

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
from clustering_common import cluster_pair, calculate_feature_importance


def build_data(sampler: TheorySampler, n_per_theory: int, n_iter: int, grid: np.ndarray, kde_bandwidth: float):
    data_per_theories = dict()

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

    return data_per_theories


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

    accuracy = [[0.0 for _ in range(n_theory)] for _ in range(n_theory)]
    stdev = [[0.0 for _ in range(n_theory)] for _ in range(n_theory)]

    pairs = [(theories[i], theories[j]) for i in range(n_theory) for j in range(i + 1, n_theory)]
    print(f"Number of pairs: {len(pairs)}")

    feature_impact_scores = [None] * len(pairs)
    pair_dict = dict(zip(pairs, [i for i in range(len(pairs))]))

    data_per_theory = build_data(cutoff_sampler, n_sample, n_iter, GRID, KDE_BANDWIDTH)

    def pair_calculation(theory1: str, theory2: str):
        print(f"Calculating accuracy for {theory1} and {theory2}")
        acc = []

        feature_impact_iter = random.randrange(0, n_iter)
        for i in range(n_iter):
            if i == feature_impact_iter:
                acc_score, feature_impact = calculate_feature_importance(data_per_theory[theory1][i], data_per_theory[theory2][i])
                acc.append(acc_score)
                feature_impact_scores[pair_dict[(theory1, theory2)]] = feature_impact
            else:
                acc_score, _ = cluster_pair(data_per_theory[theory1][i], data_per_theory[theory2][i])
                acc.append(acc_score)

        mean_acc= float(np.mean(acc))
        stdev_acc = float(np.std(acc))
        accuracy[theories_dict[theory1]][theories_dict[theory2]] = mean_acc
        accuracy[theories_dict[theory2]][theories_dict[theory1]] = mean_acc
        stdev[theories_dict[theory1]][theories_dict[theory2]] = stdev_acc
        stdev[theories_dict[theory2]][theories_dict[theory1]] = stdev_acc
        print(f"Stats for {theory1} and {theory2} with exponent clustering")
        print(f"    mean: {mean_acc}, stdev: {stdev_acc}")

    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        executor.map(lambda arg: pair_calculation(*arg), pairs)

    feature_impact_scores = np.stack(feature_impact_scores)
    feature_impact_scores = np.mean(np.abs(feature_impact_scores), axis=0)
    feature_names = [f"{GRID[i]:.4f}" for i in range(len(GRID))] + [
        "Minimal gap",
        "Maximal gap",
        "Mean gap",
        "Stdev gap",
        "Median gap",
        "1st quartile gap",
        "3rd quartile gap",
        "Number of unique dimensions",
        "Log of number of unique dimensions",
        "Mean dimension",
        "Stdev dimension",
        "Minimal dimension",
        "Maximal dimension",
        "Median dimension",
        "1st quartile dimension",
        "3rd quartile dimension"
    ]

    save_dir = f"../data/clustering/{datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}"
    os.makedirs(save_dir, exist_ok=True)

    with open(f"{save_dir}/sci_exp_cluster_pairwise_acc.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Theories"] + theories)
        for i in range(n_theory):
            writer.writerow([theories[i]] + accuracy[i])

    with open(f"{save_dir}/sci_exp_cluster_pairwise_stdev.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Theories"] + theories)
        for i in range(n_theory):
            writer.writerow([theories[i]] + stdev[i])

    with open(f"{save_dir}/sci_exp_cluster_pairwise_impact.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Features"] + feature_names)
        writer.writerow(["Scores"] + feature_impact_scores.tolist())

    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 12

    fig, ax = plt.subplots(nrows=1, ncols=2, squeeze=True)
    fig.suptitle("Pairwise clustering accuracies and stdevs")

    vmin, vmax = 0.5, 1.0

    ax[0].set_title("Accuracies")
    cmap_plot = ax[0].matshow(accuracy, vmin=vmin, vmax=vmax, cmap='gray')
    fig.colorbar(cmap_plot, ax=ax[0], shrink=0.7)

    ax[1].set_title(f"Stdevs iteration: {n_iter}")
    cmap_plot = ax[1].matshow(stdev, vmin=0.0, vmax=np.max(stdev), cmap='gray_r')
    fig.colorbar(cmap_plot, ax=ax[1], shrink=0.7)

    plt.savefig(f'{save_dir}/sci_exp_cluster_pairwise.png')
    plt.show()
