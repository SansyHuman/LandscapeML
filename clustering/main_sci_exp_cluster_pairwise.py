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



def cluster_pair(sampled: TheorySampler, grid: np.ndarray, kde_bandwidth: float):
    sample_stat = sampled.get_theory_stats()

    n_theory = sampled.get_theory_num()
    assert n_theory == 2

    theories = sample_stat.select("Name").rdd.flatMap(lambda x: x).collect()

    theories_dict = dict()
    for i in range(len(theories)):
        theories_dict[theories[i]] = i

    data_num = sampled.df.count()
    theory_data = []
    sci_data: list[SuperConformalIndex] = []

    rows = sampled.df.collect()

    for i in range(data_num):
        theory_data.append(theories_dict[rows[i]["Name"]])
        sci_data.append(SuperConformalIndex(rows[i]["SCI"]))

    X = np.stack([sci_data[i].featurize_dimensions(grid, kde_bandwidth) for i in range(data_num)])
    y_true = np.asarray(theory_data)

    Xs = StandardScaler().fit_transform(X)

    reduction_model = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        random_state=42
    )
    X_tsne = reduction_model.fit_transform(Xs)

    kmeans = KMeans(n_clusters=n_theory, n_init=10, random_state=42)
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

    theories_df = theory_sampler.spark.createDataFrame([(t,) for t in theories], ["theory"])
    pairs_df = theories_df.alias("t1").crossJoin(theories_df.alias("t2")).filter(F.col("t1.theory") < F.col("t2.theory"))
    print(f"Number of pairs: {pairs_df.count()}")

    def pair_calculation(row):
        theory1, theory2 = row["t1.theory"], row["t2.theory"]
        print(theory1, theory2)
        acc = []

        for _ in range(n_iter):
            pair_data = theory_sampler.get_manual_sample([theory1, theory2], n_sample)
            acc.append(cluster_pair(pair_data, GRID, KDE_BANDWIDTH))

        mean_acc = np.mean(acc)
        accuracy[theories_dict[theory1]][theories_dict[theory2]] = mean_acc
        print(f"Accuracy for {theory1} and {theory2}: {mean_acc}")

        return mean_acc

    results_rdd = pairs_df.rdd.map(pair_calculation)

    results_list = results_rdd.collect()
    print(results_list)

    print(accuracy)
