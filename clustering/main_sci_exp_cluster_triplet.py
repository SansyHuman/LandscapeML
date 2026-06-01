import sys
import csv
from concurrent.futures import ThreadPoolExecutor
from itertools import permutations

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import multiprocessing as mp
import datetime

from common.balanced_sample_tool import TheorySampler
from common.sci_parser import SuperConformalIndex


def build_data(sampler: TheorySampler, theories: list[str], n_per_theory: int, n_iter: int, grid: np.ndarray, kde_bandwidth: float):
    data_per_theories = dict()

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


def cluster_triplet(theory1_input: np.ndarray, theory2_input: np.ndarray, theory3_input: np.ndarray):
    data_num = theory1_input.shape[0]
    X = np.vstack((theory1_input, theory2_input, theory3_input))
    y_true = np.hstack((np.zeros(data_num, dtype=int), np.ones(data_num, dtype=int), np.full(data_num, 2, dtype=int)))

    Xs = StandardScaler().fit_transform(X)

    reduction_model = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        random_state=42
    )
    X_tsne = reduction_model.fit_transform(Xs)

    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    kmeans.fit(X_tsne)
    y_pred = kmeans.labels_

    cluster_stats = [[0, 0, 0] for _ in range(3)]
    cluster_labels = [-1, -1, -1]

    for i in range(len(y_true)):
        cluster_stats[y_true[i]][y_pred[i]] += 1

    for i in range(3):
        ranks = np.argsort(cluster_stats[i])
        for j in range(1, 4):
            if ranks[-j] not in cluster_labels:
                cluster_labels[i] = ranks[-j]
                break

    correct = 0
    for i in range(3):
        correct += cluster_stats[i][cluster_labels[i]]

    return float(correct) / (3 * data_num)


if __name__ == '__main__':
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

    data_path = input("Enter the directory of the pairwise clustering data: ")

    sorted_path = data_path
    with os.scandir(data_path) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith("pairwise_acc_sorted.csv"):
                sorted_path += f"/{entry.name}"

    sorted_data = None
    with open(sorted_path) as csvfile:
        reader = csv.reader(csvfile)
        sorted_data = list(reader)

    theories = sorted_data[0][1:]
    print("Theories in the order of decreasing accuracy scores")
    print("=======================================================")
    for i in range(len(theories)):
        print(f"{i + 1}. {theories[i]}")

    print("Choose which theories to choose:")
    print("1. Highest accuracies 2. Lowest accuracies")
    program = int(input(">>>"))

    n_theory = int(input("Enter the number of theories to choose: "))
    n_sample = int(input("Enter the number of samples from each theories: "))
    n_iter = int(input("Enter the number of iterations for each pairs: "))

    if program == 1:
        theories = theories[:n_theory]
    elif program == 2:
        theories = theories[-n_theory:]
    print(f"Selected theories: {theories}")
    theories_dict = dict()
    for i in range(n_theory):
        theories_dict[theories[i]] = i

    accuracy = np.zeros((n_theory, n_theory, n_theory))
    accuracy_for_graph = np.ones((n_theory, n_theory, n_theory))
    stdev = np.zeros((n_theory, n_theory, n_theory))

    triplets = [(theories[i], theories[j], theories[k]) for i in range(n_theory) for j in range(i + 1, n_theory) for k in range(j + 1, n_theory)]
    print(f"Number of theory triplets: {len(triplets)}")

    data_per_theory = build_data(theory_sampler, theories, n_sample, n_iter, GRID, KDE_BANDWIDTH)

    def triplet_calculation(theories: tuple[str, str, str]):
        print(f"Calculating accuracy for triplet: {theories}")
        acc = []

        for i in range(n_iter):
            acc.append(cluster_triplet(
                data_per_theory[theories[0]][i],
                data_per_theory[theories[1]][i],
                data_per_theory[theories[2]][i]
            ))

        mean_acc = float(np.mean(acc))
        stdev_acc = float(np.std(acc))

        perms = list(permutations(theories))
        for perm in perms:
            accuracy[theories_dict[perm[0]], theories_dict[perm[1]], theories_dict[perm[2]]] = mean_acc
            accuracy_for_graph[theories_dict[perm[0]], theories_dict[perm[1]], theories_dict[perm[2]]] = mean_acc
            stdev[theories_dict[perm[0]], theories_dict[perm[1]], theories_dict[perm[2]]] = stdev_acc

        print(f"Stats for {theories} with exponent clustering")
        print(f"    mean: {mean_acc}, stdev: {stdev_acc}")

    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        executor.map(triplet_calculation, triplets)

    save_dir = f"../data/clustering/{datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}"
    os.makedirs(save_dir, exist_ok=True)

    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 12

    x, y, z = np.indices(accuracy_for_graph.shape)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    acc_values = accuracy_for_graph.flatten()
    acc_scorers = 1.0 - (acc_values - 1.0 / 3.0) * (3.0 / 2.0)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle(f"Accuracies of triplets iterations: {n_iter}")

    ax.set_title("Larger dot means lower accuracy")
    scatter = ax.scatter(x, y, z, s=acc_scorers * 100, c=acc_values, vmin=1.0 / 3.0, vmax=1.0, cmap='gray', alpha=0.7)
    fig.colorbar(scatter, ax=ax, shrink=0.7)

    plt.savefig(f"{save_dir}/sci_exp_cluster_triplet.png")
    plt.show()

    accuracy = accuracy.tolist()
    stdev = stdev.tolist()

    with open(f"{save_dir}/sci_exp_cluster_triplet_acc.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for i in range(n_theory):
            writer.writerow([theories[i]] + theories)
            for j in range(n_theory):
                writer.writerow([theories[j]] + accuracy[i][j])

            writer.writerow([])

    with open(f"{save_dir}/sci_exp_cluster_triplet_stdev.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for i in range(n_theory):
            writer.writerow([theories[i]] + theories)
            for j in range(n_theory):
                writer.writerow([theories[j]] + stdev[i][j])

            writer.writerow([])
