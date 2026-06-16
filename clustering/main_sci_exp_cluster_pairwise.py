import datetime
import random
from common.balanced_sample_tool import TheorySampler
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import csv
import sys
import os
import numpy as np

import matplotlib.pyplot as plt
from common.sci_parser import SuperConformalIndex
from clustering_common import cluster_pair, calculate_feature_importance


def build_data_sci_exp(sampler: TheorySampler, n_per_theory: int, n_iter: int, grid: np.ndarray, kde_bandwidth: float):
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


def build_data_spectrum(sampler: TheorySampler, n_per_theory: int, n_iter: int, grid: np.ndarray, kde_bandwidth: float):
    data_per_theories = dict()

    stats = sampler.get_theory_stats()
    theories = stats.select("Name").rdd.flatMap(lambda x: x).collect()
    for theory in theories:
        data_per_theories[theory] = [[] for _ in range(n_iter)]

    for i in range(n_iter):
        sampled_data = sampler.get_manual_sample(theories, n_per_theory)
        rows = sampled_data.df.collect()

        for row in rows:
            data_per_theories[row["Name"]][i].append(SuperConformalIndex(row["SCI"]).featurize_relevant_spectrum(grid, kde_bandwidth))

        print(f"Data generation iteration {i + 1} completed.")

    for theory in theories:
        for i in range(n_iter):
            data_per_theories[theory][i] = np.stack(data_per_theories[theory][i])

    return data_per_theories


def build_data_sci(sampler: TheorySampler, n_per_theory: int, n_iter: int, grid: np.ndarray, kde_bandwidth: float):
    data_per_theories = dict()

    stats = sampler.get_theory_stats()
    theories = stats.select("Name").rdd.flatMap(lambda x: x).collect()
    for theory in theories:
        data_per_theories[theory] = [[] for _ in range(n_iter)]

    for i in range(n_iter):
        sampled_data = sampler.get_manual_sample(theories, n_per_theory)
        rows = sampled_data.df.collect()

        for row in rows:
            data_per_theories[row["Name"]][i].append(SuperConformalIndex(row["SCI"]).featurize_sci(grid, kde_bandwidth))

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

    print("Enter the program:")
    print("1. SCI exponents with positive coefficients")
    print("2. Relevant operator spectrum")
    print("3. Full SCI")
    program = int(input(">>"))

    program_name = ""
    number_of_ops_name = ""
    if program == 1:
        program_name = "sci_exp"
        number_of_ops_name = "Number of unique dimensions"
    elif program == 2:
        program_name = "spectrum"
        number_of_ops_name = "Number of relevant operators"
    elif program == 3:
        program_name = "sci"
        number_of_ops_name = "Sum of absolute values of coefficients"
    else:
        print("Invalid program number")
        exit(1)

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

    data_per_theory = None
    if program == 1:
        data_per_theory = build_data_sci_exp(cutoff_sampler, n_sample, n_iter, GRID, KDE_BANDWIDTH)
    elif program == 2:
        data_per_theory = build_data_spectrum(cutoff_sampler, n_sample, n_iter, GRID, KDE_BANDWIDTH)
    elif program == 3:
        data_per_theory = build_data_sci(cutoff_sampler, n_sample, n_iter, GRID, KDE_BANDWIDTH)

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
        print(f"Stats for {theory1} and {theory2} with {program_name} clustering")
        print(f"    mean: {mean_acc}, stdev: {stdev_acc}")

    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        executor.map(lambda arg: pair_calculation(*arg), pairs)

    feature_impact_scores = np.stack(feature_impact_scores)
    feature_impact_absavg = np.mean(np.abs(feature_impact_scores), axis=0)
    feature_names = [f"{GRID[i]:.4f}" for i in range(len(GRID))] + [
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

    save_dir = f"../data/clustering/{datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}"
    os.makedirs(save_dir, exist_ok=True)

    with open(f"{save_dir}/{program_name}_cluster_pairwise_acc.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Theories"] + theories)
        for i in range(n_theory):
            writer.writerow([theories[i]] + accuracy[i])

    with open(f"{save_dir}/{program_name}_cluster_pairwise_stdev.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Theories"] + theories)
        for i in range(n_theory):
            writer.writerow([theories[i]] + stdev[i])

    with open(f"{save_dir}/{program_name}_cluster_pairwise_impact.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Features"] + feature_names)
        writer.writerow(["Absolute average score"] + feature_impact_absavg.tolist())

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

    plt.savefig(f'{save_dir}/{program_name}_cluster_pairwise.png')

    plt.close("all")

    feature_impact_order = np.argsort(-feature_impact_absavg)

    fig, ax = plt.subplots()

    top_feature_names = [feature_names[feature_impact_order[i]] for i in range(10)]
    top_feature_impacts = feature_impact_absavg[feature_impact_order[:10]]

    fig.suptitle("Top 10 feature impact scores")
    bars = ax.barh(top_feature_names, top_feature_impacts, color="red")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xlabel("Absolute average score")
    plt.tight_layout()

    plt.savefig(f"{save_dir}/{program_name}_cluster_pairwise_top_impacts.png")

    plt.close("all")

    spectrum_absavg = feature_impact_absavg[:-16]

    fig, ax = plt.subplots()
    fig.suptitle("Spectral feature impact scores")

    ax.plot(GRID, spectrum_absavg, "o-", color="red", label="Absolute average score")
    ax.set_xlabel("Operator dimension")
    ax.set_ylabel("Impact score")
    ax.legend()
    plt.grid(True)

    plt.savefig(f'{save_dir}/{program_name}_cluster_pairwise_spectrum_impact.png')

    plt.close("all")

    stat_label = feature_names[-16:]
    stat_absavg = feature_impact_absavg[-16:]

    fig, ax = plt.subplots()
    fig.suptitle("Stat feature impact scores")

    x = np.arange(len(stat_label))
    width = 0.35

    ax.bar(x, stat_absavg, width, label="Absolute average score", color="red")
    ax.set_xticks(x + width / 2, stat_label, rotation=90)
    ax.set_ylabel("Absolute average SHAP value")
    ax.legend(loc="upper left", ncols=2)
    plt.tight_layout()

    plt.savefig(f'{save_dir}/{program_name}_cluster_pairwise_stat_impact.png')
