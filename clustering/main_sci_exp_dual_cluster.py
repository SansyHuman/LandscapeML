from clustering.main_sci_exp_cluster_pairwise import build_data_sci, build_data_sci_exp, build_data_spectrum
from clustering.clustering_common import cluster_pair
import datetime
import matplotlib.pyplot as plt
from common.balanced_sample_tool import TheorySampler
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import csv
import sys
import os
import numpy as np

if __name__ == '__main__':
    # os.environ["PYSPARK_PYTHON"] = "/home/subo-lee/PycharmProjects/LandscapeML/.venv/bin/python3.14t"
    # os.environ["PYSPARK_DRIVER_PYTHON"] = "/home/subo-lee/PycharmProjects/LandscapeML/.venv/bin/python3.14t"

    print(sys.version)
    print("GIL enabled:", sys._is_gil_enabled())

    os.makedirs('../data/clustering', exist_ok=True)
    csv.field_size_limit(np.iinfo(np.int32).max)

    data_filename = input("Enter data file name to load: ")

    theory_sampler = TheorySampler(data_filename)
    stats = theory_sampler.get_theory_stats()
    print("Whole data stats")
    print("==================================")
    for row in stats.collect():
        print(row.asDict())
    print()

    seiberg_pair = set()
    kutasov_pair = set()
    dual_theories = set()

    mode = "None"

    dual_datafile = input("Enter dual theory data file name to load: ")
    with open(dual_datafile, 'r') as dual_file:
        raw_data = dual_file.readlines()
        for line in raw_data:
            if line.strip() == "Seiberg":
                mode = "Seiberg"
                continue
            elif line.strip() == "Kutasov":
                mode = "Kutasov"
                continue
            elif line.strip() == "":
                continue

            if mode == "None":
                print("Invalid dual theory data format")
                exit(1)

            theory_pair = line.strip().split(",")
            if len(theory_pair) != 2:
                print("Invalid dual theory data format")
                exit(1)

            theory1 = theory_pair[0].strip()
            theory2 = theory_pair[1].strip()
            if mode == "Seiberg":
                seiberg_pair.add((theory1, theory2))
                seiberg_pair.add((theory2, theory1))
            elif mode == "Kutasov":
                kutasov_pair.add((theory1, theory2))
                kutasov_pair.add((theory2, theory1))

            dual_theories.add(theory1)
            dual_theories.add(theory2)

    print("Dual theories:", dual_theories)

    dual_sampler = theory_sampler.get_selected_theories(list(dual_theories))
    stats = dual_sampler.get_theory_stats()
    print("Dual data stats")
    print("==================================")
    for row in stats.collect():
        print(row.asDict())
    print()

    print("Enter the program:")
    print("1. SCI exponents with positive coefficients")
    print("2. Relevant operator spectrum")
    print("3. Full SCI")
    program = int(input(">>"))

    program_name = ""
    if program == 1:
        program_name = "sci_exp"
    elif program == 2:
        program_name = "spectrum"
    elif program == 3:
        program_name = "sci"
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

    dual_sampler = dual_sampler.get_theories_above_count(min_count)
    theories = dual_sampler.get_theory_stats().select("Name").rdd.flatMap(lambda x: x).collect()
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

    data_per_theory = None
    if program == 1:
        data_per_theory = build_data_sci_exp(dual_sampler, n_sample, n_iter, GRID, KDE_BANDWIDTH)
    elif program == 2:
        data_per_theory = build_data_spectrum(dual_sampler, n_sample, n_iter, GRID, KDE_BANDWIDTH)
    elif program == 3:
        data_per_theory = build_data_sci(dual_sampler, n_sample, n_iter, GRID, KDE_BANDWIDTH)

    def pair_calculation(theory1: str, theory2: str):
        print(f"Calculating accuracy for {theory1} and {theory2}")
        acc = []

        for i in range(n_iter):
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

    save_dir = f"../data/clustering/{datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}"
    os.makedirs(save_dir, exist_ok=True)

    with open(f"{save_dir}/{program_name}_dual_cluster_acc.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Theories"] + theories)
        for i in range(n_theory):
            writer.writerow([theories[i]] + accuracy[i])

    with open(f"{save_dir}/{program_name}_dual_cluster_stdev.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Theories"] + theories)
        for i in range(n_theory):
            writer.writerow([theories[i]] + stdev[i])

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

    plt.savefig(f'{save_dir}/{program_name}_dual_cluster.png')

    plt.close("all")

    pair_accuracy = [accuracy[i][j] for i in range(n_theory) for j in range(i + 1, n_theory)]
    acc_order = np.argsort(pair_accuracy).tolist()

    pairs_sorted = [pairs[acc_order[i]] for i in range(len(acc_order))]
    pair_acc_sorted = [pair_accuracy[acc_order[i]] for i in range(len(acc_order))]

    x_nondual = []
    x_seiberg = []
    x_kutasov = []
    for i in range(len(acc_order)):
        if pairs_sorted[i] in seiberg_pair:
            x_seiberg.append(i)
        elif pairs_sorted[i] in kutasov_pair:
            x_kutasov.append(i)
        else:
            x_nondual.append(i)

    pair_acc_sorted = np.asarray(pair_acc_sorted)
    x_nondual = np.asarray(x_nondual)
    x_seiberg = np.asarray(x_seiberg)
    x_kutasov = np.asarray(x_kutasov)

    fig, ax = plt.subplots()
    fig.suptitle("Accuracies of each pair")
    ax.bar(x_nondual, pair_acc_sorted[x_nondual], color='gray', label="Non-dual")
    ax.bar(x_seiberg, pair_acc_sorted[x_seiberg], color='blue', label="Seiberg")
    ax.bar(x_kutasov, pair_acc_sorted[x_kutasov], color='red', label="Kutasov")
    ax.set_xlabel("Pairs")
    ax.set_ylabel("Accuracy")
    ax.legend()

    plt.savefig(f'{save_dir}/{program_name}_dual_cluster_pair_acc.png')

    with open(f"{save_dir}/{program_name}_dual_cluster_acc_order_per_pair_{n_theory}_theories.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Theory1", "Theory2", "Dual", "Order"])
        for theory1, theory2 in seiberg_pair:
            if theory1 == theory2:
                continue
            if theory1 in theories_dict and theory2 in theories_dict:
                t1 = theories_dict[theory1]
                t2 = theories_dict[theory2]
                rank = int(np.asarray(accuracy[t1]).argsort().argsort()[t2])
                writer.writerow([theory1, theory2, "Seiberg", rank])

        for theory1, theory2 in kutasov_pair:
            if theory1 == theory2:
                continue
            if theory1 in theories_dict and theory2 in theories_dict:
                t1 = theories_dict[theory1]
                t2 = theories_dict[theory2]
                rank = int(np.asarray(accuracy[t1]).argsort().argsort()[t2])
                writer.writerow([theory1, theory2, "Kutasov", rank])
