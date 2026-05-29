import sys
import csv
import numpy as np
import os
import matplotlib.pyplot as plt


if __name__ == '__main__':
    print(sys.version)
    print("GIL enabled:", sys._is_gil_enabled())

    csv.field_size_limit(np.iinfo(np.int32).max)

    data_path = input("Enter the directory of the pairwise clustering data: ")

    acc_path = data_path
    stdev_path = data_path
    with os.scandir(data_path) as entries:
        for entry in entries:
            if entry.is_file():
                if entry.name.endswith("pairwise_acc.csv"):
                    acc_path += f"/{entry.name}"
                if entry.name.endswith("pairwise_stdev.csv"):
                    stdev_path += f"/{entry.name}"

    acc_data = None
    with open(acc_path) as csvfile:
        reader = csv.reader(csvfile)
        acc_data = list(reader)

    stdev_data = None
    with open(stdev_path) as csvfile:
        reader = csv.reader(csvfile)
        stdev_data = list(reader)

    theories = []
    for i in range(1, len(acc_data)):
        theories.append(acc_data[i][0])

    n_theory = len(theories)

    theories_dict = dict()
    for i in range(n_theory):
        theories_dict[theories[i]] = i

    acc = np.zeros((n_theory, n_theory))
    stdev = np.zeros((n_theory, n_theory))

    for i in range(n_theory):
        for j in range(n_theory):
            acc[i][j] = float(acc_data[i + 1][j + 1])
            stdev[i][j] = float(stdev_data[i + 1][j + 1])

    acc_scores = np.zeros(n_theory)
    for i in range(n_theory):
        acc_scores[i] = float(np.sum(acc[i])) / (n_theory - 1)

    order = np.argsort(-acc_scores)

    theories_sorted = []
    acc_sorted = [[0.0 for _ in range(n_theory)] for _ in range(n_theory)]
    stdev_sorted = [[0.0 for _ in range(n_theory)] for _ in range(n_theory)]
    for i in range(n_theory):
        theories_sorted.append(theories[order[i]])
        for j in range(n_theory):
            acc_sorted[i][j] = float(acc[order[i]][order[j]])
            stdev_sorted[i][j] = float(stdev[order[i]][order[j]])

    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 12

    fig, ax = plt.subplots(nrows=1, ncols=2, squeeze=True)
    fig.suptitle("Pairwise clustering accuracies and stdevs")

    vmin, vmax = 0.5, 1.0

    ax[0].set_title("Accuracies")
    cmap_plot = ax[0].matshow(acc_sorted, vmin=vmin, vmax=vmax, cmap='gray')
    fig.colorbar(cmap_plot, ax=ax[0], shrink=0.7)

    ax[1].set_title(f"Stdevs")
    cmap_plot = ax[1].matshow(stdev_sorted, vmin=0.0, vmax=np.max(stdev), cmap='gray_r')
    fig.colorbar(cmap_plot, ax=ax[1], shrink=0.7)

    plt.savefig(f'{data_path}/cluster_pairwise_sorted.png')

    with open(f"{data_path}/cluster_pairwise_acc_sorted.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Theories"] + theories_sorted)
        for i in range(n_theory):
            writer.writerow([theories_sorted[i]] + acc_sorted[i])

    with open(f"{data_path}/cluster_pairwise_stdev_sorted.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Theories"] + theories_sorted)
        for i in range(n_theory):
            writer.writerow([theories_sorted[i]] + stdev_sorted[i])
