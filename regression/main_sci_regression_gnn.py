import datetime
import sys
import os
import csv
import numpy as np
import shap
import torch
import torch_geometric.nn as pyg_nn
from torch_geometric.loader import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

from common.balanced_sample_tool import TheorySampler
from common.sci_parser import SuperConformalIndex
from common.utils import GenericDataset, FullyConnectedNetwork, MRSELoss


def build_data_sci(sampler: TheorySampler, charge_col: str, min_charge: float, max_charge: float,
               n_bins: int, n_per_bins: int, n_train: int, n_test: int, n_validate: int,
               min_dim: float, max_dim: float):
    data_train = []
    data_test = []
    data_validate = []

    def build_dataset(data: TheorySampler, datasets, add_id=False):
        rows = data.df.collect()

        dataset = []
        for row in rows:
            sci_graph = SuperConformalIndex(row["SCI"]).featurize_sci_graph(min_dim, max_dim)
            sci_graph.y = torch.tensor([[float(row[charge_col])]], dtype=torch.float32)
            if add_id:
                sci_graph.id = int(row["id"])

            dataset.append(sci_graph)

        datasets.append(dataset)

    for i in range(n_train):
        build_dataset(sampler.get_balanced_bins_sample(charge_col, min_charge, max_charge, n_bins, n_per_bins),
                      data_train)
        print(f"Train data {i + 1} built.")

    for i in range(n_test):
        build_dataset(sampler.get_balanced_bins_sample(charge_col, min_charge, max_charge, n_bins, n_per_bins),
                      data_test)
        print(f"Test data {i + 1} built.")

    for i in range(n_validate):
        build_dataset(sampler.get_balanced_bins_sample(charge_col, min_charge, max_charge, n_bins, n_per_bins),
                      data_validate, add_id=True)
        print(f"Validation data {i + 1} built.")

    return data_train, data_test, data_validate


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

    DIM_LO = float(input("Enter lower bound of dimension of operator to use: "))
    DIM_HI = float(input("Enter upper bound of dimension of operator to use: "))

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

    n_train = int(input("Enter number of training samples: "))
    n_test = int(input("Enter number of testing samples: "))
    n_validate = int(input("Enter number of validation samples: "))

    data_train, data_test, data_validate = build_data_sci(
        theory_sampler, central_charge, min_charge, max_charge, n_bins, n_per_bins,
        n_train, n_test, n_validate, DIM_LO, DIM_HI
    )

    dataloader_train = [DataLoader(data_train[i], batch_size=64, shuffle=True) for i in range(len(data_train))]
    dataloader_test = [DataLoader(data_test[i], batch_size=64, shuffle=False) for i in range(len(data_test))]
    dataloader_validate = [DataLoader(data_validate[i], batch_size=64, shuffle=False) for i in range(len(data_validate))]

    for step, data in enumerate(dataloader_train[0]):
        print(f'Step {step + 1}:')
        print('========')
        print(f'Number of graphs in the current batch: {data.num_graphs}')
        print(data)
        print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
