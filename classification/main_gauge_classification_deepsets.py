import datetime
import sys
import os
import csv
from typing import Union

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import matplotlib.pyplot as plt

from common.balanced_sample_tool import TheorySampler
from common.sci_parser import SuperConformalIndex


if __name__ == "__main__":
    print(sys.version)
    print("GIL enabled:", sys._is_gil_enabled())

    os.makedirs('../data/classification', exist_ok=True)
    csv.field_size_limit(np.iinfo(np.int32).max)

    filename = input("Enter file name to load: ")

    theory_sampler = TheorySampler(filename)
    stats = theory_sampler.get_theory_stats()
    for row in stats.collect():
        print(row.asDict())

    theory_sampler.get_gauge_group_stats().show(n=15, truncate=False)
