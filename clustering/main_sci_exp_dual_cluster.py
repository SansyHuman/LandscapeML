from clustering.main_sci_exp_cluster_pairwise import build_data_sci, build_data_sci_exp, build_data_spectrum
import datetime
import random
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

    seiberg_pair = []
    kutasov_pair = []
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
                seiberg_pair.append((theory1, theory2))
            elif mode == "Kutasov":
                kutasov_pair.append((theory1, theory2))

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

    n_sample = int(input("Enter the number of samples from each theories: "))
    n_iter = int(input("Enter the number of iterations for each pairs: "))

    theories = stats.select("Name").rdd.flatMap(lambda x: x).collect()
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
