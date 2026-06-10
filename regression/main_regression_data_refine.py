import sys
import csv
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt


if __name__ == '__main__':
    print(sys.version)
    print("GIL enabled:", sys._is_gil_enabled())

    os.makedirs('../data/regression', exist_ok=True)
    csv.field_size_limit(np.iinfo(np.int32).max)

    a_path = input("Enter the file of the a charge regression SHAP value data: ")
    c_path = input("Enter the file of the c charge regression SHAP value data: ")

    a_shap_raw = None
    with open(a_path) as csvfile:
        reader = csv.reader(csvfile)
        a_shap_raw = list(reader)

    c_shap_raw = None
    with open(c_path) as csvfile:
        reader = csv.reader(csvfile)
        c_shap_raw = list(reader)

    spectrum_label = [float(a_shap_raw[0][i]) for i in range(1, len(a_shap_raw[0]) - 16)]
    stat_label = a_shap_raw[0][-16:]
    len_spectrum = len(spectrum_label)

    a_shap_spectrum = [[float(a_shap_raw[i][j + 1]) for j in range(len_spectrum)] for i in range(1, len(a_shap_raw))]
    c_shap_spectrum = [[float(c_shap_raw[i][j + 1]) for j in range(len_spectrum)] for i in range(1, len(c_shap_raw))]

    a_shap_stat = [[float(a_shap_raw[i][-16 + j]) for j in range(16)] for i in range(1, len(a_shap_raw))]
    c_shap_stat = [[float(c_shap_raw[i][-16 + j]) for j in range(16)] for i in range(1, len(c_shap_raw))]

    a_shap_spectrum = np.average(np.abs(a_shap_spectrum), axis=0)
    c_shap_spectrum = np.average(np.abs(c_shap_spectrum), axis=0)
    a_shap_stat = np.average(np.abs(a_shap_stat), axis=0)
    c_shap_stat = np.average(np.abs(c_shap_stat), axis=0)

    file_suffix = f"{a_path.split('/')[-1]}+{c_path.split('/')[-1]}"
    save_dir = f"../data/regression/{datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}"
    os.makedirs(save_dir, exist_ok=True)

    with open(f"{save_dir}/{file_suffix}_spectrum_shap.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Charge"] + spectrum_label)
        writer.writerow(["a"] + a_shap_spectrum.tolist())
        writer.writerow(["c"] + c_shap_spectrum.tolist())
        writer.writerow(["a - c"] + (a_shap_spectrum - c_shap_spectrum).tolist())

    with open(f"{save_dir}/{file_suffix}_stat_shap.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Charge"] + stat_label)
        writer.writerow(["a"] + a_shap_stat.tolist())
        writer.writerow(["c"] + c_shap_stat.tolist())
        writer.writerow(["a - c"] + (a_shap_stat - c_shap_stat).tolist())

    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (20, 12)
    plt.rcParams['font.size'] = 15

    fig, ax = plt.subplots()
    fig.suptitle("Spectrum Absavg SHAP value of a anc c")

    ax.plot(spectrum_label, a_shap_spectrum, "o-", color="red", label="a")
    ax.plot(spectrum_label, c_shap_spectrum, "o-", color="blue", label="c")
    ax.set_yscale("log")
    ax.set_xlabel("Operator dimension")
    ax.set_ylabel("Absolute average SHAP value")
    ax.legend()

    plt.savefig(f"{save_dir}/{file_suffix}_spectrum_shap.png")

    plt.close("all")

    fig, ax = plt.subplots()
    fig.suptitle("Stat Absavg SHAP value of a anc c")

    x = np.arange(len(stat_label))
    width = 0.35

    ax.bar(x, a_shap_stat, width, label="a", color="red")
    ax.bar(x + width, c_shap_stat, width, label="c", color="blue")
    ax.set_yscale("log")
    ax.set_xticks(x + width / 2, stat_label, rotation=90)
    ax.set_ylabel("Absolute average SHAP value")
    ax.legend(loc="upper left", ncols=2)
    plt.tight_layout()

    plt.savefig(f"{save_dir}/{file_suffix}_stat_shap.png")

    with open(f"{save_dir}/original_data_path.txt", 'w', newline='') as f:
        f.write(f"{a_path}\n{c_path}")
