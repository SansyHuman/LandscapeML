import csv
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

data = None
with open("../landscape_SU2adj1nf2.csv") as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

w_index, a_index, c_index, rational_index = -1, -1, -1, -1
for i in range(len(data[0])):
    if data[0][i] == "Superpotentials":
        w_index = i
    elif data[0][i] == "CentralChargeA":
        a_index = i
    elif data[0][i] == "CentralChargeC":
        c_index = i
    elif data[0][i] == "Rational":
        rational_index = i

print(f'Superpotentials: {w_index}, A: {a_index}, C: {c_index}, Rational: {rational_index}')

superpotentials = []

for i in range(1, len(data)):
    w = data[i][w_index][1:-1].split(',')
    w_text = ""
    for term in w:
        w_text += f"{term.strip()} "
    w_text = w_text[:-1]
    superpotentials.append(w_text)

rational = [data[i + 1][rational_index] != '' for i in range(len(superpotentials))]

vectorizer = TfidfVectorizer()

t0 = time()
superpotentials_vector = vectorizer.fit_transform(superpotentials)

print(f'Vectorization took {time() - t0} seconds.')
print(f"samples: {superpotentials_vector.shape[0]}, features: {superpotentials_vector.shape[1]}")

kmeans = KMeans(n_clusters=32)
kmeans.fit(superpotentials_vector)
cluster_ids, cluster_sizes = np.unique(kmeans.labels_, return_counts=True)
print(f"Number of elements assigned to each cluster: {cluster_sizes}")

num_rationals = [0] * len(cluster_ids)
for i in range(len(rational)):
    if rational[i]:
        num_rationals[kmeans.labels_[i]] += 1

x = np.arange(len(num_rationals))
ratio = np.array(num_rationals) / np.array(cluster_sizes)

plt.rcParams["figure.figsize"] = (16, 12)
plt.bar(x, ratio)
plt.xticks(x, cluster_ids)

plt.title('Ratio of rational central charges in cluster')
plt.xlabel('Cluster')
plt.ylabel('Ratio of rational central charges')

plt.show()