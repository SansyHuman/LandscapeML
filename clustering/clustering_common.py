from typing import Union

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np


def cluster_pair(theory1_input: np.ndarray, theory2_input: np.ndarray, calculate_silhouette: bool=False) -> tuple[float, Union[float, None]]:
    """
    Cluster two sets of data using KMeans clustering after dimensionality reduction with t-SNE.
    :param theory1_input: Input data for theory 1.
    :param theory2_input: Input data for theory 2.
    :param calculate_silhouette: Whether to calculate the silhouette score.
    :return: Accuracy of the clustering model and silhouette score if calculate_silhouette is True.
    """
    data_num = theory1_input.shape[0]
    X = np.vstack((theory1_input, theory2_input))
    y_true = np.hstack((np.zeros(data_num, dtype=int), np.ones(data_num, dtype=int)), dtype=int)

    Xs = StandardScaler().fit_transform(X)

    reduction_model = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        random_state=42
    )
    X_tsne = reduction_model.fit_transform(Xs)

    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    kmeans.fit(X_tsne)
    y_pred = kmeans.labels_

    acc_direct = np.mean(y_pred == y_true)
    acc_flipped = np.mean(y_pred == (1 - y_true))
    acc = float(max(acc_direct, acc_flipped))

    sil_score = None
    if calculate_silhouette:
        sil_score = silhouette_score(X_tsne, y_pred)

    return acc, sil_score


def calculate_feature_importance(theory1_input: np.ndarray, theory2_input: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Calculate the accuracy of clustering of two sets and importance score of each feature.
    :param theory1_input: Input data for theory 1.
    :param theory2_input: Input data for theory 2.
    :return: clustering accuracy and feature importance scores.
    """

    baseline_acc, baseline_sil = cluster_pair(theory1_input, theory2_input, calculate_silhouette=True)

    feature_impacts = np.zeros(theory1_input.shape[1], dtype=float)

    for i in range(theory1_input.shape[1]):
        # Drop feature i
        theory1_reduced = np.delete(theory1_input, i, axis=1)
        theory2_reduced = np.delete(theory2_input, i, axis=1)

        acc_score, sil_score = cluster_pair(theory1_reduced, theory2_reduced, calculate_silhouette=True)

        feature_impacts[i] = baseline_sil - sil_score

    return baseline_acc, feature_impacts
