import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import pdist


class ClusteringService:
    @staticmethod
    def perform_clustering(data, metric="euclidean", method="ward"):
        if metric == "cosine" and method == "ward":
            method = "average"

        distances = pdist(data, metric=metric)
        Z = linkage(distances, method=method)
        return Z

    @staticmethod
    def evaluate_clusters(Z, true_labels, n_clusters=2):
        pred_labels = fcluster(Z, t=n_clusters, criterion="maxclust")
        ari = adjusted_rand_score(true_labels, pred_labels)
        return ari, pred_labels
