import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class NeuralStatsService:
    @staticmethod
    def compute_variance_ratio(df, x_col="PC1", y_col="PC2"):
        X = df[[x_col, y_col]].values
        total_var = np.var(X, axis=0).sum()

        centroids = df.groupby("phoneme")[[x_col, y_col]].mean().values
        between_var = np.var(centroids, axis=0).sum()

        return between_var / total_var

    @staticmethod
    def compute_cosine_ratios(df, x_col="PC1", y_col="PC2"):
        X = df[[x_col, y_col]].values
        labels = df["phoneme"].values

        sim_matrix = cosine_similarity(X)
        within_sims = []
        between_sims = []

        sample_size = min(2000, len(labels))
        indices = np.random.choice(len(labels), sample_size, replace=False)

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx_i, idx_j = indices[i], indices[j]
                if labels[idx_i] == labels[idx_j]:
                    within_sims.append(sim_matrix[idx_i, idx_j])
                else:
                    between_sims.append(sim_matrix[idx_i, idx_j])

        return np.mean(within_sims) / np.mean(between_sims)
