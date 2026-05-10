import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


class SimilarityService:
    @staticmethod
    def compute_rsm(matrix, metric="cosine"):
        dists = pdist(matrix, metric=metric)
        return squareform(dists)

    @staticmethod
    def mantel_test(m1, m2, permutations=5000):
        n = m1.shape[0]
        i_upper = np.triu_indices(n, k=1)
        v1, v2 = m1[i_upper], m2[i_upper]

        obs_corr, _ = spearmanr(v1, v2)

        count = 0
        for _ in range(permutations):
            idx = np.random.permutation(n)
            m1_perm = m1[idx, :][:, idx]
            v1_perm = m1_perm[i_upper]
            corr, _ = spearmanr(v1_perm, v2)
            if corr >= obs_corr:
                count += 1

        return obs_corr, count / permutations
