import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import json
from pathlib import Path

from src.domain.model.neural_metrics import NeuralMetrics


def compute_vsa(df, x_col, y_col):
    centroids = df.groupby("phoneme")[[x_col, y_col]].mean().values
    if len(centroids) < 3:
        return 0
    return ConvexHull(centroids).volume


def compute_neural_metrics(df, model_name="whisper"):
    X = df[["PC1", "PC2"]].values
    labels = df["phoneme"].values

    total_var = np.var(X, axis=0).sum()
    centroids = df.groupby("phoneme")[["PC1", "PC2"]].mean().values
    between_var = np.var(centroids, axis=0).sum()
    var_ratio = between_var / total_var

    sim_matrix = cosine_similarity(X)

    within_sims = []
    between_sims = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                within_sims.append(sim_matrix[i, j])
            else:
                between_sims.append(sim_matrix[i, j])

    avg_within = np.mean(within_sims)
    avg_between = np.mean(between_sims)
    sim_ratio = avg_within / avg_between

    results = {
        f"{model_name}_var_ratio": round(var_ratio, 4),
        f"{model_name}_within_sim": round(avg_within, 4),
        f"{model_name}_between_sim": round(avg_between, 4),
        f"{model_name}_sim_ratio": round(sim_ratio, 4),
    }

    output_path = Path("results/stats_neural_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Statistics neural generated: {results}")

    return results


def run_statistics():
    df = pd.read_csv("data/processed/final_analysis_table.csv")
    compute_neural_metrics(df)
    results = {}

    for status in ["fr", "ru"]:
        sub_df = df[df["l1_status"] == status]
        vsa_acoustic = compute_vsa(sub_df, "f2_lob", "f1_lob")
        vsa_neural = compute_vsa(sub_df, "PC1", "PC2")

        results[f"vsa_acoustic_{status}"] = round(vsa_acoustic, 4)
        results[f"vsa_neural_{status}"] = round(vsa_neural, 4)

    centroids_ac = df.groupby("phoneme")[["f2_lob", "f1_lob"]].mean()
    centroids_ne = df.groupby("phoneme")[["PC1", "PC2"]].mean()

    common_phonemes = centroids_ac.index.intersection(centroids_ne.index)
    dist_ac = []
    dist_ne = []

    from scipy.spatial.distance import pdist

    dist_ac = pdist(centroids_ac.loc[common_phonemes].values)
    dist_ne = pdist(centroids_ne.loc[common_phonemes].values)

    correlation, p_value = pearsonr(dist_ac, dist_ne)
    results["acoustic_neural_correlation"] = round(correlation, 4)
    results["correlation_p_value"] = round(p_value, 4)

    metrics_tool = NeuralMetrics()
    results["neural_variance_ratio"] = metrics_tool.calculate_variance_ratio(df)
    results["neural_cosine_similarity_ratio"] = metrics_tool.calculate_cosine_ratio(df)

    output_path = Path("results/stats_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Statistics generated: {results}")


if __name__ == "__main__":
    run_statistics()
