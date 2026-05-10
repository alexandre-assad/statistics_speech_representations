import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from src.domain.model.clustering_service import ClusteringService


def run_clustering():
    df = pd.read_csv("data/processed/final_analysis_table.csv")
    wh_raw = np.load("data/processed/features_whisper.npz", allow_pickle=True)

    vowels = ["i", "e", "ɛ", "a", "y", "ø", "u", "o", "ɑ"]
    df_vowels = df[df["phoneme"].isin(vowels)]

    front_back_dict = {
        "i": "front",
        "e": "front",
        "ɛ": "front",
        "a": "front",
        "y": "front",
        "ø": "front",
        "u": "back",
        "o": "back",
        "ɑ": "back",
    }

    true_labels = [front_back_dict[v] for v in vowels]

    ac_data = (
        df_vowels.groupby("phoneme")[["f1_lob", "f2_lob"]].mean().loc[vowels].values
    )

    wh_data = []
    for v in vowels:
        tids = df_vowels[df_vowels["phoneme"] == v]["token_id"].astype(str).values
        vecs = [wh_raw[t].item().get("layer_12") for t in tids if str(t) in wh_raw]
        vecs = [vec for vec in vecs if vec is not None]
        wh_data.append(np.mean(vecs, axis=0))
    wh_data = np.array(wh_data)

    service = ClusteringService()

    Z_ac = service.perform_clustering(ac_data, metric="euclidean", method="ward")
    ari_ac, _ = service.evaluate_clusters(Z_ac, true_labels, n_clusters=2)

    Z_wh = service.perform_clustering(wh_data, metric="cosine", method="average")
    ari_wh, _ = service.evaluate_clusters(Z_wh, true_labels, n_clusters=2)

    print(f"Acoustic : {ari_ac:.3f}")
    print(f"Whisper    : {ari_wh:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    dendrogram(Z_ac, labels=vowels, ax=axes[0], leaf_font_size=14)
    axes[0].set_title(f"Acoustic Clustering (ARI = {ari_ac:.3f})")
    axes[0].set_ylabel("Distance Euclidienne (Ward)")

    dendrogram(Z_wh, labels=vowels, ax=axes[1], leaf_font_size=14)
    axes[1].set_title(f"Whisper Clustering (ARI = {ari_wh:.3f})")
    axes[1].set_ylabel("Distance Cosinus (Average)")

    plt.tight_layout()
    plt.savefig("results/figures/dendrograms.png")


if __name__ == "__main__":
    run_clustering()
