import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, confusion_matrix
from src.domain.model.similarity_service import SimilarityService


def get_layer_centroids(vectors_dict, df, phonemes_list, layer_name="layer_12"):
    centroids = []
    for p in phonemes_list:
        tids = df[df["phoneme"] == p]["token_id"].astype(str).values
        p_vectors = []
        for tid in tids:
            if tid in vectors_dict:
                data = vectors_dict[tid]
                vec = data.item().get(layer_name) if hasattr(data, "item") else data
                if vec is not None:
                    p_vectors.append(vec)

        if p_vectors:
            centroids.append(np.mean(p_vectors, axis=0))
        else:
            dim = 1024 if "whisper" in str(vectors_dict) else 1024
            centroids.append(np.zeros(dim))

    return np.array(centroids)


def run_rsa_and_classification():
    df = pd.read_csv("data/processed/final_analysis_table.csv")
    wh_raw = np.load("data/processed/features_whisper.npz", allow_pickle=True)
    xl_raw = np.load("data/processed/features_xlsr.npz", allow_pickle=True)

    phonemes = sorted(df["phoneme"].unique())
    service = SimilarityService()

    cent_ac = df.groupby("phoneme")[["f2_lob", "f1_lob"]].mean().loc[phonemes].values
    cent_wh = get_layer_centroids(wh_raw, df, phonemes, "layer_4")
    cent_xl = get_layer_centroids(xl_raw, df, phonemes, "layer_4")

    rsm_ac = service.compute_rsm(cent_ac, metric="euclidean")
    rsm_wh = service.compute_rsm(cent_wh, metric="cosine")

    corr_wh, p_wh = service.mantel_test(rsm_ac, rsm_wh)
    print(f"Mantel Correlation (Acoustic vs Whisper): r={corr_wh:.3f}, p={p_wh:.4f}")

    X = []
    y = []
    for _, row in df.iterrows():
        tid = str(row["token_id"])
        if tid in wh_raw:
            vec = wh_raw[tid].item().get("layer_12")
            if vec is not None:
                X.append(vec)
                y.append(row["phoneme"])

    X, y = np.array(X), np.array(y)
    clf = NearestCentroid()
    clf.fit(X, y)
    y_pred = clf.predict(X)

    acc = accuracy_score(y, y_pred)
    print(f"Phoneme Identification Accuracy (Whisper): {acc:.2%}")

    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y, y_pred, labels=phonemes, normalize="true")
    sns.heatmap(
        cm, annot=True, xticklabels=phonemes, yticklabels=phonemes, cmap="Blues"
    )
    plt.title(f"Whisper Phoneme Confusion Matrix (Acc: {acc:.2%})")
    plt.savefig("results/figures/whisper_confusion_matrix.png")


if __name__ == "__main__":
    run_rsa_and_classification()
