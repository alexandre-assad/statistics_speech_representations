import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path


def normalise_project():
    df = pd.read_csv("data/processed/features_acoustic.csv")

    vowel_pattern = r"^[aeiouyøœɑ̃ɛ̃ɔ̃]$|[aeiouyøœɑ̃ɛ̃ɔ̃]:"
    vowels_df = df[df["phoneme"].str.contains(vowel_pattern, na=False)].copy()

    for speaker in vowels_df["speaker_id"].unique():
        speaker_mask = vowels_df["speaker_id"] == speaker
        for formant in ["f1", "f2"]:
            mean = vowels_df.loc[speaker_mask, formant].mean()
            std = vowels_df.loc[speaker_mask, formant].std()
            if std > 0:
                vowels_df.loc[speaker_mask, f"{formant}_lob"] = (
                    vowels_df.loc[speaker_mask, formant] - mean
                ) / std

    whisper_data = np.load("data/processed/features_whisper.npz", allow_pickle=True)

    vectors = []
    valid_ids = []
    for tid in vowels_df["token_id"]:
        if tid in whisper_data:
            content = whisper_data[tid].item()
            vec = content.get("layer_12")

            if vec is not None and not np.isnan(vec).any():
                if np.linalg.norm(vec) < 500:
                    vectors.append(vec)
                    valid_ids.append(tid)

    if len(vectors) > 10:
        X = np.array(vectors)

        scaler = StandardScaler()
        pca = PCA(n_components=2)
        neural_2d = pca.fit_transform(scaler.fit_transform(X))

        pca_df = pd.DataFrame(neural_2d, columns=["PC1", "PC2"])
        pca_df["token_id"] = valid_ids

        final_df = pd.merge(vowels_df, pca_df, on="token_id")
        final_df.drop_duplicates(subset=["token_id"], inplace=True)

        final_df = final_df[
            (final_df["f1_lob"].abs() <= 3) & (final_df["f2_lob"].abs() <= 3)
        ]

        final_df.to_csv("data/processed/final_analysis_table.csv", index=False)
        print(f"Final analysis table created with {len(final_df)} valid tokens.")
    else:
        print("Error: Not enough valid vectors for PCA.")
        return


if __name__ == "__main__":
    normalise_project()
