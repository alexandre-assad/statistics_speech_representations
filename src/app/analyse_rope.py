import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.domain.model.rope_service import ROPEService


def run_rope_analysis():
    df = pd.read_csv("data/processed/final_analysis_table.csv")
    wh_raw = np.load("data/processed/features_whisper.npz", allow_pickle=True)

    feature_to_test = "f1" if "f1" in df.columns else "f1_lob"
    rope_ac_limit = 20 if feature_to_test == "f1" else 0.4

    service = ROPEService()
    phonemes = sorted(df["phoneme"].unique())

    neural_rope_limit = service.compute_neural_rope_baseline(df, wh_raw)
    print(f"Neural ROPE defined at : [0, {neural_rope_limit:.4f}]")

    results = []

    for p in phonemes:
        ci_low, mean_diff, ci_high = service.bootstrap_acoustic_diff(
            df, p, feature=feature_to_test
        )
        classif_ac = service.classify_rope(
            ci_low, ci_high, -rope_ac_limit, rope_ac_limit
        )

        results.append(
            {
                "Phoneme": p,
                "Modality": "Acoustic (F1)",
                "Estimate": mean_diff,
                "CI_Lower": ci_low,
                "CI_Upper": ci_high,
                "ROPE_Class": classif_ac,
            }
        )

    results_df = pd.DataFrame(results)
    print(
        results_df[
            ["Phoneme", "Modality", "Estimate", "CI_Lower", "CI_Upper", "ROPE_Class"]
        ]
    )

    plt.figure(figsize=(8, 6))
    for i, row in results_df.iterrows():
        color = (
            "green"
            if row["ROPE_Class"] == "Equivalent"
            else ("red" if row["ROPE_Class"] == "Non-equivalent" else "orange")
        )
        plt.plot([row["CI_Lower"], row["CI_Upper"]], [i, i], color=color, lw=2)
        plt.plot(row["Estimate"], i, "ko")

    plt.axvspan(-rope_ac_limit, rope_ac_limit, color="grey", alpha=0.2, label="ROPE")
    plt.axvline(0, color="black", linestyle="--")

    plt.yticks(range(len(results_df)), results_df["Phoneme"])
    plt.title(f"Forest Plot - Différence L1/L2 ({feature_to_test})")
    plt.xlabel(f"Différence Moyenne (Natifs - Apprenants)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/forest_plot_acoustic.png")


if __name__ == "__main__":
    run_rope_analysis()
