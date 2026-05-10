import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


class PlottingService:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def plot_vowel_chart_with_ellipses(self, df, filename):
        plt.figure(figsize=(10, 8))
        unique_phonemes = df["phoneme"].unique()
        colors = sns.color_palette("husl", len(unique_phonemes))

        ax = plt.gca()

        for i, phoneme in enumerate(unique_phonemes):
            for status in ["fr", "ru"]:
                subset = df[(df["phoneme"] == phoneme) & (df["l1_status"] == status)]
                if len(subset) < 5:
                    continue

                color = colors[i]
                marker = "o" if status == "fr" else "X"
                plt.scatter(
                    subset["f2_lob"],
                    subset["f1_lob"],
                    alpha=0.15,
                    s=15,
                    color=color,
                    marker=marker,
                )

                points = subset[["f2_lob", "f1_lob"]].values
                mean = np.mean(points, axis=0)
                cov = np.cov(points, rowvar=False)

                vals, vecs = np.linalg.eigh(cov)
                order = vals.argsort()[::-1]
                vals, vecs = vals[order], vecs[:, order]

                theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

                width, height = 2 * np.sqrt(vals * 5.991)

                ell = Ellipse(
                    xy=mean,
                    width=width,
                    height=height,
                    angle=theta,
                    edgecolor=color,
                    facecolor="none",
                    linewidth=2,
                    linestyle="--" if status == "ru" else "-",
                    alpha=0.8,
                )
                ax.add_patch(ell)

        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.title("Vowel Chart with 95% Confidence Ellipses (Lobanov)", fontsize=14)
        plt.xlabel("F2 (Normalized)")
        plt.ylabel("F1 (Normalized)")
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300)
        plt.close()

    def plot_formant_boxplots(self, df, filename):
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        sns.boxplot(data=df, x="phoneme", y="f1_lob", hue="l1_status", ax=axes[0])
        axes[0].set_title("F1 Distribution by Phoneme and L1 Status")

        sns.boxplot(data=df, x="phoneme", y="f2_lob", hue="gender", ax=axes[1])
        axes[1].set_title("F2 Distribution by Phoneme and Gender")

        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()

    def plot_variability(self, df, filename):
        subset_speakers = df["speaker_id"].unique()[:3]
        subset_vowels = ["i", "a", "u"]
        data = df[
            df["speaker_id"].isin(subset_speakers) & df["phoneme"].isin(subset_vowels)
        ]

        plt.figure(figsize=(10, 6))
        sns.violinplot(
            data=data, x="phoneme", y="f1_lob", hue="speaker_id", inner="stick"
        )
        plt.title("Intra-speaker Variability (F1)")
        plt.savefig(self.output_dir / filename)
        plt.close()

    def plot_vowel_space(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: str,
        filename: str,
        hue_col: str = "phoneme",
        invert_axes: bool = True,
    ):
        plt.figure(figsize=(10, 8))

        plot = sns.scatterplot(
            data=df, x=x_col, y=y_col, hue=hue_col, style="l1_status", alpha=0.5, s=60
        )

        centroids = (
            df.groupby(["phoneme", "l1_status"])[[x_col, y_col]].mean().reset_index()
        )
        sns.scatterplot(
            data=centroids,
            x=x_col,
            y=y_col,
            hue="phoneme",
            style="l1_status",
            s=200,
            edgecolor="black",
            legend=False,
            marker="X",
        )

        if invert_axes:
            plt.gca().invert_xaxis()
            plt.gca().invert_yaxis()

        plt.title(title, fontsize=15)
        plt.xlabel(f"Normalized {x_col}", fontsize=12)
        plt.ylabel(f"Normalized {y_col}", fontsize=12)
        plt.legend(
            bbox_to_anchor=(1.05, 1), loc="upper left", title="Phoneme / L1 Status"
        )

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300)
        plt.close()
