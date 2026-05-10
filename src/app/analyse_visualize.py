from pathlib import Path
import pandas as pd

from src.infra.visualisation.plotting_service import PlottingService


def run_neural_visualization(df, plot_service):
    plot_service.plot_vowel_space(
        df,
        x_col="PC1",
        y_col="PC2",
        title="Neural Space: Colored by Phoneme",
        filename="neural_space_phoneme.png",
        hue_col="phoneme",
        invert_axes=False,
    )

    plot_service.plot_vowel_space(
        df,
        x_col="PC1",
        y_col="PC2",
        title="Neural Space: Colored by L1 Status",
        filename="neural_space_l1.png",
        hue_col="l1_status",
        invert_axes=False,
    )

    plot_service.plot_vowel_space(
        df,
        x_col="PC1",
        y_col="PC2",
        title="Neural Space: Colored by Gender",
        filename="neural_space_gender.png",
        hue_col="gender",
        invert_axes=False,
    )


def run_visualization():
    data_path = Path("data/processed/final_analysis_table.csv")
    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)

    plot_service = PlottingService(Path("results/figures"))

    plot_service.plot_vowel_chart_with_ellipses(df, "vowel_chart_ellipses.png")

    plot_service.plot_formant_boxplots(df, "formant_distributions.png")

    plot_service.plot_variability(df, "intra_speaker_variability.png")

    run_neural_visualization(df, plot_service)

    plot_service.plot_vowel_space(
        df,
        x_col="f2_lob",
        y_col="f1_lob",
        title="Acoustic Vowel Space (Lobanov Z-scores)",
        filename="vowel_space_acoustic.png",
    )

    plot_service.plot_vowel_space(
        df,
        x_col="PC1",
        y_col="PC2",
        title="Neural Vowel Space (Whisper PCA - Layer 12)",
        filename="vowel_space_neural.png",
        invert_axes=False,
    )

    print("Visualizations successfully saved to results/figures/")


if __name__ == "__main__":
    run_visualization()
