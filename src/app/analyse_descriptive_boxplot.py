import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def generate_boxplots():
    df = pd.read_csv("data/processed/final_analysis_table.csv")
    
    vowels = ['i', 'e', 'ɛ', 'a', 'y', 'ø', 'u', 'o', 'ɑ']
    df_v = df[df['phoneme'].isin(vowels)].copy()
    
    df_v['Group'] = df_v['l1_status'] + "_" + df_v['gender']
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    sns.boxplot(data=df_v, x='phoneme', y='f1_lob', hue='Group', ax=axes[0], order=vowels)
    axes[0].set_title("Box plots of F1 (Lobanov) per phoneme")
    
    sns.boxplot(data=df_v, x='phoneme', y='f2_lob', hue='Group', ax=axes[1], order=vowels)
    axes[1].set_title("Box plots of F2 (Lobanov) per phoneme")
    
    plt.tight_layout()
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    plt.savefig("results/figures/boxplots_f1_f2.png", dpi=300)

if __name__ == "__main__":
    generate_boxplots()