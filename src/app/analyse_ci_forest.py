import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from pathlib import Path

def run_lme_forest_plot():    
    df = pd.read_csv("data/processed/final_analysis_table.csv").dropna(subset=['f1_lob', 'f2_lob'])
    vowels = sorted(['i', 'e', 'ɛ', 'a', 'y', 'ø', 'u', 'o', 'ɑ'])
    df = df[df['phoneme'].isin(vowels)].copy()

    results = []
    
    formula = "{} ~ C(l1_status) + C(gender)"
    
    for feature in ['f1_lob', 'f2_lob']:
        for p in vowels:
            df_p = df[df['phoneme'] == p]
            
            if len(df_p['l1_status'].unique()) < 2: continue
                
            try:
                md = smf.mixedlm(formula.format(feature), df_p, groups=df_p["speaker_id"])
                mdf = md.fit(method='cg', reml=False)
                
                target_coef = 'C(l1_status)[T.ru]'
                
                if target_coef in mdf.params:
                    estimate = mdf.params[target_coef]
                    ci_lower = mdf.conf_int().loc[target_coef, 0]
                    ci_upper = mdf.conf_int().loc[target_coef, 1]
                    
                    results.append({
                        'Feature': feature.upper(),
                        'Phoneme': p,
                        'Estimate': estimate,
                        'CI_Lower': ci_lower,
                        'CI_Upper': ci_upper
                    })
            except Exception as e:
                print(f"Fail of LME for {p} on {feature} : {e}")

    res_df = pd.DataFrame(results)
    res_df.to_csv("results/lme_confidence_intervals.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    
    for i, feature in enumerate(['F1_LOB', 'F2_LOB']):
        ax = axes[i]
        subset = res_df[res_df['Feature'] == feature]
        
        y_pos = np.arange(len(subset))
        
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, zorder=1)
        
        for j, (_, row) in enumerate(subset.iterrows()):
            color = 'red' if (row['CI_Lower'] > 0 or row['CI_Upper'] < 0) else 'grey'
            ax.plot([row['CI_Lower'], row['CI_Upper']], [j, j], color=color, lw=2, zorder=2)
            ax.plot(row['Estimate'], j, 'ko', markersize=6, zorder=3)
            
        ax.set_yticks(y_pos)
        ax.set_yticklabels(subset['Phoneme'], fontsize=12)
        ax.set_title(f"Effect of L1 (Russe) on {feature}\n", fontsize=14)
        ax.set_xlabel("Contrast estimated (Lobanov Z-score)", fontsize=12)
        ax.grid(axis='x', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig("results/figures/lme_forest_plot.png", dpi=300)

if __name__ == "__main__":
    run_lme_forest_plot()