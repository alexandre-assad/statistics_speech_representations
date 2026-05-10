import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from pathlib import Path
import time

def compute_neural_rope_baseline(df, neural_dict, layer_name='layer_12'):
    intra_dists = []
    for (speaker, phoneme), group in df.groupby(['speaker_id', 'phoneme']):
        tids = group['token_id'].astype(str).values
        vecs = [neural_dict[t].item().get(layer_name) for t in tids if str(t) in neural_dict and neural_dict[t].item().get(layer_name) is not None]
        
        if len(vecs) > 1:
            for i in range(len(vecs)):
                for j in range(i+1, len(vecs)):
                    intra_dists.append(cosine(vecs[i], vecs[j]))
                    
    return np.mean(intra_dists) if intra_dists else 0.05 

def bootstrap_neural_distance(df_p, neural_dict, B=2000, layer_name='layer_12'):
    speaker_centroids = {}
    speaker_l1 = {}
    
    for speaker, group in df_p.groupby('speaker_id'):
        tids = group['token_id'].astype(str).values
        vecs = [neural_dict[t].item().get(layer_name) for t in tids if str(t) in neural_dict and neural_dict[t].item().get(layer_name) is not None]
        
        if len(vecs) > 0:
            speaker_centroids[speaker] = np.mean(vecs, axis=0)
            speaker_l1[speaker] = group['l1_status'].iloc[0]
            
    fr_speakers = [s for s, l1 in speaker_l1.items() if l1 == 'fr']
    ru_speakers = [s for s, l1 in speaker_l1.items() if l1 == 'ru']
    
    if len(fr_speakers) < 2 or len(ru_speakers) < 2:
        return np.nan, np.nan, np.nan
        
    boot_dists = []
    for _ in range(B):
        samp_fr = np.random.choice(fr_speakers, size=len(fr_speakers), replace=True)
        samp_ru = np.random.choice(ru_speakers, size=len(ru_speakers), replace=True)
        
        c_fr = np.mean([speaker_centroids[s] for s in samp_fr], axis=0)
        c_ru = np.mean([speaker_centroids[s] for s in samp_ru], axis=0)
        
        boot_dists.append(cosine(c_fr, c_ru))
        
    return np.percentile(boot_dists, 2.5), np.mean(boot_dists), np.percentile(boot_dists, 97.5)

def run_neural_rope_forest_plot():
    t0 = time.time()
    
    df = pd.read_csv("data/processed/final_analysis_table.csv")
    wh_raw = np.load("data/processed/features_whisper.npz", allow_pickle=True)
    xl_raw = np.load("data/processed/features_xlsr.npz", allow_pickle=True)
    
    vowels = sorted(['i', 'e', 'ɛ', 'a', 'y', 'ø', 'u', 'o', 'ɑ'])
    df = df[df['phoneme'].isin(vowels)].copy()

    results = []
    models = [('Whisper', wh_raw), ('XLS-R', xl_raw)]
    
    for model_name, raw_data in models:
        print(f"\n Computing for {model_name}...")
        
        delta_0 = compute_neural_rope_baseline(df, raw_data)
        print(f"ROPE [0, {delta_0:.4f}]")
        
        for p in vowels:
            df_p = df[df['phoneme'] == p]
            ci_low, mean_dist, ci_high = bootstrap_neural_distance(df_p, raw_data, B=2000)
            
            if np.isnan(ci_low):
                classif = "NaN"
            elif ci_low > delta_0:
                classif = "Non-equivalent" 
            elif ci_high <= delta_0:
                classif = "Equivalent"     
            else:
                classif = "Indeterminate"
                
            results.append({
                'Model': model_name,
                'Phoneme': p,
                'ROPE_Limit': delta_0,
                'Estimate': mean_dist,
                'CI_Lower': ci_low,
                'CI_Upper': ci_high,
                'Class': classif
            })

    res_df = pd.DataFrame(results).dropna()
    res_df.to_csv("results/neural_rope_results.csv", index=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    
    for i, model_name in enumerate(['Whisper', 'XLS-R']):
        ax = axes[i]
        subset = res_df[res_df['Model'] == model_name]
        
        y_pos = np.arange(len(subset))
        delta_0 = subset['ROPE_Limit'].iloc[0] if len(subset) > 0 else 0
        
        ax.axvspan(0, delta_0, color='grey', alpha=0.2, label='ROPE (Noise Floor)')
        ax.axvline(delta_0, color='black', linestyle='--', linewidth=1)
        
        for j, (_, row) in enumerate(subset.iterrows()):
            color = 'red' if row['Class'] == 'Non-equivalent' else ('green' if row['Class'] == 'Equivalent' else 'orange')
            ax.plot([row['CI_Lower'], row['CI_Upper']], [j, j], color=color, lw=2)
            ax.plot(row['Estimate'], j, 'ko', markersize=6)
            
        ax.set_yticks(y_pos)
        ax.set_yticklabels(subset['Phoneme'], fontsize=12)
        ax.set_title(f"{model_name}\nDistance Cosinus L1 vs L2", fontsize=14)
        ax.set_xlabel("Distance Cosinus", fontsize=12)
        if i == 0: ax.legend()

    plt.tight_layout()
    plt.savefig("results/figures/neural_rope_forest_plot.png", dpi=300)

if __name__ == "__main__":
    run_neural_rope_forest_plot()