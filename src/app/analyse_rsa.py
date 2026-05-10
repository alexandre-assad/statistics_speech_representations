import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from pathlib import Path

from src.domain.model.ras_service import RSAService

def run_rsa_full():   
    Path("results/figures").mkdir(parents=True, exist_ok=True) 
    df = pd.read_csv("data/processed/final_analysis_table.csv")
    wh_raw = np.load("data/processed/features_whisper.npz", allow_pickle=True)
    xl_raw = np.load("data/processed/features_xlsr.npz", allow_pickle=True)
    
    vowels = ['i', 'e', 'ɛ', 'a', 'y', 'ø', 'u', 'o', 'ɑ']
    df = df[df['phoneme'].isin(vowels)].copy()
    
    centroids_ac = []
    centroids_wh = []
    centroids_xl = []
    labels = []
    
    for (speaker, phoneme), group in df.groupby(['speaker_id', 'phoneme']):
        f1_mean = group['f1_lob'].mean()
        f2_mean = group['f2_lob'].mean()
        
        tids = group['token_id'].astype(str).values
        
        wh_vecs = [wh_raw[t].item().get('layer_12') for t in tids if str(t) in wh_raw and wh_raw[t].item().get('layer_12') is not None]
        xl_vecs = [xl_raw[t].item().get('layer_12') for t in tids if str(t) in xl_raw and xl_raw[t].item().get('layer_12') is not None]
        
        if not np.isnan(f1_mean) and len(wh_vecs) > 0 and len(xl_vecs) > 0:
            centroids_ac.append([f1_mean, f2_mean])
            centroids_wh.append(np.mean(wh_vecs, axis=0))
            centroids_xl.append(np.mean(xl_vecs, axis=0))
            labels.append(f"{phoneme}_{speaker}")

    centroids_ac = np.array(centroids_ac)
    centroids_wh = np.array(centroids_wh)
    centroids_xl = np.array(centroids_xl)
    
    service = RSAService()
    rsm_ac = service.compute_rsm(centroids_ac, metric='euclidean', is_neural=False)
    rsm_wh = service.compute_rsm(centroids_wh, metric='cosine', is_neural=True)
    rsm_xl = service.compute_rsm(centroids_xl, metric='cosine', is_neural=True)

    r_ac_wh, p_ac_wh = service.mantel_test(rsm_ac, rsm_wh)
    r_ac_xl, p_ac_xl = service.mantel_test(rsm_ac, rsm_xl)
    r_wh_xl, p_wh_xl = service.mantel_test(rsm_wh, rsm_xl)

    print(f"1. Acoustic vs Whisper (Dac vs DWh) : r = {r_ac_wh:.4f}, p = {p_ac_wh:.4f}")
    print(f"2. Acoustic vs XLS-R   (Dac vs DXL) : r = {r_ac_xl:.4f}, p = {p_ac_xl:.4f}")
    print(f"3. Whisper vs XLS-R    (DWh vs DXL) : r = {r_wh_xl:.4f}, p = {p_wh_xl:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.heatmap(rsm_ac, ax=axes[0], cmap='viridis', xticklabels=False, yticklabels=False)
    axes[0].set_title("RSM Acoustique (-Euclidean)")
    
    sns.heatmap(rsm_wh, ax=axes[1], cmap='magma', xticklabels=False, yticklabels=False)
    axes[1].set_title(f"RSM Whisper (Cosine Sim)\nMantel vs Ac: r={r_ac_wh:.3f}")
    
    sns.heatmap(rsm_xl, ax=axes[2], cmap='magma', xticklabels=False, yticklabels=False)
    axes[2].set_title(f"RSM XLS-R (Cosine Sim)\nMantel vs Wh: r={r_wh_xl:.3f}")
    
    plt.tight_layout()
    plt.savefig("results/figures/rsm_full_comparison.png")

if __name__ == "__main__":
    run_rsa_full()