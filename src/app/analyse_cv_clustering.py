import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def run_cv_clustering():
    try:
        df_ac = pd.read_csv("data/processed/features_acoustic.csv")
    except Exception:
        df_ac = pd.read_csv("data/processed/final_analysis_table.csv")
        
    wh_raw = np.load("data/processed/features_whisper.npz", allow_pickle=True)
    xl_raw = np.load("data/processed/features_xlsr.npz", allow_pickle=True)

    target_vowels = ['i', 'a', 'u', 'y', 'e', 'o'] 
    target_consonants = ['p', 't', 's', 'ʃ', 'm', 'l', 'ʁ'] 
    
    available_phonemes = df_ac['phoneme'].unique()
    vowels = [p for p in target_vowels if p in available_phonemes]
    consonants = [p for p in target_consonants if p in available_phonemes]
    all_phonemes = vowels + consonants
        
    df = df_ac[df_ac['phoneme'].isin(all_phonemes)].copy()

    cv_dict = {p: 'V' if p in vowels else 'C' for p in all_phonemes}
    true_cv_labels = [cv_dict[p] for p in all_phonemes]

    ac_data = []
    dur_col = 'duration_ms' if 'duration_ms' in df.columns else 'duration'
    
    for p in all_phonemes:
        df_p = df[df['phoneme'] == p]
        
        f1_m = df_p['f1'].mean() if 'f1' in df.columns else 0
        f2_m = df_p['f2'].mean() if 'f2' in df.columns else 0
        dur_m = df_p[dur_col].mean() if dur_col in df.columns else 0
        
        ac_data.append([
            0 if np.isnan(f1_m) else f1_m,
            0 if np.isnan(f2_m) else f2_m,
            0 if np.isnan(dur_m) else dur_m
        ])

    ac_data = np.array(ac_data)
    
    scaler = StandardScaler()
    ac_data_scaled = scaler.fit_transform(ac_data)

    wh_data, xl_data = [], []
    for p in all_phonemes:
        df_p = df[df['phoneme'] == p]
        tids = df_p['token_id'].astype(str).values
        
        v_wh = [wh_raw[t].item().get('layer_12') for t in tids if t in wh_raw and wh_raw[t].item().get('layer_12') is not None]
        wh_data.append(np.mean(v_wh, axis=0) if v_wh else np.zeros(1024))
        
        v_xl = [xl_raw[t].item().get('layer_12') for t in tids if t in xl_raw and xl_raw[t].item().get('layer_12') is not None]
        xl_data.append(np.mean(v_xl, axis=0) if v_xl else np.zeros(1024))
        
    wh_data = np.array(wh_data)
    xl_data = np.array(xl_data)

    configs = [
        ('Acoustic (Hybride)', ac_data_scaled, 'euclidean', 'ward'),
        ('Whisper', wh_data, 'cosine', 'average'), # Ward n'est pas possible avec Cosine
        ('XLS-R', xl_data, 'cosine', 'average')
    ]
    
    results = []
    linkage_matrices = {}

    for name, data, metric, method in configs:
        clean_data = np.nan_to_num(data)
        
        distances = pdist(clean_data, metric=metric)
        Z = linkage(distances, method=method)
        linkage_matrices[name] = Z
        
        pred_labels = fcluster(Z, t=2, criterion='maxclust')
        ari = adjusted_rand_score(true_cv_labels, pred_labels)
        results.append({'Representation': name, 'ARI (C vs V)': round(ari, 3)})

    print(pd.DataFrame(results).to_string(index=False))

    Path("results/figures").mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    for i, (name, Z) in enumerate(linkage_matrices.items()):
        labels_with_class = [f"{p} ({cv_dict[p]})" for p in all_phonemes]
        dendrogram(Z, labels=labels_with_class, ax=axes[i], leaf_font_size=10)
        axes[i].set_title(f"{name}\nARI (C vs V) = {results[i]['ARI (C vs V)']}")
        
    plt.tight_layout()
    plt.savefig("results/figures/dendrograms_cv.png", dpi=300)

if __name__ == "__main__":
    run_cv_clustering()