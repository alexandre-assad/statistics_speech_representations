import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

def run_full_clustering():
    df = pd.read_csv("data/processed/final_analysis_table.csv")
    wh_raw = np.load("data/processed/features_whisper.npz", allow_pickle=True)
    xl_raw = np.load("data/processed/features_xlsr.npz", allow_pickle=True)
    
    vowels = ['i', 'e', 'ɛ', 'a', 'y', 'ø', 'u', 'o', 'ɑ']
    df_vowels = df[df['phoneme'].isin(vowels)]

    fb_dict = {'i': 'front', 'e': 'front', 'ɛ': 'front', 'a': 'front', 'y': 'front', 'ø': 'front', 
               'u': 'back', 'o': 'back', 'ɑ': 'back'}
    true_fb = [fb_dict[v] for v in vowels]
    
    hml_dict = {'i': 'high', 'y': 'high', 'u': 'high', 
                'e': 'mid', 'ɛ': 'mid', 'ø': 'mid', 'o': 'mid', 
                'a': 'low', 'ɑ': 'low'}
    true_hml = [hml_dict[v] for v in vowels]

    ac_data = df_vowels.groupby('phoneme')[['f1_lob', 'f2_lob']].mean().loc[vowels].values
    
    wh_data, xl_data = [], []
    for v in vowels:
        tids = df_vowels[df_vowels['phoneme'] == v]['token_id'].astype(str).values
        
        vecs_wh = [wh_raw[t].item().get('layer_12') for t in tids if str(t) in wh_raw and wh_raw[t].item().get('layer_12') is not None]
        wh_data.append(np.mean(vecs_wh, axis=0))
        
        vecs_xl = [xl_raw[t].item().get('layer_12') for t in tids if str(t) in xl_raw and xl_raw[t].item().get('layer_12') is not None]
        xl_data.append(np.mean(vecs_xl, axis=0))
        
    wh_data = np.array(wh_data)
    xl_data = np.array(xl_data)

    configs = [
        ('Acoustic', ac_data, 'euclidean', 'ward'),
        ('Whisper', wh_data, 'cosine', 'average'), 
        ('XLS-R', xl_data, 'cosine', 'average')
    ]
    
    results = []
    linkage_matrices = {}

    for name, data, metric, method in configs:
        distances = pdist(data, metric=metric)
        Z = linkage(distances, method=method)
        linkage_matrices[name] = Z
        
        pred_2 = fcluster(Z, t=2, criterion='maxclust')
        ari_fb = adjusted_rand_score(true_fb, pred_2)
        
        pred_3 = fcluster(Z, t=3, criterion='maxclust')
        ari_hml = adjusted_rand_score(true_hml, pred_3)
        
        results.append({
            'Representation': name,
            'ARI (Front/Back)': round(ari_fb, 3),
            'ARI (High/Mid/Low)': round(ari_hml, 3)
        })

    res_df = pd.DataFrame(results)
    print("\nBest ARI scores :")
    print(res_df.to_string(index=False))
    res_df.to_csv("results/clustering_ari_scores.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (name, Z) in enumerate(linkage_matrices.items()):
        dendrogram(Z, labels=vowels, ax=axes[i], leaf_font_size=12)
        axes[i].set_title(f"{name} Clustering")
    plt.tight_layout()
    plt.savefig("results/figures/dendrograms_all.png")

if __name__ == "__main__":
    run_full_clustering()