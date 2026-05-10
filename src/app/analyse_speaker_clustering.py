import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

def run_speaker_clustering():
    df = pd.read_csv("data/processed/final_analysis_table.csv")
    wh_raw = np.load("data/processed/features_whisper.npz", allow_pickle=True)
    xl_raw = np.load("data/processed/features_xlsr.npz", allow_pickle=True)
    
    vowels = sorted(['i', 'e', 'ɛ', 'a', 'y', 'ø', 'u', 'o', 'ɑ'])
    df = df[df['phoneme'].isin(vowels)].copy()

    speaker_info = df.groupby('speaker_id').agg({'l1_status': 'first', 'gender': 'first'}).reset_index()
    speakers = speaker_info['speaker_id'].values
    
    true_l1 = speaker_info['l1_status'].values
    true_gender = speaker_info['gender'].values

    global_ac = df.groupby('phoneme')[['f1_lob', 'f2_lob']].mean()
    
    global_wh, global_xl = {}, {}
    for p in vowels:
        tids = df[df['phoneme'] == p]['token_id'].astype(str).values
        wh_v = [wh_raw[t].item().get('layer_12') for t in tids if str(t) in wh_raw and wh_raw[t].item().get('layer_12') is not None]
        xl_v = [xl_raw[t].item().get('layer_12') for t in tids if str(t) in xl_raw and xl_raw[t].item().get('layer_12') is not None]
        global_wh[p] = np.mean(wh_v, axis=0) if wh_v else np.zeros(1024)
        global_xl[p] = np.mean(xl_v, axis=0) if xl_v else np.zeros(1024)

    X_ac, X_wh, X_xl = [], [], []
    
    for speaker in speakers:
        df_s = df[df['speaker_id'] == speaker]
        vec_ac, vec_wh, vec_xl = [], [], []
        
        for p in vowels:
            df_sp = df_s[df_s['phoneme'] == p]
            
            if not df_sp.empty and not pd.isna(df_sp['f1_lob'].mean()):
                vec_ac.extend([df_sp['f1_lob'].mean(), df_sp['f2_lob'].mean()])
            else:
                vec_ac.extend(global_ac.loc[p].values) # Imputation
                
            tids = df_sp['token_id'].astype(str).values
            
            wh_v = [wh_raw[t].item().get('layer_12') for t in tids if str(t) in wh_raw and wh_raw[t].item().get('layer_12') is not None]
            vec_wh.extend(np.mean(wh_v, axis=0) if wh_v else global_wh[p])
            
            xl_v = [xl_raw[t].item().get('layer_12') for t in tids if str(t) in xl_raw and xl_raw[t].item().get('layer_12') is not None]
            vec_xl.extend(np.mean(xl_v, axis=0) if xl_v else global_xl[p])
            
        X_ac.append(vec_ac)
        X_wh.append(vec_wh)
        X_xl.append(vec_xl)

    configs = [
        ('Acoustic (Lobanov)', np.array(X_ac), 'euclidean', 'ward'),
        ('Whisper', np.array(X_wh), 'cosine', 'average'),
        ('XLS-R', np.array(X_xl), 'cosine', 'average')
    ]
    
    results = []
    linkage_matrices = {}

    for name, data, metric, method in configs:
        Z = linkage(pdist(data, metric=metric), method=method)
        linkage_matrices[name] = Z
        
        pred_labels = fcluster(Z, t=2, criterion='maxclust')
        
        ari_l1 = adjusted_rand_score(true_l1, pred_labels)
        ari_gender = adjusted_rand_score(true_gender, pred_labels)
        
        results.append({
            'Representation': name,
            'ARI (L1 status)': round(ari_l1, 3),
            'ARI (Gender)': round(ari_gender, 3)
        })

    res_df = pd.DataFrame(results)
    print("\nARI Scores (Clustering des Locuteurs - Section 9.3) :")
    print(res_df.to_string(index=False))
    res_df.to_csv("results/speaker_clustering_ari.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (name, Z) in enumerate(linkage_matrices.items()):
        labels = [f"{l1}_{g}_{sp[-2:]}" for sp, l1, g in zip(speakers, true_l1, true_gender)]
        
        dendrogram(Z, labels=labels, ax=axes[i], leaf_font_size=9, orientation='top')
        axes[i].set_title(f"{name}\nARI(L1)={results[i]['ARI (L1 status)']} | ARI(Gen)={results[i]['ARI (Gender)']}")
        
    plt.tight_layout()
    plt.savefig("results/figures/dendrograms_speakers.png")

if __name__ == "__main__":
    run_speaker_clustering()