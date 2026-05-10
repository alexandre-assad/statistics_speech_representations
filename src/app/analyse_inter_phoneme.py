import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import mahalanobis
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
from pathlib import Path

def compute_pooled_covariance(df, feature_cols, class_col):
    cov_pooled = np.zeros((len(feature_cols), len(feature_cols)))
    total_n = len(df)
    k = df[class_col].nunique()
    
    for _, group in df.groupby(class_col):
        X = group[feature_cols].values
        if len(X) > 1:
            cov = np.cov(X, rowvar=False)
            cov_pooled += (len(X) - 1) * cov
            
    cov_pooled /= (total_n - k)
    return cov_pooled

def run_inter_phoneme():    
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv("data/processed/final_analysis_table.csv").dropna(subset=['f1_lob', 'f2_lob'])
    wh_raw = np.load("data/processed/features_whisper.npz", allow_pickle=True)
    vowels = sorted(['i', 'e', 'ɛ', 'a', 'y', 'ø', 'u', 'o', 'ɑ'])
    df = df[df['phoneme'].isin(vowels)].copy()

    print("\n--- 1. Accoustic matrix (Mahalanobis) ---")
    features = ['f1_lob', 'f2_lob']
    cov_pooled = compute_pooled_covariance(df, features, 'phoneme')
    inv_cov_pooled = np.linalg.inv(cov_pooled)
    
    centroids = df.groupby('phoneme')[features].mean()
    mahalanobis_matrix = pd.DataFrame(index=vowels, columns=vowels, dtype=float)
    
    for p1 in vowels:
        for p2 in vowels:
            d = mahalanobis(centroids.loc[p1].values, centroids.loc[p2].values, inv_cov_pooled)
            mahalanobis_matrix.loc[p1, p2] = d
            
    print("Mahalanobis matrix computed")
    mahalanobis_matrix.to_csv("results/mahalanobis_distance_matrix.csv")

    print("\n--- 2. Bootstrap CI (95%) on phoneme pair ---")
    pairs_to_test = [('e', 'ɛ'), ('y', 'u'), ('a', 'ɑ')]
    B = 1000 
    
    for p1, p2 in pairs_to_test:
        df_pair = df[df['phoneme'].isin([p1, p2])]
        speakers = df_pair['speaker_id'].unique()
        boot_dists = []
        
        for _ in range(B):
            samp_speakers = np.random.choice(speakers, size=len(speakers), replace=True)
            blocks = [df_pair[df_pair['speaker_id'] == s] for s in samp_speakers]
            boot_df = pd.concat(blocks)
            
            c1 = boot_df[boot_df['phoneme'] == p1][features].mean().values
            c2 = boot_df[boot_df['phoneme'] == p2][features].mean().values
            boot_dists.append(np.linalg.norm(c1 - c2))
            
        ci_lower, ci_upper = np.percentile(boot_dists, [2.5, 97.5])
        print(f"Pair {p1}-{p2} | Mean Dist: {np.mean(boot_dists):.2f} | 95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")

    print("\n--- 3. Leave-One-Speaker-Out Classification ---")
    
    X_wh, X_ac, y_all, groups = [], [], [], []
    features = ['f1_lob', 'f2_lob']
    
    for _, row in df.iterrows():
        tid = str(row['token_id'])
        if tid in wh_raw and wh_raw[tid].item().get('layer_12') is not None:
            X_wh.append(wh_raw[tid].item().get('layer_12'))
            X_ac.append([row['f1_lob'], row['f2_lob']]) 
            y_all.append(row['phoneme'])
            groups.append(row['speaker_id'])
            
    X_wh = np.array(X_wh)
    X_ac = np.array(X_ac)
    y_all = np.array(y_all)
    groups = np.array(groups)
    
    logo = LeaveOneGroupOut()
    clf = NearestCentroid()
    
    preds_ac = np.zeros_like(y_all, dtype=object)
    preds_wh = np.zeros_like(y_all, dtype=object)

    print(f"cross validation on  {len(np.unique(groups))} speakers")
    for train_idx, test_idx in logo.split(X_ac, y_all, groups):
        clf.fit(X_ac[train_idx], y_all[train_idx])
        preds_ac[test_idx] = clf.predict(X_ac[test_idx])
        
        clf.fit(X_wh[train_idx], y_all[train_idx])
        preds_wh[test_idx] = clf.predict(X_wh[test_idx])

    acc_ac = accuracy_score(y_all, preds_ac)
    acc_wh = accuracy_score(y_all, preds_wh)
    f1_ac = f1_score(y_all, preds_ac, average='macro')
    f1_wh = f1_score(y_all, preds_wh, average='macro')
    
    print(f"Acoustique -> Accuracy: {acc_ac:.1%}, Macro-F1: {f1_ac:.3f}")
    print(f"Whisper    -> Accuracy: {acc_wh:.1%}, Macro-F1: {f1_wh:.3f}")

    ac_correct = (preds_ac == y_all)
    wh_correct = (preds_wh == y_all)
    
    both_correct = np.sum(ac_correct & wh_correct)
    ac_only = np.sum(ac_correct & ~wh_correct)
    wh_only = np.sum(~ac_correct & wh_correct)
    both_wrong = np.sum(~ac_correct & ~wh_correct)
    
    table = [[both_correct, ac_only],
             [wh_only, both_wrong]]
             
    result = mcnemar(table, exact=False, correction=True)
    print(f"McNemar p-value = {result.pvalue:.4e}")
    if result.pvalue < 0.05:
        print("The perf difference is statisticaly significative.")

    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_all, preds_wh, labels=vowels, normalize='true')
    sns.heatmap(cm, annot=True, xticklabels=vowels, yticklabels=vowels, cmap='Blues', fmt='.2f')
    plt.title(f"Whisper Confusion Matrix (LOSO Acc: {acc_wh:.1%})")
    plt.savefig("results/figures/loso_whisper_confusion.png")

if __name__ == "__main__":
    run_inter_phoneme()