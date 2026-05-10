import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from scipy.spatial.distance import cosine
from pathlib import Path

def run_group_comparisons():
    
    df = pd.read_csv("data/processed/final_analysis_table.csv")
    wh_raw = np.load("data/processed/features_whisper.npz", allow_pickle=True)
    
    vowels = sorted(['i', 'e', 'ɛ', 'a', 'y', 'ø', 'u', 'o', 'ɑ'])
    df_vowels = df[df['phoneme'].isin(vowels)].copy()

    acoustic_results = []
    
    for feature in ['f1_lob', 'f2_lob']:
        for p in vowels:
            df_p = df_vowels[df_vowels['phoneme'] == p].dropna(subset=[feature])
            fr_data = df_p[df_p['l1_status'] == 'fr'][feature].values
            ru_data = df_p[df_p['l1_status'] == 'ru'][feature].values
            
            if len(fr_data) < 5 or len(ru_data) < 5: continue
            
            _, p_shap_fr = stats.shapiro(fr_data)
            _, p_shap_ru = stats.shapiro(ru_data)
            _, p_levene = stats.levene(fr_data, ru_data)
            
            is_normal = p_shap_fr > 0.05 and p_shap_ru > 0.05
            is_equal_var = p_levene > 0.05
            
            if is_normal and is_equal_var:
                test_name = "t-test"
                stat, p_val = stats.ttest_ind(fr_data, ru_data)
            else:
                test_name = "Mann-Whitney"
                stat, p_val = stats.mannwhitneyu(fr_data, ru_data, alternative='two-sided')
                
            acoustic_results.append({
                'Feature': feature, 'Phoneme': p, 'Test': test_name, 'p_uncorrected': p_val
            })

    ac_df = pd.DataFrame(acoustic_results)
    _, ac_df['p_FDR_BH'], _, _ = multipletests(ac_df['p_uncorrected'], alpha=0.05, method='fdr_bh')
    print(ac_df.to_string(index=False))
    ac_df.to_csv("results/acoustic_l1_l2_tests.csv", index=False)

    speaker_df = df_vowels.groupby(['speaker_id', 'gender'])[['f1_lob', 'f2_lob']].mean().reset_index()
    
    m_f1 = speaker_df[speaker_df['gender'] == 'm']['f1_lob'].values
    f_f1 = speaker_df[speaker_df['gender'] == 'f']['f1_lob'].values
    _, p_gender_f1 = stats.ttest_ind(m_f1, f_f1)
    
    m_f2 = speaker_df[speaker_df['gender'] == 'm']['f2_lob'].values
    f_f2 = speaker_df[speaker_df['gender'] == 'f']['f2_lob'].values
    _, p_gender_f2 = stats.ttest_ind(m_f2, f_f2)
    
    print(f"Test F1 (Man vs Woman) - p-value : {p_gender_f1:.4f}")
    print(f"Test F2 (Man vs Woman) - p-value : {p_gender_f2:.4f}")
    print("\n--- 3. Tests neural (Permutation Cosine, B=5000) ---")
    neural_results = []
    
    for p in vowels:
        df_p = df_vowels[df_vowels['phoneme'] == p]
        
        speaker_vecs = {}
        speaker_labels = {}
        
        for speaker, group in df_p.groupby('speaker_id'):
            tids = group['token_id'].astype(str).values
            vecs = [wh_raw[t].item().get('layer_12') for t in tids if str(t) in wh_raw and wh_raw[t].item().get('layer_12') is not None]
            if vecs:
                speaker_vecs[speaker] = np.mean(vecs, axis=0)
                speaker_labels[speaker] = group['l1_status'].iloc[0]
                
        speakers = list(speaker_vecs.keys())
        vectors = np.array([speaker_vecs[s] for s in speakers])
        labels = np.array([speaker_labels[s] for s in speakers])
        
        if len(set(labels)) < 2: continue
        
        def get_centroid_dist(lbls):
            c_fr = np.mean(vectors[lbls == 'fr'], axis=0)
            c_ru = np.mean(vectors[lbls == 'ru'], axis=0)
            return cosine(c_fr, c_ru)
            
        obs_dist = get_centroid_dist(labels)
        
        B = 5000
        count = 0
        for _ in range(B):
            permuted_labels = np.random.permutation(labels)
            perm_dist = get_centroid_dist(permuted_labels)
            if perm_dist >= obs_dist:
                count += 1
                
        p_val_perm = (count + 1) / (B + 1)
        neural_results.append({'Phoneme': p, 'Obs_Cosine_Dist': obs_dist, 'p_uncorrected': p_val_perm})

    ne_df = pd.DataFrame(neural_results)
    _, ne_df['p_FDR_BH'], _, _ = multipletests(ne_df['p_uncorrected'], alpha=0.05, method='fdr_bh')
    print(ne_df.to_string(index=False))
    ne_df.to_csv("results/neural_l1_l2_permutations.csv", index=False)

if __name__ == "__main__":
    run_group_comparisons()