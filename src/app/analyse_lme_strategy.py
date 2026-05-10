import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.decomposition import PCA
from pathlib import Path

def likelihood_ratio_test(model_restricted, model_full):
    llf_full = model_full.llf
    llf_restr = model_restricted.llf
    df_full = model_full.df_modelwc
    df_restr = model_restricted.df_modelwc
    
    lr_stat = -2 * (llf_restr - llf_full)
    df_diff = df_full - df_restr
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, p_value

def calculate_r_squared(model, df, response_var):
    fixed_predict = model.predict(df)
    var_f = np.var(fixed_predict)
    
    var_r = model.cov_re.iloc[0, 0]
    
    var_e = model.scale
    
    total_var = var_f + var_r + var_e
    r2_marginal = var_f / total_var
    r2_conditional = (var_f + var_r) / total_var
    
    return r2_marginal, r2_conditional

def assign_vowel_height(phoneme):
    if phoneme in ['i', 'y', 'u']: return 'high'
    if phoneme in ['a', 'ɑ']: return 'low'
    return 'mid'

def build_nested_models(df, response_var, name=""):
    print(f"\n{'='*50}\n Modélisation : {name}\n{'='*50}")
    
    fit_kwargs = {'method': 'cg', 'reml': False}

    m1 = smf.mixedlm(f"{response_var} ~ 1", df, groups=df["speaker_id"]).fit(**fit_kwargs)
    icc = m1.cov_re.iloc[0,0] / (m1.cov_re.iloc[0,0] + m1.scale)
    print(f"M1 (Null)      - ICC: {icc:.4f} | AIC: {m1.aic:.1f}")

    m2 = smf.mixedlm(f"{response_var} ~ C(l1_status) + C(gender)", df, groups=df["speaker_id"]).fit(**fit_kwargs)
    _, p_lrt12 = likelihood_ratio_test(m1, m2)
    print(f"M2 (Main)      - AIC: {m2.aic:.1f} | LRT p-value: {p_lrt12:.4e}")

    m3 = smf.mixedlm(f"{response_var} ~ C(l1_status) * C(gender)", df, groups=df["speaker_id"]).fit(**fit_kwargs)
    _, p_lrt23 = likelihood_ratio_test(m2, m3)
    print(f"M3 (Full)      - AIC: {m3.aic:.1f} | LRT p-value: {p_lrt23:.4e}")

    m4 = smf.mixedlm(f"{response_var} ~ C(l1_status) * C(gender) + C(height)", df, groups=df["speaker_id"]).fit(**fit_kwargs)
    _, p_lrt34 = likelihood_ratio_test(m3, m4)
    print(f"M4 (Extended)  - AIC: {m4.aic:.1f} | LRT p-value: {p_lrt34:.4e}")

    best_model = m4
    r2_m, r2_c = calculate_r_squared(best_model, df, response_var)
    
    print(f"Best model : ({name}) :")
    print(f"Marginal R² : {r2_m:.3f} ({r2_m*100:.1f}%)")
    print(f"Conditional R² : {r2_c:.3f} ({r2_c*100:.1f}%)")
    
    return {'Name': name, 'R2_m': r2_m, 'R2_c': r2_c}

def run_lme_strategy():
    df = pd.read_csv("data/processed/final_analysis_table.csv")
    wh_raw = np.load("data/processed/features_whisper.npz", allow_pickle=True)
    xl_raw = np.load("data/processed/features_xlsr.npz", allow_pickle=True) # Ajout de XLS-R
    
    vowels = ['i', 'e', 'ɛ', 'a', 'y', 'ø', 'u', 'o', 'ɑ']
    df = df[df['phoneme'].isin(vowels)].copy()
    df['height'] = df['phoneme'].apply(assign_vowel_height)
    df = df.dropna(subset=['f1_lob'])

    pca = PCA(n_components=1)
    
    for model_name, raw_data, col_name in [('Whisper', wh_raw, 'PC1_wh'), ('XLS-R', xl_raw, 'PC1_xl')]:
        vecs, valid_idx = [], []
        for idx, row in df.iterrows():
            tid = str(row['token_id'])
            if tid in raw_data and raw_data[tid].item().get('layer_12') is not None:
                vecs.append(raw_data[tid].item().get('layer_12'))
                valid_idx.append(idx)
        
        pc1_vals = pca.fit_transform(np.array(vecs))
        df.loc[valid_idx, col_name] = pc1_vals[:, 0]

    df_clean = df.dropna(subset=['f1_lob', 'PC1_wh', 'PC1_xl']).copy()

    results = []
    results.append(build_nested_models(df_clean, 'f1_lob', "Acoustique (F1)"))
    results.append(build_nested_models(df_clean, 'PC1_wh', "Neural (Whisper)"))
    results.append(build_nested_models(df_clean, 'PC1_xl', "Neural (XLS-R)"))
    

    res_df = pd.DataFrame(results).round(3)
    print(res_df.to_string(index=False))
    res_df.to_csv("results/r2_comparison.csv", index=False)

if __name__ == "__main__":
    run_lme_strategy()