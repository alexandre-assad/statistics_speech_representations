import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path

def calculate_iqr(x):
    return np.percentile(x.dropna(), 75) - np.percentile(x.dropna(), 25)

def calculate_cv(x):
    mean_val = np.mean(x.dropna())
    return np.std(x.dropna(), ddof=1) / mean_val if mean_val != 0 else np.nan

def run_descriptive_stats():
    
    csv_path = "data/processed/final_analysis_table.csv"
    if not Path(csv_path).exists():
        print(f"Error : {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    vowels = ['i', 'e', 'ɛ', 'a', 'y', 'ø', 'u', 'o', 'ɑ']
    df_vowels = df[df['phoneme'].isin(vowels)].copy()
    
    df_vowels['group'] = df_vowels['l1_status'] + '_' + df_vowels['gender']

    print("\n--- PARTIE 1 : TABLEAU DESCRIPTIF (F1 & F2 Lobanov) ---")
    
    agg_funcs = {
        'f1_lob': ['mean', 'median', 'std', calculate_iqr, calculate_cv],
        'f2_lob': ['mean', 'median', 'std', calculate_iqr, calculate_cv]
    }
    
    stats_table = df_vowels.groupby(['phoneme', 'group']).agg(agg_funcs).round(3)
    
    stats_table.columns = ['F1_Mean', 'F1_Median', 'F1_SD', 'F1_IQR', 'F1_CV',
                           'F2_Mean', 'F2_Median', 'F2_SD', 'F2_IQR', 'F2_CV']
    
    stats_table.to_csv("results/descriptive_statistics_table.csv")
    print(stats_table.head(10))

    
    variance_results = []
    
    for phoneme in vowels:
        df_p = df_vowels[df_vowels['phoneme'] == phoneme].dropna(subset=['f1_lob'])
        
        if len(df_p) < 10: 
            continue
            
        md = smf.mixedlm("f1_lob ~ 1", df_p, groups=df_p["speaker_id"])
        mdf = md.fit(method='cg')
        
        var_inter = mdf.cov_re.iloc[0, 0] 
        var_intra = mdf.scale            
        var_total = var_inter + var_intra
        
        variance_results.append({
            'Phoneme': phoneme,
            'Total_Var': round(var_total, 4),
            'Inter_Speaker_Var': round(var_inter, 4),
            'Intra_Speaker_Var': round(var_intra, 4),
            '%_Inter': round((var_inter / var_total) * 100, 1),
            '%_Intra': round((var_intra / var_total) * 100, 1)
        })

    var_df = pd.DataFrame(variance_results)
    var_df.to_csv("results/variance_decomposition.csv", index=False)
    print(var_df.to_string(index=False))

if __name__ == "__main__":
    run_descriptive_stats()