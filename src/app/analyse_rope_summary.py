import pandas as pd
from pathlib import Path

def generate_rope_summary_table():
    try:
        ac_df = pd.read_csv("results/lme_confidence_intervals.csv") 
        ne_df = pd.read_csv("results/neural_rope_results.csv")
    except FileNotFoundError as e:
        print(f"File missing : {e}")
        return

    summary = []

    ac_f1 = ac_df[ac_df['Feature'] == 'F1_LOB']
    for _, row in ac_f1.iterrows():
        rope_limit = 0.4 
        ci_low, ci_upper = row['CI_Lower'], row['CI_Upper']
        
        if ci_low >= -rope_limit and ci_upper <= rope_limit:
            classif = "Equivalent"
        elif ci_upper < -rope_limit or ci_low > rope_limit:
            classif = "Non-equivalent"
        else:
            classif = "Indeterminate"

        summary.append({
            'Phoneme': row['Phoneme'],
            'Representation': 'Acoustic (F1)',
            'Point Estimate': round(row['Estimate'], 3),
            'CI 95%': f"[{ci_low:.3f}, {ci_upper:.3f}]",
            'ROPE Classification': classif
        })

    for _, row in ne_df.iterrows():
        summary.append({
            'Phoneme': row['Phoneme'],
            'Representation': row['Model'], 
            'Point Estimate': round(row['Estimate'], 3),
            'CI 95%': f"[{row['CI_Lower']:.3f}, {row['CI_Upper']:.3f}]",
            'ROPE Classification': row['Class']
        })

    summary_df = pd.DataFrame(summary)
    
    summary_df = summary_df.sort_values(by=['Phoneme', 'Representation'])
    
    summary_df.to_csv("results/final_rope_summary_table.csv", index=False)
    
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    generate_rope_summary_table()