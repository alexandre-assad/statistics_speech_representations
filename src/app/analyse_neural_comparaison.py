import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from src.domain.model.ras_service import RSAService
from pathlib import Path

def compute_metrics(X_2d, labels):
    overall_mean = np.mean(X_2d, axis=0)
    between_var = 0
    total_var = np.var(X_2d, axis=0).sum() * len(X_2d)
    
    for label in np.unique(labels):
        cluster = X_2d[labels == label]
        cluster_mean = np.mean(cluster, axis=0)
        between_var += len(cluster) * np.sum((cluster_mean - overall_mean)**2)
    
    return between_var / total_var

def run_neural_comparison():    
    df = pd.read_csv("data/processed/final_analysis_table.csv")
    wh_raw = np.load("data/processed/features_whisper.npz", allow_pickle=True)
    xl_raw = np.load("data/processed/features_xlsr.npz", allow_pickle=True)

    Path("results/figures").mkdir(parents=True, exist_ok=True)
    
    def get_matrix(raw_data, layer='layer_12'):
        X, y = [], []
        for _, row in df.iterrows():
            tid = str(row['token_id'])
            if tid in raw_data:
                vec = raw_data[tid].item().get(layer)
                if vec is not None:
                    X.append(vec)
                    y.append(row['phoneme'])
        return np.array(X), np.array(y)

    results = []
    models = {'Whisper': wh_raw, 'XLS-R': xl_raw}
    methods = {'PCA': PCA(n_components=2), 'UMAP': UMAP(n_components=2, n_neighbors=15, min_dist=0.1)}

    for model_name, raw in models.items():
        X, y = get_matrix(raw)
        
        for method_name, transformer in methods.items():
            print(f"Processing {model_name} with {method_name}...")
            X_2d = transformer.fit_transform(X)
            
            var_ratio = compute_metrics(X_2d, y)
            
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=y, palette='viridis', s=10)
            plt.title(f"{model_name} - {method_name}\nBetween-class Var Ratio: {var_ratio:.3f}")
            plt.savefig(f"results/figures/projection_{model_name.lower()}_{method_name.lower()}.png")
            plt.close()
            
            results.append({
                'Model': model_name,
                'Method': method_name,
                'Var_Ratio_2D': var_ratio
            })

    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    res_df.to_csv("results/neural_comparison_metrics.csv", index=False)

if __name__ == "__main__":
    run_neural_comparison()