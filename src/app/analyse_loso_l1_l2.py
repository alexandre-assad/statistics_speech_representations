import pandas as pd
import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score

def get_l1_l2_accuracy():
    df = pd.read_csv("data/processed/final_analysis_table.csv")
    wh_raw = np.load("data/processed/features_whisper.npz", allow_pickle=True)
    
    df = df.dropna(subset=['f1_lob', 'f2_lob'])
    vowels = ['i', 'e', 'ɛ', 'a', 'y', 'ø', 'u', 'o', 'ɑ']
    df = df[df['phoneme'].isin(vowels)].copy()
    
    X_ac = df[['f1_lob', 'f2_lob']].values
    y = df['phoneme'].values
    speakers = df['speaker_id'].values
    l1_status = df['l1_status'].values
    
    X_wh = np.array([wh_raw[str(t)].item().get('layer_12') for t in df['token_id']])
    
    preds_ac, preds_wh = np.zeros_like(y), np.zeros_like(y)
    
    for spk in np.unique(speakers):
        train_idx = (speakers != spk)
        test_idx = (speakers == spk)
        
        clf_ac = NearestCentroid().fit(X_ac[train_idx], y[train_idx])
        preds_ac[test_idx] = clf_ac.predict(X_ac[test_idx])
        
        clf_wh = NearestCentroid().fit(X_wh[train_idx], y[train_idx])
        preds_wh[test_idx] = clf_wh.predict(X_wh[test_idx])
        
    df['pred_ac'] = preds_ac
    df['pred_wh'] = preds_wh
    
    for l1 in ['fr', 'ru']:
        mask = (df['l1_status'] == l1)
        acc_ac = accuracy_score(df[mask]['phoneme'], df[mask]['pred_ac'])
        acc_wh = accuracy_score(df[mask]['phoneme'], df[mask]['pred_wh'])
        print(f"[{l1.upper()}] Accuracy Acoustic : {acc_ac:.1%} | Whisper : {acc_wh:.1%} | Gain : +{(acc_wh - acc_ac)*100:.1f} pts")

if __name__ == "__main__":
    get_l1_l2_accuracy()