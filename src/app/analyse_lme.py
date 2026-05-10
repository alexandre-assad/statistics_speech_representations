import pandas as pd
from src.domain.model.lme_service import LMEService


def run_lme_analysis():
    df = pd.read_csv("data/processed/final_analysis_table.csv")

    df_a = df[df["phoneme"] == "a"].copy()

    df_a = df_a.dropna(subset=["f1_lob", "PC1", "l1_status", "gender", "speaker_id"])

    service = LMEService()
    _, icc_ac = service.fit_null_model(df_a, "f1_lob")
    print(f"ICC Acoustic (F1 Lobanov) : {icc_ac:.4f}")

    _, icc_wh = service.fit_null_model(df_a, "PC1")
    print(f"ICC Neural (Whisper PC1)    : {icc_wh:.4f}")

    if icc_ac > icc_wh:
        print("The speaker has more variance in accoustic than in neural")
    else:
        print("The speaker has more variance in neural than in accoustic")

    res_ac = service.fit_full_model(df_a, "f1_lob")
    p_val_inter_ac = res_ac.pvalues["C(l1_status)[T.ru]:C(gender)[T.m]"]
    print(f"\nAcoustic (F1) - p-value interaction L1:Gender = {p_val_inter_ac:.4f}")
    print(res_ac.summary().tables[1])

    res_wh = service.fit_full_model(df_a, "PC1")
    p_val_inter_wh = res_wh.pvalues["C(l1_status)[T.ru]:C(gender)[T.m]"]
    print(f"\nNeural (PC1) - p-value interaction L1:Gender = {p_val_inter_wh:.4f}")
    print(res_wh.summary().tables[1])


if __name__ == "__main__":
    run_lme_analysis()
