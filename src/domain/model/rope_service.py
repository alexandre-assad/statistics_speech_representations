import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine


class ROPEService:
    @staticmethod
    def classify_rope(ci_lower, ci_upper, rope_lower, rope_upper):
        if ci_lower >= rope_lower and ci_upper <= rope_upper:
            return "Equivalent"
        elif ci_upper < rope_lower or ci_lower > rope_upper:
            return "Non-equivalent"
        else:
            return "Indeterminate"

    @staticmethod
    def bootstrap_acoustic_diff(df, phoneme, feature="f1_lob", B=2000):
        df_p = df[df["phoneme"] == phoneme]

        speaker_means = (
            df_p.groupby(["speaker_id", "l1_status"])[feature].mean().reset_index()
        )
        fr_speakers = speaker_means[speaker_means["l1_status"] == "fr"][
            "speaker_id"
        ].values
        ru_speakers = speaker_means[speaker_means["l1_status"] == "ru"][
            "speaker_id"
        ].values

        diffs = []
        for _ in range(B):
            samp_fr = np.random.choice(fr_speakers, size=len(fr_speakers), replace=True)
            samp_ru = np.random.choice(ru_speakers, size=len(ru_speakers), replace=True)

            mean_fr = speaker_means[speaker_means["speaker_id"].isin(samp_fr)][
                feature
            ].mean()
            mean_ru = speaker_means[speaker_means["speaker_id"].isin(samp_ru)][
                feature
            ].mean()

            diffs.append(mean_fr - mean_ru)

        return np.percentile(diffs, 2.5), np.mean(diffs), np.percentile(diffs, 97.5)

    @staticmethod
    def compute_neural_rope_baseline(df, neural_dict, layer_name="layer_12"):
        intra_dists = []
        for speaker, group in df.groupby("speaker_id"):
            for phoneme, p_group in group.groupby("phoneme"):
                tids = p_group["token_id"].astype(str).values
                vecs = [
                    neural_dict[t].item().get(layer_name)
                    for t in tids
                    if str(t) in neural_dict
                ]
                vecs = [v for v in vecs if v is not None]

                if len(vecs) > 1:
                    for i in range(len(vecs)):
                        for j in range(i + 1, len(vecs)):
                            intra_dists.append(cosine(vecs[i], vecs[j]))

        return np.mean(intra_dists)
