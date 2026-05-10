import parselmouth
import numpy as np
from src.domain.model.phoneme_token import PhonemeToken
from src.domain.dataclass.accoustic_features import AcousticFeatures
from src.domain.abstract.acoustic_extractor_service import AcousticExtractorService


class PraatAcousticExtractor(AcousticExtractorService):
    def extract(self, token: PhonemeToken, audio_path: str) -> AcousticFeatures:
        sound = parselmouth.Sound(str(audio_path))
        duration = token.offset - token.onset

        if duration < 0.01:
            return AcousticFeatures(f1=None, f2=None, f3=None, f0=None)

        sound_segment = sound.extract_part(from_time=token.onset, to_time=token.offset)

        max_formant = 5000 if token.speaker.gender == "F" else 4500
        formants = sound_segment.to_formant_burg(
            time_step=0.005, max_number_of_formants=5, maximum_formant=max_formant
        )

        rel_mid = duration / 2
        f1 = formants.get_value_at_time(1, rel_mid)
        f2 = formants.get_value_at_time(2, rel_mid)
        f3 = (
            formants.get_value_at_time(3, rel_mid)
            if any(v in token.label for v in "aeiouyøœ")
            else None
        )

        p_floor = 75 if duration > 0.04 else 160

        try:
            pitch = sound_segment.to_pitch(pitch_floor=p_floor)
            pitch_values = pitch.selected_array["frequency"]
            valid_pitch = pitch_values[pitch_values > 0]
            f0 = np.mean(valid_pitch) if len(valid_pitch) > 0 else None
        except:
            f0 = None

        f1_traj, f2_traj = None, None
        if duration > 0.08:
            t25, t75 = duration * 0.25, duration * 0.75
            f1_traj = {
                "25%": formants.get_value_at_time(1, t25),
                "75%": formants.get_value_at_time(1, t75),
            }
            f2_traj = {
                "25%": formants.get_value_at_time(2, t25),
                "75%": formants.get_value_at_time(2, t75),
            }

        return AcousticFeatures(
            f1=f1, f2=f2, f3=f3, f0=f0, f1_trajectory=f1_traj, f2_trajectory=f2_traj
        )
