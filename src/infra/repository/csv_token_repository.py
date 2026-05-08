import pandas as pd
from pathlib import Path
from collections.abc import Iterable

from src.domain.abstract.token_repository import TokenRepository
from src.domain.dataclass.speaker_context import SpeakerContext
from src.domain.model.phoneme_token import PhonemeToken

class CSVTokenRepository(TokenRepository):
    def __init__(self, output_path: Path):
        self.output_path = output_path

    def save_tokens(self, tokens: Iterable[PhonemeToken]) -> None:
        data = []
        for t in tokens:
            row = {
                "token_id": t.token_id,
                "speaker_id": t.speaker.speaker_id,
                "sentence_id": t.sentence_id,
                "file_path": t.file_path,
                "word": t.word,
                "repetition": t.repetition_index,
                "phoneme": t.label,
                "onset": t.onset,
                "offset": t.offset,
                "duration": t.duration_ms,
                "l1_status": t.speaker.l1_status,
                "gender": t.speaker.gender
            }
            
            if t.acoustic:
                row.update({
                    "f1": t.acoustic.f1,
                    "f2": t.acoustic.f2,
                    "f3": t.acoustic.f3,
                    "f0": t.acoustic.f0,
                })
                if t.acoustic.f1_trajectory:
                    row["f1_25"] = t.acoustic.f1_trajectory.get("25%")
                    row["f1_75"] = t.acoustic.f1_trajectory.get("75%")
                    row["f2_25"] = t.acoustic.f2_trajectory.get("25%")
                    row["f2_75"] = t.acoustic.f2_trajectory.get("75%")

            data.append(row)
        
        df = pd.DataFrame(data)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, index=False)

    def load_all_tokens(self) -> list[PhonemeToken]:
        if not self.output_path.exists():
            return []
        
        df = pd.read_csv(self.output_path)
        tokens = []
        for _, row in df.iterrows():
            context = SpeakerContext(
                speaker_id=row['speaker_id'],
                l1_status=row['l1_status'],
                gender=row['gender']
            )
            tokens.append(PhonemeToken(
                token_id=row['token_id'],
                label=row['phoneme'],
                sentence_id=row['sentence_id'],
                file_path=row['file_path'],
                repetition_index=row['repetition'],
                word=row['word'],
                onset=row['onset'],
                offset=row['offset'],
                speaker=context
            )
        )
        return tokens

    def update_acoustic_features(self, token_id: str, features: dict) -> None:
        raise NotImplementedError

    def save_neural_representations(self, model_name: str, layer: int, representations: dict) -> None:
        raise NotImplementedError