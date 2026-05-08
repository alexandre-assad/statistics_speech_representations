import pandas as pd
from praatio import textgrid
from pathlib import Path

from src.domain.dataclass.speaker_context import SpeakerContext
from src.domain.model.phoneme_token import PhonemeToken


class TextGridParser:
    def __init__(self, metadata_path: Path):
        self.metadata = pd.read_csv(metadata_path, sep=';')

    def parse_file(self, word: str, tg_path: Path, speaker_id: str, sentence_id: str, repetition: int) -> list[PhonemeToken]:
        tg = textgrid.openTextgrid(tg_path, includeEmptyIntervals=False)

        tier = tg.getTier('phones') 

        spk_info = self.metadata[self.metadata['spk'] == speaker_id].iloc[0]
        context = SpeakerContext(
            speaker_id=speaker_id,
            l1_status=spk_info['L1'], 
            gender=spk_info['Gender']        
        )

        tokens = []
        
        for i, entry in enumerate(tier.entries):
            token_id = f"{speaker_id}_{sentence_id}_r{repetition}_p{i}"
            audio_rel_path = str(tg_path.with_suffix('.wav'))
            tokens.append(PhonemeToken(
                token_id=token_id,
                label=entry.label,
                sentence_id=sentence_id,
                repetition_index=repetition,
                word=word,
                file_path=audio_rel_path,
                onset=entry.start,
                offset=entry.end,
                speaker=context
            ))
        return tokens