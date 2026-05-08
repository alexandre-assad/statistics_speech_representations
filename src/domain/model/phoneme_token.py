from dataclasses import dataclass

from src.domain.dataclass.speaker_context import SpeakerContext
from src.domain.dataclass.accoustic_features import AcousticFeatures

@dataclass
class PhonemeToken:
    token_id: str             
    label: str               
    word: str
    sentence_id: str          
    repetition_index: int     
    onset: float              
    offset: float      
    file_path: str       
    speaker: SpeakerContext 
    acoustic: AcousticFeatures | None = None  
    
    @property
    def duration_ms(self) -> float:
        return (self.offset - self.onset) * 1000

    def is_long_vowel(self) -> bool:
        return self.duration_ms > 80