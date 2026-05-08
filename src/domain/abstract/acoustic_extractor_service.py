from abc import ABC, abstractmethod

from src.domain.model.phoneme_token import PhonemeToken
from src.domain.dataclass.accoustic_features import AcousticFeatures

class AcousticExtractorService(ABC):
    
    @abstractmethod
    def extract(self, token: PhonemeToken, audio_path: str) -> AcousticFeatures:
        pass