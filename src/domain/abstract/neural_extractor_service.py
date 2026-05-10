from abc import ABC, abstractmethod
import numpy as np

from src.domain.model.phoneme_token import PhonemeToken


class NeuralExtractorService(ABC):

    @abstractmethod
    def extract_batch(
        self, tokens: list[PhonemeToken], audio_path: str, layers: list[int]
    ) -> dict[str, np.ndarray]:
        pass
