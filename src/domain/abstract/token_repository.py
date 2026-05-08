from abc import ABC, abstractmethod
from collections.abc import Iterable

from src.domain.model.phoneme_token import PhonemeToken

class TokenRepository(ABC):

    @abstractmethod
    def save_tokens(self, tokens: Iterable[PhonemeToken]) -> None:
        pass

    @abstractmethod
    def load_all_tokens(self) -> list[PhonemeToken]:
        pass

    @abstractmethod
    def update_acoustic_features(self, token_id: str, features: dict) -> None:
        pass

    @abstractmethod
    def save_neural_representations(self, model_name: str, layer: int, representations: dict) -> None:
        pass