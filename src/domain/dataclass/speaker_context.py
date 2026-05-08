from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class SpeakerContext:
    speaker_id: str
    l1_status: Literal["L1", "L2"]
    gender: Literal["F", "M"]     
