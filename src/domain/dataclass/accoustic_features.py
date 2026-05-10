from dataclasses import dataclass


@dataclass(frozen=True)
class AcousticFeatures:
    f1: float | None = None
    f2: float | None = None
    f3: float | None = None
    f0: float | None = None
    scg: float | None = None
    f1_trajectory: dict[str, float] | None = None
    f2_trajectory: dict[str, float] | None = None
