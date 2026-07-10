from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class EventTimeline:
    setup: int | None = None
    first_pull: int | None = None
    transition: int | None = None
    power_position: int | None = None
    extension: int | None = None
    pull_under: int | None = None
    catch: int | None = None
    recovery: int | None = None

    confidence: float = 0.0
    version: str = "clean_events_v2_shadow"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
