from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class RouterPrediction:
    router: str
    label: str | None
    confidence: float = 0.0
    reason: str = ""
    lock: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
