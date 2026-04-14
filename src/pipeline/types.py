"""Data types returned by the tire extraction pipeline."""

from typing import Dict, List, Optional
from dataclasses import dataclass, field

from schemas.tire_info import TireInfo


@dataclass
class TextDetectionResult:
    """A single detected + recognized text region."""
    text: str
    bbox: List[int]
    confidence: float


@dataclass
class PipelineResult:
    """Result of the tire extraction pipeline."""
    tire_info: TireInfo
    text_detections: List[TextDetectionResult] = field(default_factory=list)
    output_files: Dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineStepEvent:
    """Incremental progress event emitted by the streaming pipeline."""
    name: str
    duration: float
    output_files: Dict[str, str] = field(default_factory=dict)
    text_detections: List[TextDetectionResult] = field(default_factory=list)
    tire_info: Optional[TireInfo] = None
