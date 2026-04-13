"""Pydantic response models for the API."""

from typing import Dict, List
from enum import Enum

from pydantic import BaseModel


class TextDetectionResponse(BaseModel):
    text: str
    bbox: List[int]
    confidence: float


class FieldWithBBoxResponse(BaseModel):
    value: str
    source_bboxes: List[List[int]]


class TireInfoResponse(BaseModel):
    manufacturer: FieldWithBBoxResponse
    model: FieldWithBBoxResponse
    size: FieldWithBBoxResponse
    load_speed: FieldWithBBoxResponse
    dot: FieldWithBBoxResponse
    special_markings: List[FieldWithBBoxResponse]


class ExtractionMode(str, Enum):
    pipeline = "pipeline"
    llm_only = "llm_only"


class ExtractionResponse(BaseModel):
    mode: str
    tire_info: TireInfoResponse
    text_detections: List[TextDetectionResponse]
    output_files: Dict[str, str]
