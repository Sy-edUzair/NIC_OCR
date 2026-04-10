from __future__ import annotations
from pydantic import BaseModel, Field
import numpy as np
from typing import NamedTuple


class NICFields(BaseModel):
    cnic_number: str = Field(default="Not Found")
    name: str = Field(default="Not Found")
    father_or_husband_name: str = Field(default="Not Found")
    gender: str = Field(default="Not Found")
    date_of_birth: str = Field(default="Not Found")
    country_of_stay: str = Field(default="Not Found")


class PipelineResult(NamedTuple):
    image: np.ndarray
    steps_applied: list[str]
    step_images: list[tuple[str, np.ndarray]]
    original_size: tuple[int, int]
    processed_size: tuple[int, int]


class ImageMetadata(BaseModel):
    original_size: tuple[int, int] = Field(
        description="(width, height) of the image as received"
    )
    processed_size: tuple[int, int] = Field(
        description="(width, height) after preprocessing"
    )
    preprocessing_steps: list[str] = Field(
        description="Ordered list of CV transforms applied to the image"
    )


class OCRMetadata(BaseModel):
    engine: str = Field(description="OCR engine used (e.g. 'tesseract')")
    language: str = Field(description="EasyOCR language detected")
    mean_confidence: float = Field(
        description="Mean per-word confidence (0–100). -1 if unavailable."
    )
    word_count: int = Field(description="Total number of words detected")
    extraction_status: str = Field(description="'success' or 'partial' or 'failed'")


class OCRResponse(BaseModel):
    raw_text: str = Field(description="Full extracted text block")
    lines: list[str] = Field(description="Text split into non-empty lines")
    nic_fields: NICFields = Field(
        description="Structured NIC fields extracted from OCR output"
    )
    image_metadata: ImageMetadata
    ocr_metadata: OCRMetadata
    warnings: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    detail: str
    error_code: str | None = None
