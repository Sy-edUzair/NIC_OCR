from __future__ import annotations
import re
import cv2
import easyocr
from preprocess import preprocess
from schemas import NICFields, OCRMetadata, OCRResponse

reader = None


def get_reader():
    global reader
    if reader is None:
        reader = easyocr.Reader(["en"], gpu=False, download_enabled=False)
    return reader


class OCRProcessingError(Exception):
    """Raised when OCR processing cannot complete successfully."""


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_cnic(cnic_like: str) -> str:
    digits = re.sub(r"\D", "", cnic_like)
    if len(digits) != 13:
        return "Not Found"
    return f"{digits[:5]}-{digits[5:12]}-{digits[12]}"


def extract_fields(text: str) -> NICFields:
    cleaned = clean_text(text)
    fields = NICFields()

    cnic_match = re.search(r"\b\d{5}[\-\.\s]?\d{7}[\-\.\s]?\d\b", cleaned)
    if cnic_match:
        fields.cnic_number = normalize_cnic(cnic_match.group(0))

    name_match = re.search(
        r"Name\s*[:\-]?\s*([A-Za-z\s]+?)(?=\s+(?:Father|Husband|Gender|Country|Date|CNIC|$))",
        cleaned,
        re.IGNORECASE,
    )
    if name_match:
        fields.name = name_match.group(1).strip()

    father_match = re.search(
        r"\b(?:Father(?:\s+Name)?|Husband(?:\s+Name)?)\s*[:\-]?\s*([A-Za-z\s]+?)(?=\s+(?:Gender|Country|Date|CNIC|Name|$))",
        cleaned,
        re.IGNORECASE,
    )
    if father_match:
        fields.father_or_husband_name = father_match.group(1).strip()

    gender_match = re.search(r"\b(Male|Female|M|F)\b", cleaned, re.IGNORECASE)
    if gender_match:
        g = gender_match.group(1).upper()
        fields.gender = "Male" if g in ("M", "MALE") else "Female"

    dob_labeled = re.search(
        r"Date\s*of\s*Birth\s*[:\-]?\s*([0-3]?\d[\.\-/][0-1]?\d[\.\-/](?:19|20)\d{2})",
        cleaned,
        re.IGNORECASE,
    )
    if dob_labeled:
        fields.date_of_birth = dob_labeled.group(1).replace("-", ".").replace("/", ".")
    else:
        dob_fallback = re.search(
            r"\b([0-3]?\d[\.\-/][0-1]?\d[\.\-/](?:19|20)\d{2})\b", cleaned
        )
        if dob_fallback:
            fields.date_of_birth = (
                dob_fallback.group(1).replace("-", ".").replace("/", ".")
            )

    country_match = re.search(
        r"Country(?:\s+of)?(?:\s+Stay)?\s*[:\-]?\s*([A-Za-z\s]+?)(?=\s+(?:Identity|Gender|Date|CNIC|Name|$))",
        cleaned,
        re.IGNORECASE,
    )
    if country_match:
        fields.country_of_stay = country_match.group(1).strip()

    return fields


def _build_extraction_status(
    raw_text: str, mean_confidence: float, word_count: int
) -> str:
    if not raw_text or word_count == 0:
        return "failed"
    if mean_confidence < 45.0:
        return "partial"
    return "success"


def process_nic_image(image_bytes: bytes):
    reader = get_reader()
    pipeline_result = preprocess(image_bytes)

    result = reader.readtext(pipeline_result.image)
    confidences = [item[2] for item in result]
    lines = [item[1].strip() for item in result if item[1].strip()]
    raw_text = "\n".join(lines)
    mean_conf = (
        round(sum(confidences) / len(confidences) * 100, 2) if confidences else -1.0
    )

    status = _build_extraction_status(raw_text, mean_conf, len(lines))
    warnings: list[str] = []
    if status != "success":
        warnings.append("OCR extraction is incomplete. Please provide a clearer image.")

    return OCRResponse(
        raw_text=raw_text,
        lines=lines,
        nic_fields=extract_fields(raw_text),
        image_metadata={
            "original_size": pipeline_result.original_size,
            "processed_size": pipeline_result.processed_size,
            "preprocessing_steps": pipeline_result.steps_applied,
        },
        ocr_metadata=OCRMetadata(
            engine="easyocr",
            language="en",
            mean_confidence=mean_conf,
            word_count=len(lines),
            extraction_status=status,
        ),
        warnings=warnings,
    )
