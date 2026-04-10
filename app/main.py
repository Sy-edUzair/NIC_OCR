from __future__ import annotations
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from config import settings
from ocr import process_nic_image
from schemas import ErrorResponse, OCRResponse


app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
)


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


def _validate_upload(image: UploadFile, image_bytes: bytes) -> None:
    if image.content_type not in settings.ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported content type: {image.content_type}. "
                f"Allowed types: {', '.join(settings.ALLOWED_CONTENT_TYPES)}"
            ),
        )

    max_size_bytes = int(settings.MAX_IMAGE_SIZE_MB * 1024 * 1024)
    if len(image_bytes) > max_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds max size of {settings.MAX_IMAGE_SIZE_MB} MB",
        )

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")


@app.post(
    "/v1/ocr/nic",
    response_model=OCRResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["ocr"],
)
async def extract_nic_text(image: UploadFile = File(...)) -> OCRResponse:
    image_bytes = await image.read()
    _validate_upload(image=image, image_bytes=image_bytes)

    try:
        return process_nic_image(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected OCR processing error: {exc}",
        ) from exc


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):
    payload = ErrorResponse(
        detail=str(exc.detail), error_code=f"HTTP_{exc.status_code}"
    )
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())
