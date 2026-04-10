from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_TITLE: str = "NIC OCR API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "FastAPI service that preprocesses NIC images with OpenCV "
        "and extracts text via EasyOCR."
    )
    DEBUG: bool = False
    MAX_IMAGE_SIZE_MB: float = 10.0
    ALLOWED_CONTENT_TYPES: list[str] = [
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp",
        "image/tiff",
    ]

    TARGET_WIDTH: int = 1200
    MIN_ACCEPTABLE_WIDTH: int = 900
    MAX_UPSCALE_FACTOR: float = 1.5
    CLAHE_TILE_SIZE: int = 8
    DENOISE_H: int = 5
    MORPH_KERNEL_SIZE: int = 2
    MIN_DOC_AREA: int = 30000
    MIN_SKEW_ANGLE_DEG: float = 1.0

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
