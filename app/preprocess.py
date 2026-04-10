from __future__ import annotations
import cv2
import numpy as np
from config import settings
from schemas import PipelineResult
import torch
from PIL import Image
from super_image import EdsrModel, ImageLoader


def assess_image(img_gray: np.ndarray) -> dict:
    """Compute quality signals to drive adaptive processing decisions."""
    laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    mean_brightness = float(np.mean(img_gray))
    std_brightness = float(np.std(img_gray))

    return {
        "sharpness": laplacian_var,  # <100 = blurry, >500 = sharp
        "brightness": mean_brightness,  # 0-255
        "contrast": std_brightness,  # low = flat/washed out
        "is_blurry": laplacian_var < 120,
        "is_dark": mean_brightness < 80,
        "is_bright": mean_brightness > 190,
        "is_low_contrast": std_brightness < 40,
    }


def to_gray(img: np.ndarray):
    if img.ndim == 2:
        return img, "grayscale: skipped (already single-channel)"
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), "grayscale: BGR -> GRAY"


def resize(img: np.ndarray, target_width: int):
    h, w = img.shape[:2]
    if w > target_width:
        scale = target_width / w
        resized = cv2.resize(
            img, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA
        )
        return resized, f"resize: downscale"

    if w == settings.MIN_ACCEPTABLE_WIDTH:
        return img, f"resize: skipped"

    # if w < target_width, we can consider upscaling but with strict limits to avoid quality loss
    max_w = int(w * settings.MAX_UPSCALE_FACTOR)
    new_w = min(settings.MIN_ACCEPTABLE_WIDTH, max_w)
    if new_w == w:
        return img, "resize: skipped (upscale constrained)"
    scale = new_w / w
    new_h = int(h * scale)
    up = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    # to soften pixelation without losing edges
    bilateral = cv2.bilateralFilter(up, d=5, sigmaColor=35, sigmaSpace=35)
    blurred = cv2.GaussianBlur(bilateral, (0, 0), 1.2)
    enhanced = cv2.addWeighted(bilateral, 1.35, blurred, -0.35, 0)
    return enhanced, f"resize: upscale {w}x{h} -> {new_w}x{new_h}"


def upscale_with_super_image(
    img: np.ndarray, upscale_factor: int = 2, device: str = "cpu"
) -> tuple[np.ndarray, str]:
    h, w = img.shape[:2]

    try:
        model = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=upscale_factor)
        model = model.to(device)
        model.eval()

        if img.ndim == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img

        # super_image's ImageLoader expects a PIL image
        pil_img = Image.fromarray(img_rgb)
        inputs = ImageLoader.load_image(pil_img)
        inputs = inputs.to(device)

        with torch.no_grad():
            upscaled_tensor = model(inputs)

        # Convert back to numpy
        upscaled = (
            upscaled_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
        ).astype(np.uint8)

        if img.ndim == 2:
            upscaled = cv2.cvtColor(upscaled, cv2.COLOR_RGB2GRAY)

        new_h, new_w = upscaled.shape[:2]
        return (
            upscaled,
            f"super-image: EDSR {upscale_factor}x {w}x{h} -> {new_w}x{new_h}",
        )

    except Exception as e:
        print(f"Warning: super-image failed ({e}), falling back to LANCZOS4")
        new_w, new_h = int(w * upscale_factor), int(h * upscale_factor)
        upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        return (
            upscaled,
            f"resize: LANCZOS4 upscale {w}x{h} -> {new_w}x{new_h} (fallback)",
        )


def denoise(img: np.ndarray, strength: int):
    if strength == 0:
        return img, "denoise: skipped (image is sharp)"
    denoised = cv2.fastNlMeansDenoising(img, None, strength, 7, 21)
    return denoised, f"denoise: NLM h={strength}"


def apply_clahe(img: np.ndarray, clip: float, tile: int):
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return c.apply(img), f"CLAHE: clip={clip}, tile={tile}x{tile}"


def brightness_gamma_correct(img: np.ndarray, brightness: float):
    gamma = 0.6 if brightness < 60 else 0.8
    inv = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, table), f"gamma: correction gamma={gamma:.1f}"


def sharpen(img: np.ndarray, strength: float):
    if strength == 0:
        return img, "sharpen: skipped"
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    sharpened = cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)
    return sharpened, f"sharpen: unsharp-mask strength={strength}"


def morphology(img: np.ndarray):
    k = settings.MORPH_KERNEL_SIZE
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
    result = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    return result, f"morphology: open+close (kernel={k}x{k})"


def preprocess(image_bytes: bytes) -> PipelineResult:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(
            "Could not decode image - unsupported format or corrupted data."
        )

    original_h, original_w = img.shape[:2]
    steps: list[str] = []
    step_images: list[tuple[str, np.ndarray]] = [("original", img.copy())]

    def record(result):
        new_img, step = result
        steps.append(step)
        step_images.append((step.split(":")[0], new_img.copy()))
        return new_img

    img = record(to_gray(img))
    quality = assess_image(img)

    h, w = img.shape[:2]
    if w < settings.MIN_ACCEPTABLE_WIDTH:
        try:
            img = record(upscale_with_super_image(img, upscale_factor=2, device="cpu"))
        except Exception as e:
            print(f"Super-image upscaling failed: {e}, using traditional resize")
            img = record(resize(img, settings.TARGET_WIDTH))
    else:
        img = record(resize(img, settings.TARGET_WIDTH))

    if quality["is_dark"]:
        img = record(brightness_gamma_correct(img, quality["brightness"]))

    denoise_strength = (
        0 if not quality["is_blurry"] else (8 if quality["sharpness"] < 50 else 5)
    )
    img = record(denoise(img, denoise_strength))

    if quality["is_low_contrast"] or quality["is_dark"]:
        clip = 4.0 if quality["contrast"] < 25 else 3.0
    else:
        clip = 1.5  # mild enhancement for already-decent images
    img = record(apply_clahe(img, clip=clip, tile=settings.CLAHE_TILE_SIZE))

    sharpen_strength = (
        0.0 if quality["sharpness"] > 500 else (0.7 if quality["is_blurry"] else 0.4)
    )
    img = record(sharpen(img, sharpen_strength))

    if quality["is_blurry"] or quality["is_low_contrast"]:
        img = record(morphology(img))
    else:
        steps.append("morphology: skipped (image quality sufficient)")

    processed_h, processed_w = img.shape[:2]
    return PipelineResult(
        image=img,
        steps_applied=steps,
        step_images=step_images,
        original_size=(original_w, original_h),
        processed_size=(processed_w, processed_h),
    )
