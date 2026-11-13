"""Shared image I/O, paper-ratio padding, and binarization utilities."""

from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

ImageInput = Union[str, Path, Image.Image, np.ndarray]

PAPER_RATIOS: dict[str, tuple[float, float]] = {
    "a4": (210.0, 297.0),
    "postcard": (100.0, 148.0),
    "square": (1.0, 1.0),
}


def load_pil_rgb(image: ImageInput, *, bgr_input: bool = False) -> Image.Image:
    """Load an image from path, PIL Image, or numpy array as an RGB PIL Image.

    RGBA/transparent images are composited onto a pure white background.
    """
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"입력 이미지를 찾을 수 없습니다: {path}")
        pil_img = Image.open(path)
    elif isinstance(image, Image.Image):
        pil_img = image
    elif isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            pil_img = Image.fromarray(arr, mode="L").convert("RGB")
            return pil_img
        if arr.ndim == 3:
            if arr.shape[2] == 4:
                if bgr_input:
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
                pil_img = Image.fromarray(arr, mode="RGBA")
            elif arr.shape[2] == 3:
                if bgr_input:
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                return Image.fromarray(arr, mode="RGB")
            else:
                raise ValueError(f"지원하지 않는 채널 수입니다: {arr.shape}")
        else:
            raise ValueError(f"지원하지 않는 배열 차원입니다: {arr.ndim}")
    else:
        raise TypeError(f"지원하지 않는 입력 타입입니다: {type(image)}")

    if pil_img.mode in ("RGBA", "LA") or ("transparency" in pil_img.info):
        rgba = pil_img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return pil_img.convert("RGB")


def smooth_photo_preserve_edges(pil_rgb: Image.Image, strength: int = 1) -> Image.Image:
    """Smooth fine textures (hair strands, skin pores) while keeping major contours."""
    if strength <= 0:
        return pil_rgb
    bgr = cv2.cvtColor(np.array(pil_rgb), cv2.COLOR_RGB2BGR)
    for _ in range(strength):
        bgr = cv2.bilateralFilter(bgr, d=9, sigmaColor=75, sigmaSpace=75)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), mode="RGB")


def binarize_gray(
    gray: np.ndarray,
    remove_specks: bool = True,
    threshold: int | None = None,
    min_component_area: int = 2,
) -> np.ndarray:
    """Binarize a grayscale line art image (white background, black lines).

    If `threshold` (0-255) is given, pixels <= threshold become black; lower values
    keep only stronger/darker lines and drop faint details.
    """
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    if threshold is None:
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    if remove_specks and min_component_area > 0:
        inv = 255 - binary
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
        for label_id in range(1, num_labels):
            if stats[label_id, cv2.CC_STAT_AREA] <= min_component_area:
                binary[labels == label_id] = 255

    return binary


def pad_to_paper_ratio(image: Image.Image, paper: str | None) -> Image.Image:
    """Pad an image with white borders to match the target paper aspect ratio."""
    if not paper or paper == "none":
        return image
    if paper not in PAPER_RATIOS:
        raise ValueError(f"지원하지 않는 용지 규격입니다: {paper}. 가능 값: {list(PAPER_RATIOS)}")

    ratio_w, ratio_h = PAPER_RATIOS[paper]
    target_aspect = ratio_w / ratio_h
    w, h = image.size
    current_aspect = w / h

    if abs(current_aspect - target_aspect) < 1e-3:
        return image

    if current_aspect > target_aspect:
        new_h = int(round(w / target_aspect))
        canvas_w, canvas_h = w, max(h, new_h)
    else:
        new_w = int(round(h * target_aspect))
        canvas_w, canvas_h = max(w, new_w), h

    bg_color = 255 if image.mode == "L" else (255, 255, 255)
    canvas = Image.new(image.mode, (canvas_w, canvas_h), bg_color)
    canvas.paste(image, ((canvas_w - w) // 2, (canvas_h - h) // 2))
    return canvas
