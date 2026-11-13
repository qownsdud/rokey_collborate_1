"""Lightweight OpenCV-based photo-to-sketch conversion module."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image

from utils import ImageInput, binarize_gray, load_pil_rgb, pad_to_paper_ratio

CVSketchMode = Literal["dodge", "canny", "adaptive"]


@dataclass
class LineartCVConfig:
    """Configuration for OpenCV-based lightweight sketch conversion."""

    mode: CVSketchMode = "canny"
    blur_ksize: int = 21
    bilateral_d: int = 9
    bilateral_sigma: float = 75.0
    canny_low: int = 50
    canny_high: int = 150
    adaptive_block_size: int = 11
    adaptive_c: int = 4
    max_side: int | None = None
    binarize: bool = False
    paper: str | None = None


def _ensure_odd_ksize(ksize: int) -> int:
    k = max(3, int(ksize))
    return k if k % 2 == 1 else k + 1


def _resize_max_side(rgb: np.ndarray, max_side: int | None) -> np.ndarray:
    if max_side is None or max_side <= 0:
        return rgb
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return rgb
    scale = max_side / longest
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


class LineartConverterCV:
    """Fast CPU-only OpenCV sketch converter supporting dodge, canny, and adaptive modes."""

    def __init__(self, config: LineartCVConfig | None = None) -> None:
        self.config = config or LineartCVConfig()

    def convert(
        self,
        image: ImageInput,
        *,
        mode: CVSketchMode | None = None,
        max_side: int | None = None,
        binarize: bool | None = None,
        paper: str | None = None,
        bgr_input: bool = False,
    ) -> Image.Image:
        """Convert an input photo to a grayscale sketch PIL Image."""
        pil_rgb = load_pil_rgb(image, bgr_input=bgr_input)
        rgb = np.array(pil_rgb)
        rgb = _resize_max_side(
            rgb, self.config.max_side if max_side is None else max_side
        )
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        sketch_mode = self.config.mode if mode is None else mode
        if sketch_mode == "dodge":
            sketch = self._dodge_sketch(gray)
        elif sketch_mode == "canny":
            sketch = self._canny_sketch(gray)
        elif sketch_mode == "adaptive":
            sketch = self._adaptive_sketch(gray)
        else:
            raise ValueError(
                f"지원하지 않는 OpenCV 변환 모드입니다: {sketch_mode}. "
                "가능 값: 'dodge', 'canny', 'adaptive'"
            )

        do_binarize = self.config.binarize if binarize is None else binarize
        if do_binarize and sketch_mode != "canny":
            sketch = binarize_gray(sketch)

        target_paper = self.config.paper if paper is None else paper
        return pad_to_paper_ratio(Image.fromarray(sketch, mode="L"), target_paper)

    def _dodge_sketch(self, gray: np.ndarray) -> np.ndarray:
        """Pencil-sketch effect via grayscale inversion, Gaussian blur, and color dodge."""
        k = _ensure_odd_ksize(self.config.blur_ksize)
        inverted = 255 - gray
        blurred = cv2.GaussianBlur(inverted, (k, k), 0)
        return cv2.divide(gray, 255 - blurred, scale=256.0)

    def _canny_sketch(self, gray: np.ndarray) -> np.ndarray:
        """Clean contour lines via bilateral filtering and Canny edge detection."""
        smoothed = cv2.bilateralFilter(
            gray,
            d=self.config.bilateral_d,
            sigmaColor=self.config.bilateral_sigma,
            sigmaSpace=self.config.bilateral_sigma,
        )
        edges = cv2.Canny(smoothed, self.config.canny_low, self.config.canny_high)
        return 255 - edges

    def _adaptive_sketch(self, gray: np.ndarray) -> np.ndarray:
        """High-contrast line extraction via bilateral filtering and adaptive thresholding."""
        smoothed = cv2.bilateralFilter(
            gray,
            d=self.config.bilateral_d,
            sigmaColor=self.config.bilateral_sigma,
            sigmaSpace=self.config.bilateral_sigma,
        )
        block = _ensure_odd_ksize(self.config.adaptive_block_size)
        return cv2.adaptiveThreshold(
            smoothed,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block,
            self.config.adaptive_c,
        )

    def convert_to_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        **kwargs,
    ) -> Path:
        """Convert an image file and save the resulting sketch to disk."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result = self.convert(input_path, **kwargs)
        result.save(out_path)
        return out_path
