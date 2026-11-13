"""Deep-learning line art extraction module using controlnet_aux.LineartDetector."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from controlnet_aux import LineartDetector

from utils import (
    ImageInput,
    binarize_gray,
    load_pil_rgb,
    pad_to_paper_ratio,
    smooth_photo_preserve_edges,
)


@dataclass
class LineartDLConfig:
    """Configuration for LineartDetector-based conversion."""

    pretrained_model_or_path: str = "lllyasviel/Annotators"
    coarse: bool = False
    detect_resolution: int = 512
    image_resolution: int = 512
    pre_smooth: int = 2
    invert_to_white_bg: bool = True
    binarize: bool = True
    line_threshold: int | None = 200
    min_line_area: int = 5
    paper: str | None = None
    device: str | None = None


class LineartConverterDL:
    """Stateful wrapper around controlnet_aux.LineartDetector for photo-to-lineart."""

    def __init__(self, config: LineartDLConfig | None = None) -> None:
        self.config = config or LineartDLConfig()
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = LineartDetector.from_pretrained(
            self.config.pretrained_model_or_path
        )
        self.detector.to(self.device)

    def convert(
        self,
        image: ImageInput,
        *,
        coarse: bool | None = None,
        detect_resolution: int | None = None,
        image_resolution: int | None = None,
        pre_smooth: int | None = None,
        invert_to_white_bg: bool | None = None,
        binarize: bool | None = None,
        line_threshold: int | None = None,
        min_line_area: int | None = None,
        paper: str | None = None,
        bgr_input: bool = False,
    ) -> Image.Image:
        """Convert an input photo to a line art PIL Image."""
        pil_rgb = load_pil_rgb(image, bgr_input=bgr_input)

        use_coarse = self.config.coarse if coarse is None else coarse
        det_res = (
            self.config.detect_resolution
            if detect_resolution is None
            else detect_resolution
        )
        img_res = (
            self.config.image_resolution
            if image_resolution is None
            else image_resolution
        )
        smooth_strength = (
            self.config.pre_smooth if pre_smooth is None else pre_smooth
        )
        do_invert = (
            self.config.invert_to_white_bg
            if invert_to_white_bg is None
            else invert_to_white_bg
        )
        do_binarize = self.config.binarize if binarize is None else binarize
        thresh = (
            self.config.line_threshold
            if line_threshold is None
            else line_threshold
        )
        min_area = (
            self.config.min_line_area if min_line_area is None else min_line_area
        )
        target_paper = self.config.paper if paper is None else paper

        if smooth_strength > 0:
            pil_rgb = smooth_photo_preserve_edges(pil_rgb, strength=smooth_strength)

        result = self.detector(
            pil_rgb,
            coarse=use_coarse,
            detect_resolution=det_res,
            image_resolution=img_res,
        )

        # LineartDetector outputs white lines on black background; invert for white-paper sketch.
        gray = result.convert("L")
        if do_invert:
            gray = ImageOps.invert(gray)

        if do_binarize or thresh is not None or min_area > 2:
            gray_arr = binarize_gray(
                np.array(gray),
                threshold=thresh,
                min_component_area=min_area,
            )
            gray = Image.fromarray(gray_arr, mode="L")

        return pad_to_paper_ratio(gray, target_paper)

    def convert_to_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        **kwargs,
    ) -> Path:
        """Convert an image file and save the resulting line art to disk."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result = self.convert(input_path, **kwargs)
        result.save(out_path)
        return out_path
