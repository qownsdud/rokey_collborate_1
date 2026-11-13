"""Visualization utilities to inspect extracted vector paths vs original line art."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from extractor import ExtractionResult


def render_pure_vector_image(result: ExtractionResult, out_path: Path, line_thickness: int = 1) -> Path:
    """Render extracted vector strokes smoothly onto a white canvas (2x supersampled AA)."""
    scale = 2
    canvas = np.full((result.height * scale, result.width * scale, 3), 255, dtype=np.uint8)
    scaled_thickness = max(1, line_thickness)
    for stroke in result.strokes:
        pts = np.array([[round(x * scale), round(y * scale)] for x, y in stroke], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=False, color=(0, 0, 0), thickness=scaled_thickness, lineType=cv2.LINE_AA)
    out = cv2.resize(canvas, (result.width, result.height), interpolation=cv2.INTER_AREA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)
    return out_path


def render_comparison_sheet(
    original_image_path: str,
    result: ExtractionResult,
    out_path: Path,
) -> Path:
    """Save a 3-panel visual comparison: Original | 1px Skeleton | Vector Strokes + Pen-Up Path."""
    orig = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    if orig is None:
        raise FileNotFoundError(f"원본 이미지를 읽을 수 없습니다: {original_image_path}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 7), dpi=150)

    # Panel 1: Original B&W line art
    axes[0].imshow(orig, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"1. Input Line Art ({result.width}x{result.height})", fontsize=11)
    axes[0].axis("off")

    # Panel 2: 1px thinned skeleton (inverted so lines are black on white)
    axes[1].imshow(255 - result.skeleton, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"2. Thinned 1px Skeleton\n(raw pts: {result.raw_point_count:,})", fontsize=11)
    axes[1].axis("off")

    # Panel 3: Vector strokes colored by TSP order + red dashed pen-up moves
    ax = axes[2]
    ax.set_xlim(0, result.width)
    ax.set_ylim(result.height, 0)
    ax.set_aspect("equal")
    ax.set_facecolor("white")

    cmap = plt.get_cmap("viridis")
    n = max(len(result.strokes), 1)
    for idx, stroke in enumerate(result.strokes):
        xs = [p[0] for p in stroke]
        ys = [p[1] for p in stroke]
        color = cmap(idx / max(n - 1, 1))
        ax.plot(xs, ys, color=color, linewidth=1.2)

        if idx > 0:
            prev_end = result.strokes[idx - 1][-1]
            ax.plot(
                [prev_end[0], xs[0]],
                [prev_end[1], ys[0]],
                color="red",
                linestyle="--",
                linewidth=0.5,
                alpha=0.45,
            )

    reduction_pct = (
        (1.0 - result.simplified_point_count / result.raw_point_count) * 100.0
        if result.raw_point_count > 0
        else 0.0
    )
    ax.set_title(
        f"3. Vector Strokes ({len(result.strokes)} strokes, {result.simplified_point_count:,} pts, "
        f"-{reduction_pct:.1f}%)\nRed dashed = Pen-Up travel | Time = {result.elapsed_sec:.3f}s",
        fontsize=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    return out_path
