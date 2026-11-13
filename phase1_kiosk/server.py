"""FastAPI kiosk server for Phase 1 camera UI + Phase 2-B (phase2_lib_lineart) DL line art."""

from __future__ import annotations

from datetime import datetime
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
STATIC_DIR = BASE_DIR / "static"
CONFIRMED_DIR = BASE_DIR / "confirmed_outputs"
CONFIRMED_HISTORY_DIR = CONFIRMED_DIR / "history"
PHASE2_LIB_DIR = ROOT_DIR / "phase2_lib_lineart"
PHASE2_LIB_PYTHON = PHASE2_LIB_DIR / ".venv" / "bin" / "python"

for p in (str(BASE_DIR), str(PHASE2_LIB_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from preprocess import normalize_photo_bytes  # noqa: E402
from utils import PAPER_RATIOS  # noqa: E402

app = FastAPI(
    title="카메라 라인아트 협동로봇 키오스크",
    description="노트북 카메라로 촬영한 사진을 로컬 딥러닝(phase2_lib_lineart)으로 흑백 선화 변환합니다.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _compute_black_ratio(png_bytes: bytes) -> float:
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if gray is None or gray.size == 0:
        return 0.0
    return float(np.mean(gray < 128))


DL_DETAIL_PRESETS: dict[str, dict[str, object]] = {
    "high": {
        "label": "세밀하게 (디테일 많이)",
        "coarse": False,
        "pre_smooth": 0,
        "line_threshold": 220,
        "min_line_area": 2,
    },
    "medium": {
        "label": "표준 (균형)",
        "coarse": False,
        "pre_smooth": 1,
        "line_threshold": 210,
        "min_line_area": 3,
    },
    "low": {
        "label": "단순하게 (굵은 선 위주)",
        "coarse": False,
        "pre_smooth": 2,
        "line_threshold": 200,
        "min_line_area": 5,
    },
}


@app.get("/")
def serve_kiosk_ui() -> FileResponse:
    """Serve the camera-only kiosk frontend."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def api_health() -> dict[str, object]:
    """Check Phase 2-B (phase2_lib_lineart) local DL readiness."""
    dl_ready = PHASE2_LIB_PYTHON.is_file()
    return {
        "status": "ok" if dl_ready else "degraded",
        "dl_ready": dl_ready,
    }


@app.post("/api/kiosk/photo")
async def kiosk_photo_to_lineart(
    photo: UploadFile = File(..., description="웹캠 촬영 또는 업로드 사진"),
    detail: str = Form("medium", description="선화 디테일 수준 (high, medium, low)"),
    paper: str = Form("a4", description="용지 규격 (a4, postcard, square)"),
) -> Response:
    """Convert webcam photo into B&W line art via phase2_lib_lineart (Local DL)."""
    if paper not in PAPER_RATIOS:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 용지 규격입니다: {paper}")
    if detail not in DL_DETAIL_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 디테일 옵션입니다: {detail}. 가능 값: {list(DL_DETAIL_PRESETS)}",
        )
    if not PHASE2_LIB_PYTHON.is_file():
        raise HTTPException(
            status_code=503,
            detail="로컬 DL 가상환경(phase2_lib_lineart/.venv)을 찾을 수 없습니다.",
        )

    raw_photo = await photo.read()
    try:
        clean_photo_bytes = normalize_photo_bytes(raw_photo, max_side=1024)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    preset = DL_DETAIL_PRESETS[detail]
    t0 = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        in_path = tmp_path / "webcam_input.png"
        out_dir = tmp_path / "out"
        in_path.write_bytes(clean_photo_bytes)

        cmd = [
            str(PHASE2_LIB_PYTHON),
            str(PHASE2_LIB_DIR / "main.py"),
            "--input",
            str(in_path),
            "--method",
            "dl",
            "--resolution",
            "512",
            "--pre-smooth",
            str(preset["pre_smooth"]),
            "--line-threshold",
            str(preset["line_threshold"]),
            "--min-line-area",
            str(preset["min_line_area"]),
            "--binarize",
            "--paper",
            paper,
            "--output-dir",
            str(out_dir),
        ]
        if preset["coarse"]:
            cmd.append("--coarse")

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            err_msg = proc.stderr.strip() or proc.stdout.strip() or "DL 변환 실행 실패"
            raise HTTPException(status_code=500, detail=f"DL 선화 변환 실패: {err_msg[:300]}")

        produced = list(out_dir.glob("*.png"))
        if not produced:
            raise HTTPException(status_code=500, detail="DL 선화 결과 파일이 생성되지 않았습니다.")

        out_bytes = produced[0].read_bytes()
        with Image.open(io.BytesIO(out_bytes)) as im:
            w, h = im.size
        elapsed = time.perf_counter() - t0
        return Response(
            content=out_bytes,
            media_type="image/png",
            headers={
                "X-Elapsed-Sec": f"{elapsed:.2f}",
                "X-Black-Ratio": f"{_compute_black_ratio(out_bytes):.4f}",
                "X-Image-Size": f"{w}x{h}",
                "X-Engine": f"Local DL ({detail})",
            },
        )


@app.post("/api/kiosk/confirm")
async def confirm_lineart_for_robot(
    image: UploadFile = File(..., description="최종 확정할 흑백 라인아트 PNG"),
    paper: str = Form("a4", description="용지 규격 (a4, postcard, square)"),
    engine: str = Form("Local DL", description="생성 엔진 정보"),
) -> dict[str, object]:
    """Save the confirmed line art PNG and metadata for Phase 3 robot path planning."""
    raw_bytes = await image.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="확정할 이미지 데이터가 비어 있습니다.")

    try:
        with Image.open(io.BytesIO(raw_bytes)) as im:
            rgb_im = im.convert("RGB")
            w, h = rgb_im.size
            png_buf = io.BytesIO()
            rgb_im.save(png_buf, format="PNG")
            clean_png = png_buf.getvalue()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="유효한 이미지 파일이 아닙니다.") from exc

    CONFIRMED_DIR.mkdir(parents=True, exist_ok=True)
    CONFIRMED_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    latest_path = CONFIRMED_DIR / "latest_lineart.png"
    meta_path = CONFIRMED_DIR / "latest_meta.json"
    archived_prev: str | None = None

    if latest_path.is_file():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        archive_path = CONFIRMED_HISTORY_DIR / f"lineart_{timestamp}.png"
        latest_path.replace(archive_path)
        archived_prev = str(archive_path)

    if meta_path.is_file():
        meta_path.unlink()

    latest_path.write_bytes(clean_png)

    return {
        "status": "confirmed",
        "message": "로봇팔 드로잉용 이미지가 최종 선택되었습니다.",
        "confirmed_at": datetime.now().isoformat(timespec="seconds"),
        "paper": paper,
        "engine": engine,
        "width": w,
        "height": h,
        "black_ratio": round(_compute_black_ratio(clean_png), 4),
        "latest_file": str(latest_path),
        "archived_previous_file": archived_prev,
    }


@app.get("/api/kiosk/latest")
def get_latest_confirmed() -> dict[str, object]:
    """Return info of the latest confirmed line art PNG."""
    latest_path = CONFIRMED_DIR / "latest_lineart.png"
    if not latest_path.is_file():
        raise HTTPException(status_code=404, detail="아직 최종 선택된 라인아트 이미지가 없습니다.")
    with Image.open(latest_path) as im:
        w, h = im.size
    return {
        "latest_file": str(latest_path),
        "width": w,
        "height": h,
    }
