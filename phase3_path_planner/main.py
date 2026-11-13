"""CLI entrypoint for Phase 3 vector path extraction and visual inspection."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from extractor import extract_vector_paths  # noqa: E402
from visualize import render_comparison_sheet, render_pure_vector_image  # noqa: E402


def _archive_previous_outputs(out_dir: Path, keep_names: set[str]) -> int:
    """Move previous output files (excluding current targets) into out_dir/history/."""
    if not out_dir.is_dir():
        return 0
    history_dir = out_dir / "history"
    moved = 0
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for item in sorted(out_dir.iterdir()):
        if not item.is_file() or item.name in keep_names:
            continue
        history_dir.mkdir(parents=True, exist_ok=True)
        dest = history_dir / item.name
        if dest.exists():
            dest = history_dir / f"{item.stem}_{stamp}{item.suffix}"
        shutil.move(str(item), str(dest))
        moved += 1
    return moved


def main() -> None:
    root_dir = BASE_DIR.parent
    default_input = root_dir / "phase1_kiosk" / "confirmed_outputs" / "latest_lineart.png"
    default_out_dir = BASE_DIR / "outputs"

    parser = argparse.ArgumentParser(description="Phase 3: Extract vector strokes and render visual outputs.")
    parser.add_argument("--input", type=Path, default=default_input, help="입력 흑백 라인아트 이미지 경로")
    parser.add_argument("--out-dir", type=Path, default=default_out_dir, help="결과 저장 디렉토리")
    parser.add_argument("--epsilon", type=float, default=0.45, help="approxPolyDP 허용 오차 (px)")
    parser.add_argument("--min-length", type=float, default=2.5, help="최소 선 길이 (px)")
    parser.add_argument("--thickness", type=int, default=1, help="순수 벡터 이미지 선 두께 (px)")
    parser.add_argument("--bridge-gap", type=float, default=4.5, help="끊어진 끝점 자동 연결 최대 거리 (px)")
    parser.add_argument("--smooth", type=int, default=2, help="Chaikin 곡선 스무딩 반복 횟수")
    parser.add_argument("--no-thick-contours", action="store_true", help="굵은 영역 외곽선 보존 비활성화")
    parser.add_argument("--morph-clean", action="store_true", help="미세 잡티 제거용 모폴로지 연산 적용")
    args = parser.parse_args()

    result = extract_vector_paths(
        image_path=str(args.input),
        epsilon=args.epsilon,
        min_length_px=args.min_length,
        morph_clean=args.morph_clean,
        preserve_thick_contours=not args.no_thick_contours,
        bridge_gap_px=args.bridge_gap,
        smooth_iterations=args.smooth,
    )

    stem = args.input.stem
    args.out_dir.mkdir(parents=True, exist_ok=True)
    archived_count = _archive_previous_outputs(args.out_dir, keep_names=set())

    vector_img_path = render_pure_vector_image(result, args.out_dir / f"{stem}_vector.png", line_thickness=args.thickness)
    compare_img_path = render_comparison_sheet(str(args.input), result, args.out_dir / f"{stem}_comparison.png")

    json_path = args.out_dir / f"{stem}_strokes_px.json"
    payload = [
        {"stroke_id": i + 1, "points": [[round(x, 2), round(y, 2)] for x, y in stroke]}
        for i, stroke in enumerate(result.strokes)
    ]
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    reduction = (
        (1.0 - result.simplified_point_count / result.raw_point_count) * 100.0
        if result.raw_point_count > 0
        else 0.0
    )
    print(f"[Phase 3 완료] 입력: {args.input}")
    if archived_count:
        print(f" - 이전 출력 {archived_count}개 -> {args.out_dir / 'history'} 이동 완료")
    print(f" - 처리 시간: {result.elapsed_sec:.3f}s")
    print(f" - Stroke 수: {len(result.strokes)}")
    print(f" - 점 개수: {result.raw_point_count:,} -> {result.simplified_point_count:,} ({reduction:.1f}% 감소)")
    print(f" - 총 그리기 길이: {result.draw_length_px:.1f} px | Pen-Up 이동 길이: {result.travel_length_px:.1f} px")
    print(f" - 순수 벡터 이미지: {vector_img_path}")
    print(f" - 비교 시트 이미지: {compare_img_path}")
    print(f" - 픽셀 Stroke JSON: {json_path}")


if __name__ == "__main__":
    main()
