"""CLI test & benchmark runner comparing DL (LineartDetector) and OpenCV sketch methods."""

import argparse
from pathlib import Path
import time

from lineart_cv import LineartCVConfig, LineartConverterCV


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="사진을 라인드로잉(선화/스케치)으로 변환하고 속도와 품질을 비교합니다."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="입력 이미지 파일 경로 (예: samples/person.jpg)",
    )
    parser.add_argument(
        "--method",
        choices=["dl", "cv", "both"],
        default="cv",
        help="변환 방식 선택 (dl: controlnet_aux, cv: OpenCV, both: 둘 다 실행)",
    )
    parser.add_argument(
        "--cv-mode",
        choices=["dodge", "canny", "adaptive"],
        default="canny",
        help="OpenCV 변환 세부 모드 (기본: canny)",
    )
    parser.add_argument(
        "--coarse",
        action="store_true",
        help="LineartDetector coarse 모드 활성화 (더 굵고 단순한 선화)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="LineartDetector 처리 해상도 (기본: 512)",
    )
    parser.add_argument(
        "--pre-smooth",
        type=int,
        default=2,
        help="DL 입력 전 경계 보존 블러(bilateralFilter) 반복 횟수 (0=미사용, 기본: 2)",
    )
    parser.add_argument(
        "--line-threshold",
        type=int,
        default=200,
        help="선 강도 임계값 (0~255, 낮을수록 진한 주선만 남고 옅은 잔선 제거, 기본: 200)",
    )
    parser.add_argument(
        "--min-line-area",
        type=int,
        default=5,
        help="유지할 최소 선 조각 픽셀 면적 (높일수록 짧은 잔선/점 제거, 기본: 5)",
    )
    parser.add_argument(
        "--binarize",
        action="store_true",
        help="출력 이미지를 순수 흑백(이진화)으로 후처리",
    )
    parser.add_argument(
        "--paper",
        choices=["none", "a4", "postcard", "square"],
        default="none",
        help="목표 용지 비율 패딩 적용 (기본: none)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="결과 이미지 저장 디렉토리 (기본: outputs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    paper = None if args.paper == "none" else args.paper

    if args.method in ("cv", "both"):
        cv_cfg = LineartCVConfig(
            mode=args.cv_mode,
            max_side=args.resolution,
            binarize=args.binarize,
            paper=paper,
        )
        converter_cv = LineartConverterCV(cv_cfg)
        t0 = time.perf_counter()
        cv_out = output_dir / f"{stem}_cv_{args.cv_mode}.png"
        converter_cv.convert_to_file(input_path, cv_out)
        elapsed_cv = time.perf_counter() - t0
        print(f"[OpenCV:{args.cv_mode}] 완료 ({elapsed_cv * 1000:.1f} ms) -> {cv_out}")

    if args.method in ("dl", "both"):
        from lineart_dl import LineartConverterDL, LineartDLConfig

        t_load_start = time.perf_counter()
        dl_cfg = LineartDLConfig(
            coarse=args.coarse,
            detect_resolution=args.resolution,
            image_resolution=args.resolution,
            pre_smooth=args.pre_smooth,
            binarize=args.binarize,
            line_threshold=args.line_threshold,
            min_line_area=args.min_line_area,
            paper=paper,
        )
        converter_dl = LineartConverterDL(dl_cfg)
        t_load = time.perf_counter() - t_load_start

        t_inf_start = time.perf_counter()
        suffix = "_dl"
        if args.coarse:
            suffix += "_coarse"
        if args.pre_smooth > 0:
            suffix += f"_smooth{args.pre_smooth}"
        if args.line_threshold is not None or args.min_line_area > 2:
            suffix += "_clean"
        dl_out = output_dir / f"{stem}{suffix}.png"
        converter_dl.convert_to_file(input_path, dl_out)
        t_inf = time.perf_counter() - t_inf_start
        print(
            f"[DL:LineartDetector ({converter_dl.device})] "
            f"모델 로드: {t_load:.2f} s | 추론: {t_inf:.2f} s -> {dl_out}"
        )


if __name__ == "__main__":
    main()
