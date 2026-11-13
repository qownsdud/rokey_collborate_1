"""Phase 1 input preprocessing: keyword validation/filtering and image checks."""

from __future__ import annotations

import io
import re
from PIL import Image
import numpy as np

# 기본 금칙어 목록 (필요에 따라 확장 가능)
FORBIDDEN_WORDS: set[str] = {
    "nude",
    "nsfw",
    "porn",
    "violence",
    "blood",
    "gore",
    "kill",
    "weapon",
    "gun",
    "knife",
    "나체",
    "음란",
    "폭력",
    "유혈",
    "살인",
    "무기",
    "총",
    "칼",
    "욕설",
}

# 자주 사용하는 한글 키워드 -> 영문 기본 매핑 (Cloudflare FLUX 프롬프트 인식률 향상)
KO_TO_EN_DICT: dict[str, str] = {
    "고양이": "cat",
    "강아지": "dog",
    "토끼": "rabbit",
    "곰": "bear",
    "여우": "fox",
    "새": "bird",
    "물고기": "fish",
    "공룡": "dinosaur",
    "로봇": "robot",
    "우주복": "spacesuit",
    "우주": "space",
    "우주선": "spaceship",
    "행성": "planet",
    "별": "star",
    "달": "moon",
    "태양": "sun",
    "꽃": "flower",
    "나무": "tree",
    "집": "house",
    "성": "castle",
    "자동차": "car",
    "비행기": "airplane",
    "웃는 모습": "smiling",
    "웃는": "smiling",
    "미소": "smiling",
    "행복한": "happy",
    "귀여운": "cute",
    "멋진": "cool",
    "모자": "hat",
    "왕관": "crown",
    "안경": "glasses",
    "날개": "wings",
    "하트": "heart",
    "사람": "person",
    "소녀": "girl",
    "소년": "boy",
}


def validate_and_clean_keywords(raw_keywords: str | list[str]) -> list[str]:
    """Validate forbidden words, clean whitespace/symbols, and normalize up to 3 keywords."""
    if isinstance(raw_keywords, str):
        items = [k.strip() for k in raw_keywords.split(",") if k.strip()]
    else:
        items = [str(k).strip() for k in raw_keywords if str(k).strip()]

    if not items:
        raise ValueError("최소 1개 이상의 키워드를 입력해 주세요.")

    cleaned: list[str] = []
    for item in items[:3]:
        # 특수문자 제거 (한글, 영문, 숫자, 공백, 하이픈 허용)
        sanitized = re.sub(r"[^\w\s가-힣-]", "", item).strip()
        if not sanitized:
            continue

        lowered = sanitized.lower()
        for bad in FORBIDDEN_WORDS:
            if bad in lowered:
                raise ValueError(f"부적절한 키워드가 포함되어 있습니다: '{sanitized}'")

        mapped = KO_TO_EN_DICT.get(sanitized, sanitized)
        cleaned.append(mapped)

    if not cleaned:
        raise ValueError("유효한 키워드가 없습니다. 다시 입력해 주세요.")

    return cleaned


def validate_non_empty_sketch(image_bytes: bytes, min_dark_ratio: float = 0.0005) -> None:
    """Ensure uploaded sketch is a valid image and not completely blank."""
    if not image_bytes:
        raise ValueError("업로드된 이미지가 비어 있습니다.")

    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
            rgba = img.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            gray = Image.alpha_composite(bg, rgba).convert("L")
        else:
            gray = img.convert("L")
    except Exception as exc:
        raise ValueError("유효한 이미지 파일이 아닙니다.") from exc

    arr = np.asarray(gray, dtype=np.uint8)
    dark_ratio = float(np.mean(arr < 240))
    if dark_ratio < min_dark_ratio:
        raise ValueError("캔버스가 비어 있습니다. 스케치를 먼저 그려주세요.")


def normalize_photo_bytes(image_bytes: bytes, max_side: int = 1024) -> bytes:
    """Decode webcam/uploaded photo, orient/resize if oversized, and return RGB PNG bytes."""
    if not image_bytes:
        raise ValueError("사진 데이터가 비어 있습니다.")

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError("유효한 사진 이미지가 아닙니다.") from exc

    w, h = img.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / longest
        img = img.resize(
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            Image.Resampling.LANCZOS,
        )

    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()
