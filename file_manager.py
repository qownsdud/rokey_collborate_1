#!/usr/bin/env python3
"""Workspace file manifest (files.yaml) & real-time tree manager.

Allows viewing and renaming/moving workspace files from a single YAML file
(`files.yaml`), and watching both the filesystem and `files.yaml` in real time.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import yaml

DEFAULT_IGNORES: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
        ".DS_Store",
    }
)

RENAME_DIR_KEY = "_rename_dir"
MANIFEST_FILENAME = "files.yaml"

YAML_HEADER = """\
# ============================================================================
# 프로젝트 파일 트리 & 일괄 이름 관리 (files.yaml)
# ============================================================================
# [사용 방법]
# 1. 파일 이름 변경: `기존파일명:` 뒤에 `새파일명`을 입력하고 저장하세요.
#    예) app.py: server.py
# 2. 파일 이동: 상대 경로를 입력하세요. (예: app.py: src/server.py)
# 3. 폴더 이름 변경: 해당 폴더 블록 안에 `_rename_dir: 새폴더명`을 추가하세요.
#
# [실행 명령어]
# - 실시간 감시 & 트리 출력: python3 file_manager.py watch
# - 변경 사항 수동 적용:     python3 file_manager.py apply
# - 현재 트리만 보기:        python3 file_manager.py tree
# - YAML 최신화:             python3 file_manager.py sync
# ============================================================================

"""


def _is_ignored(name: str, ignores: frozenset[str]) -> bool:
    return name in ignores


def scan_tree(root: Path, ignores: frozenset[str] = DEFAULT_IGNORES) -> dict[str, Any]:
    """Scan `root` directory and return a nested dict representing the file tree."""
    dirs: list[Path] = []
    files: list[Path] = []

    try:
        with os.scandir(root) as it:
            for entry in it:
                if _is_ignored(entry.name, ignores):
                    continue
                if entry.is_dir(follow_symlinks=False):
                    dirs.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    files.append(Path(entry.path))
    except OSError:
        return {}

    dirs.sort(key=lambda p: p.name.lower())
    files.sort(key=lambda p: p.name.lower())

    tree: dict[str, Any] = {}
    for d in dirs:
        tree[d.name] = scan_tree(d, ignores)
    for f in files:
        tree[f.name] = None
    return tree


def dump_yaml_tree(tree: dict[str, Any]) -> str:
    """Serialize the nested tree dict into clean, human-editable YAML."""
    payload = {"tree": tree}
    body = yaml.safe_dump(
        payload,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        indent=2,
    )
    # Remove explicit `null` for empty file values so lines look like `filename:`
    lines = []
    for line in body.splitlines():
        if line.endswith(": null"):
            lines.append(line[: -len(" null")])
        else:
            lines.append(line)
    return YAML_HEADER + "\n".join(lines) + "\n"


def write_manifest(root: Path, manifest_path: Path, ignores: frozenset[str] = DEFAULT_IGNORES) -> str:
    """Scan `root` and write `manifest_path`. Returns the written YAML text."""
    tree = scan_tree(root, ignores)
    content = dump_yaml_tree(tree)
    manifest_path.write_text(content, encoding="utf-8")
    return content


def _resolve_safe_target(root: Path, current_dir: Path, target_spec: str) -> Path | None:
    """Resolve `target_spec` relative to `current_dir`, ensuring it stays inside `root`."""
    root_resolved = root.resolve()
    candidate = (current_dir / target_spec).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        return None
    return candidate


def _collect_operations(
    node: dict[str, Any],
    root: Path,
    rel_dir: Path,
    file_ops: list[tuple[Path, Path]],
    dir_ops: list[tuple[Path, Path]],
) -> None:
    """Recursively collect file and directory rename/move operations from parsed YAML."""
    current_dir = root / rel_dir

    for key, val in node.items():
        if key == RENAME_DIR_KEY:
            if isinstance(val, str) and val.strip() and rel_dir != Path("."):
                new_name = val.strip()
                target = _resolve_safe_target(root, current_dir.parent, new_name)
                if target is not None and target != current_dir.resolve():
                    dir_ops.append((current_dir, target))
            continue

        if isinstance(val, dict):
            _collect_operations(val, root, rel_dir / key, file_ops, dir_ops)
        elif isinstance(val, str) and val.strip():
            new_spec = val.strip()
            src = current_dir / key
            target = _resolve_safe_target(root, current_dir, new_spec)
            if target is not None and target != src.resolve():
                file_ops.append((src, target))


def apply_manifest(
    root: Path,
    manifest_path: Path,
    dry_run: bool = False,
) -> tuple[list[str], list[str]]:
    """Apply rename/move operations specified in `manifest_path`.

    Returns `(applied_messages, warning_messages)`.
    """
    applied: list[str] = []
    warnings: list[str] = []

    if not manifest_path.exists():
        warnings.append(f"매니페스트 파일이 없습니다: {manifest_path}")
        return applied, warnings

    raw = manifest_path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict) or not isinstance(data.get("tree"), dict):
        warnings.append("YAML 형식이 올바르지 않습니다. 최상위에 `tree:` 매핑이 필요합니다.")
        return applied, warnings

    file_ops: list[tuple[Path, Path]] = []
    dir_ops: list[tuple[Path, Path]] = []
    _collect_operations(data["tree"], root, Path("."), file_ops, dir_ops)

    # Execute file operations first, then deepest directory operations first.
    dir_ops.sort(key=lambda pair: len(pair[0].parts), reverse=True)

    for src, dst in file_ops + dir_ops:
        rel_src = src.relative_to(root)
        try:
            rel_dst = dst.relative_to(root.resolve())
        except ValueError:
            rel_dst = dst

        if not src.exists():
            warnings.append(f"건너뜀 (원본 없음): {rel_src}")
            continue
        if dst.exists():
            warnings.append(f"건너뜀 (대상 이미 존재): {rel_src} -> {rel_dst}")
            continue

        if dry_run:
            applied.append(f"[DRY-RUN] {rel_src} -> {rel_dst}")
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        applied.append(f"변경 완료: {rel_src} -> {rel_dst}")

    return applied, warnings


def format_ascii_tree(
    tree: dict[str, Any],
    prefix: str = "",
    root_label: str | None = None,
) -> str:
    """Format a nested tree dict as a readable ASCII tree."""
    lines: list[str] = []
    if root_label is not None:
        lines.append(f"{root_label}/")

    items = list(tree.items())
    total = len(items)
    for idx, (name, val) in enumerate(items):
        is_last = idx == total - 1
        connector = "└── " if is_last else "├── "
        child_prefix = prefix + ("    " if is_last else "│   ")
        if isinstance(val, dict):
            lines.append(f"{prefix}{connector}{name}/")
            if val:
                lines.append(format_ascii_tree(val, prefix=child_prefix))
        else:
            lines.append(f"{prefix}{connector}{name}")

    return "\n".join(line for line in lines if line)


def _snapshot_tree(tree: dict[str, Any]) -> str:
    return hashlib.sha256(
        yaml.safe_dump(tree, allow_unicode=True, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _render_watch_screen(
    root: Path,
    tree: dict[str, Any],
    manifest_path: Path,
    logs: list[str],
    clear_screen: bool = True,
) -> None:
    if clear_screen and sys.stdout.isatty():
        print("\033[2J\033[H", end="")

    now = time.strftime("%H:%M:%S")
    print(f"=== 실시간 파일 트리 감시 중 [{now}] ===")
    print(f"루트: {root}")
    print(f"관리 파일: {manifest_path.name} (저장 시 자동 반영, Ctrl+C로 종료)\n")
    print(format_ascii_tree(tree, root_label=root.name))
    if logs:
        print("\n--- 최근 활동 ---")
        for msg in logs[-8:]:
            print(f"  {msg}")
    sys.stdout.flush()


def watch_loop(
    root: Path,
    manifest_path: Path,
    interval: float = 0.5,
    ignores: frozenset[str] = DEFAULT_IGNORES,
) -> None:
    """Watch both the filesystem and `manifest_path` in real time."""
    if not manifest_path.exists():
        write_manifest(root, manifest_path, ignores)

    tree = scan_tree(root, ignores)
    fs_sig = _snapshot_tree(tree)
    yaml_text = manifest_path.read_text(encoding="utf-8")
    yaml_sig = _hash_text(yaml_text)
    logs: list[str] = ["감시 시작됨."]

    _render_watch_screen(root, tree, manifest_path, logs)

    try:
        while True:
            time.sleep(interval)

            # 1. Check if files.yaml was edited by the user
            if manifest_path.exists():
                current_yaml_text = manifest_path.read_text(encoding="utf-8")
                current_yaml_sig = _hash_text(current_yaml_text)
                if current_yaml_sig != yaml_sig:
                    try:
                        applied, warnings = apply_manifest(root, manifest_path, dry_run=False)
                        for msg in applied + warnings:
                            logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
                        new_yaml_text = write_manifest(root, manifest_path, ignores)
                        yaml_sig = _hash_text(new_yaml_text)
                        tree = scan_tree(root, ignores)
                        fs_sig = _snapshot_tree(tree)
                        _render_watch_screen(root, tree, manifest_path, logs)
                        continue
                    except yaml.YAMLError as exc:
                        logs.append(f"[{time.strftime('%H:%M:%S')}] YAML 문법 오류 (수정 대기 중): {exc}")
                        yaml_sig = current_yaml_sig
                        _render_watch_screen(root, tree, manifest_path, logs)
                        continue

            # 2. Check if filesystem changed externally
            current_tree = scan_tree(root, ignores)
            current_fs_sig = _snapshot_tree(current_tree)
            if current_fs_sig != fs_sig:
                new_yaml_text = write_manifest(root, manifest_path, ignores)
                yaml_sig = _hash_text(new_yaml_text)
                tree = current_tree
                fs_sig = current_fs_sig
                logs.append(f"[{time.strftime('%H:%M:%S')}] 파일 구조 변경 감지 -> {manifest_path.name} 갱신")
                _render_watch_screen(root, tree, manifest_path, logs)
    except KeyboardInterrupt:
        print("\n감시를 종료합니다.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="워크스페이스 파일명 일괄 관리(files.yaml) 및 실시간 트리 뷰어"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="관리할 워크스페이스 루트 경로 (기본값: 스크립트 위치)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=MANIFEST_FILENAME,
        help=f"매니페스트 YAML 파일명 (기본값: {MANIFEST_FILENAME})",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("sync", help="현재 폴더 구조를 스캔해 files.yaml을 생성/갱신합니다.")

    apply_parser = subparsers.add_parser("apply", help="files.yaml의 이름 변경/이동 내역을 실제 파일에 적용합니다.")
    apply_parser.add_argument("--dry-run", action="store_true", help="실제 변경 없이 예정 작업만 출력합니다.")

    subparsers.add_parser("tree", help="현재 파일 트리 구조를 터미널에 출력합니다.")

    watch_parser = subparsers.add_parser("watch", help="파일 시스템과 files.yaml을 실시간으로 양방향 감시합니다.")
    watch_parser.add_argument("--interval", type=float, default=0.5, help="감시 주기(초, 기본값: 0.5)")

    args = parser.parse_args(argv)
    root = args.root.resolve()
    manifest_path = root / args.manifest

    if args.command == "sync":
        write_manifest(root, manifest_path)
        print(f"갱신 완료: {manifest_path}")
        return 0

    if args.command == "apply":
        applied, warnings = apply_manifest(root, manifest_path, dry_run=args.dry_run)
        for msg in applied:
            print(msg)
        for msg in warnings:
            print(f"경고: {msg}", file=sys.stderr)
        if not args.dry_run:
            write_manifest(root, manifest_path)
        if not applied and not warnings:
            print("변경할 항목이 없습니다.")
        return 0

    if args.command == "tree":
        tree = scan_tree(root)
        print(format_ascii_tree(tree, root_label=root.name))
        return 0

    if args.command == "watch":
        watch_loop(root, manifest_path, interval=args.interval)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
