#!/usr/bin/env python3
"""Tests for file_manager.py."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

import file_manager


class FileManagerTests(unittest.TestCase):
    def test_sync_and_ignores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".venv" / "bin").mkdir(parents=True)
            (root / ".venv" / "bin" / "python").write_text("", encoding="utf-8")
            (root / "docs").mkdir()
            (root / "docs" / "가이드.txt").write_text("hello", encoding="utf-8")
            (root / "empty_dir").mkdir()
            (root / "app.py").write_text("print(1)", encoding="utf-8")

            manifest = root / "files.yaml"
            content = file_manager.write_manifest(root, manifest)
            parsed = yaml.safe_load(content)

            tree = parsed["tree"]
            self.assertNotIn(".venv", tree)
            self.assertIn("docs", tree)
            self.assertIn("가이드.txt", tree["docs"])
            self.assertEqual(tree["empty_dir"], {})
            self.assertIsNone(tree["app.py"])

    def test_apply_file_rename_move_and_dir_rename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "old.py").write_text("code", encoding="utf-8")
            (root / "notes.txt").write_text("note", encoding="utf-8")

            manifest = root / "files.yaml"
            file_manager.write_manifest(root, manifest)

            data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            data["tree"]["notes.txt"] = "docs/readme.txt"
            data["tree"]["src"]["old.py"] = "new.py"
            data["tree"]["src"]["_rename_dir"] = "core"
            manifest.write_text(
                yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )

            applied, warnings = file_manager.apply_manifest(root, manifest)
            self.assertEqual(warnings, [])
            self.assertEqual(len(applied), 3)
            self.assertTrue((root / "docs" / "readme.txt").exists())
            self.assertTrue((root / "core" / "new.py").exists())
            self.assertFalse((root / "src").exists())
            self.assertFalse((root / "notes.txt").exists())

    def test_collision_and_traversal_protection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("a", encoding="utf-8")
            (root / "b.txt").write_text("b", encoding="utf-8")

            manifest = root / "files.yaml"
            file_manager.write_manifest(root, manifest)

            data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            data["tree"]["a.txt"] = "b.txt"  # collision
            data["tree"]["b.txt"] = "../outside.txt"  # traversal
            manifest.write_text(
                yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )

            applied, warnings = file_manager.apply_manifest(root, manifest)
            self.assertEqual(applied, [])
            self.assertEqual(len(warnings), 1)
            self.assertIn("대상 이미 존재", warnings[0])
            self.assertTrue((root / "a.txt").exists())
            self.assertTrue((root / "b.txt").exists())

    def test_format_ascii_tree(self) -> None:
        tree = {"dir_a": {"nested.py": None}, "root.txt": None}
        out = file_manager.format_ascii_tree(tree, root_label="proj")
        self.assertIn("proj/", out)
        self.assertIn("├── dir_a/", out)
        self.assertIn("│   └── nested.py", out)
        self.assertIn("└── root.txt", out)


if __name__ == "__main__":
    unittest.main()
