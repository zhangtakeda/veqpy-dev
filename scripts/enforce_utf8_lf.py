#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
直接修复单个目录下指定代码目录中的文件编码和换行格式.

规则:
1. 所有代码文件必须使用 UTF-8.
2. 所有代码文件必须使用 LF.

用法:
    python enforce_utf8_lf.py
    python enforce_utf8_lf.py /path/to/project --dirs veqpy tests

可选:
    python enforce_utf8_lf.py /path/to/project --dirs veqpy tests --ext .py .md .toml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

DEFAULT_EXTENSIONS = {
    ".py",
    ".pyi",
    ".sh",
    ".bash",
    ".zsh",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".md",
    ".txt",
}

DEFAULT_DIRS = {
    "docs",
    "scripts",
    "tests",
    "veqpy",
}

DEFAULT_TOP_LEVEL_FILES = {
    ".gitignore",
    "LICENSE",
    "README.md",
    "TODO.md",
    "pyproject.toml",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite selected files to UTF-8 encoding and LF line endings.")
    parser.add_argument(
        "folder",
        type=Path,
        nargs="?",
        default=Path.cwd(),
        help="Project root folder",
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=None,
        help="Folder names to scan, e.g. --dirs veqpy tests doc",
    )
    parser.add_argument(
        "--ext",
        nargs="*",
        default=None,
        help="File extensions to include, e.g. --ext .py .md .toml",
    )
    return parser.parse_args(argv)


def iter_target_files(root: Path, target_dirs: set[str], extensions: set[str]) -> Iterable[Path]:
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        if path.name in DEFAULT_TOP_LEVEL_FILES or path.suffix.lower() in extensions:
            yield path

    for dir_name in sorted(target_dirs):
        target_dir = root / dir_name
        if not target_dir.exists() or not target_dir.is_dir():
            print(f"[WARN] skip missing directory: {dir_name}")
            continue

        for path in target_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                yield path


def fix_file(path: Path) -> tuple[bool, str]:
    raw = path.read_bytes()

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return False, "not valid UTF-8, skipped"

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    new_raw = normalized.encode("utf-8")

    if new_raw != raw:
        path.write_bytes(new_raw)
        return True, "rewritten to UTF-8 + LF"

    return True, "already OK"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    root = args.folder.resolve()
    if not root.exists() or not root.is_dir():
        print(f"ERROR: folder does not exist or is not a directory: {root}")
        return 2

    extensions = set(args.ext) if args.ext else set(DEFAULT_EXTENSIONS)
    target_dirs = set(args.dirs) if args.dirs else set(DEFAULT_DIRS)

    checked = 0
    skipped = 0
    changed = 0

    for path in iter_target_files(root, target_dirs, extensions):
        checked += 1
        rel = path.relative_to(root)

        ok, message = fix_file(path)
        if not ok:
            skipped += 1
            print(f"[SKIP] {rel} -> {message}")
            continue

        if message != "already OK":
            changed += 1
            print(rel)

    print(f"DONE: checked {checked} files, changed {changed} files, skipped {skipped} files.")
    return 0 if skipped == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
