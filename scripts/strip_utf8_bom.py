#!/usr/bin/env python3
"""Strip UTF-8 BOM from text files."""

from __future__ import annotations

import argparse
from pathlib import Path


TEXT_SUFFIXES = {
    ".h",
    ".hpp",
    ".hh",
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".md",
    ".txt",
    ".py",
    ".ps1",
    ".sh",
    ".json",
    ".yml",
    ".yaml",
    ".cmake",
}


def iter_targets(repo: Path, explicit_paths: list[str]) -> list[Path]:
    if explicit_paths:
        out: list[Path] = []
        for p in explicit_paths:
            path = (repo / p).resolve() if not Path(p).is_absolute() else Path(p)
            if path.exists() and path.is_file():
                out.append(path)
        return out

    out: list[Path] = []
    for path in repo.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(repo).as_posix()
        if rel.startswith(".git/"):
            continue
        if path.name in (".gitignore", ".gitattributes", ".editorconfig") or path.suffix.lower() in TEXT_SUFFIXES:
            out.append(path)
    return sorted(out)


def strip_bom(path: Path) -> bool:
    data = path.read_bytes()
    if not data.startswith(b"\xEF\xBB\xBF"):
        return False
    path.write_bytes(data[3:])
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    changed = 0
    for path in iter_targets(repo, args.paths):
        if strip_bom(path):
            changed += 1
            print(f"stripped: {path}")

    print(f"changed_files={changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
