#!/usr/bin/env python3
"""Check repository hygiene gates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REQUIRED_GITIGNORE_PATTERNS = [
    "trace/",
    "dumps/",
    "build/",
    "out/",
    "dist/",
    "logs/",
    "reports/",
    "archives/",
    "*.zip",
    "*.tar",
    "*.tar.gz",
    "*.tgz",
    "*.gz",
]

ARCHIVE_SUFFIXES = (".zip", ".tar", ".tgz", ".gz")
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


def has_utf8_bom(path: Path) -> bool:
    with path.open("rb") as f:
        head = f.read(3)
    return head == b"\xEF\xBB\xBF"


def should_check_bom(path: Path) -> bool:
    if path.name in (".gitignore", ".gitattributes", ".editorconfig"):
        return True
    return path.suffix.lower() in TEXT_SUFFIXES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    findings: list[str] = []

    gitignore = repo / ".gitignore"
    if not gitignore.exists():
        findings.append("missing .gitignore")
    else:
        text = gitignore.read_text(encoding="utf-8", errors="ignore")
        for pattern in REQUIRED_GITIGNORE_PATTERNS:
            if pattern not in text:
                findings.append(f".gitignore missing `{pattern}`")

    for path in sorted(repo.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(repo).as_posix()
        if rel.startswith(".git/"):
            continue

        if rel.endswith(".tar.gz"):
            findings.append(f"archive in repo: {rel}")
        elif path.suffix.lower() in ARCHIVE_SUFFIXES:
            findings.append(f"archive in repo: {rel}")

        if should_check_bom(path) and has_utf8_bom(path):
            findings.append(f"utf8-bom found: {rel}")

    if findings:
        print("FAIL: check_repo_hygiene")
        for item in findings:
            print(item)
        return 1

    print("PASS: check_repo_hygiene")
    return 0


if __name__ == "__main__":
    sys.exit(main())
