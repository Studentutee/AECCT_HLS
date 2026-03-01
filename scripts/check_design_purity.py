#!/usr/bin/env python3
"""Check design files for disallowed constructs."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


TEXT_SUFFIXES = {".h", ".hpp", ".hh", ".c", ".cc", ".cpp", ".cxx"}
DESIGN_ROOTS = ("src", "include", "gen")

RULES = [
    ("include_step0", re.compile(r'#include\s*["<][^">]*step0\.h[">]')),
    ("trace_macro", re.compile(r"\bAECCT_.*TRACE_MODE\b")),
    ("host_cmath", re.compile(r"#include\s*<cmath>")),
    ("host_std_math", re.compile(r"\bstd::(?:sqrt|exp|log|pow)\b")),
    ("union_keyword", re.compile(r"\bunion\b")),
    ("float_keyword", re.compile(r"\bfloat\b")),
]


def iter_design_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for rel in DESIGN_ROOTS:
        base = root / rel
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                files.append(path)
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    findings: list[str] = []

    for path in iter_design_files(repo):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for rule_name, rule in RULES:
                if not rule.search(line):
                    continue
                if rule_name == "float_keyword" and "ac_ieee_float" in line:
                    continue
                rel = path.relative_to(repo).as_posix()
                findings.append(f"{rel}:{lineno}: {rule_name}: {line.strip()}")

    if findings:
        print("FAIL: check_design_purity")
        for item in findings:
            print(item)
        return 1

    print("PASS: check_design_purity")
    return 0


if __name__ == "__main__":
    sys.exit(main())
