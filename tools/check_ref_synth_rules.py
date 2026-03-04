#!/usr/bin/env python3
import re
import sys
from pathlib import Path


FORBIDDEN_HEADERS = {
    "vector",
    "string",
    "fstream",
    "filesystem",
    "sstream",
    "algorithm",
    "cmath",
    "limits",
    "iomanip",
    "iostream",
}

FORBIDDEN_SYMBOL_PATTERNS = [
    r"\bstd::exp\b",
    r"\bstd::sqrt\b",
    r"\bstd::log\b",
    r"\bstd::pow\b",
]

FORBIDDEN_ALLOC_PATTERNS = [
    r"\bnew\b",
    r"\bmalloc\s*\(",
    r"\bfree\s*\(",
]

SOURCE_SUFFIXES = {".h", ".hpp", ".hh", ".c", ".cc", ".cpp", ".cxx"}
INCLUDE_RE = re.compile(r'^\s*#\s*include\s*([<"])\s*([^">]+)\s*[">]')


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*", "", text)
    return text


def strip_string_literals(text: str) -> str:
    text = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)
    text = re.sub(r"'(?:\\.|[^'\\])*'", "''", text)
    return text


def parse_includes(text: str) -> list[tuple[str, str]]:
    includes = []
    for line in text.splitlines():
        m = INCLUDE_RE.match(line)
        if m is not None:
            includes.append((m.group(1), m.group(2).strip()))
    return includes


def resolve_local_include(
    include_name: str,
    current_file: Path,
    synth_dir: Path,
    include_dir: Path,
    weights_dir: Path,
) -> Path | None:
    candidates = [
        current_file.parent / include_name,
        synth_dir / include_name,
        include_dir / include_name,
        weights_dir / include_name,
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c.resolve()
    return None


def path_in_allowed_roots(path: Path, roots: list[Path]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def detect_division_operator(text: str) -> bool:
    no_comments = strip_comments(text)
    no_strings = strip_string_literals(no_comments)
    body_lines = []
    for line in no_strings.splitlines():
        if line.lstrip().startswith("#"):
            continue
        body_lines.append(line)
    body = "\n".join(body_lines)
    return re.search(r"(?<![*/])/(?![*/])", body) is not None


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    synth_dir = repo_root / "AECCT_ac_ref" / "synth"
    include_dir = repo_root / "AECCT_ac_ref" / "include"
    weights_dir = repo_root / "data" / "weights"

    if not synth_dir.exists():
        print(f"FAIL: synth dir missing: {synth_dir}")
        return 1

    allowed_roots = [synth_dir.resolve(), include_dir.resolve(), weights_dir.resolve()]

    initial_files = []
    for p in synth_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in SOURCE_SUFFIXES:
            initial_files.append(p.resolve())

    scanned: dict[Path, str] = {}
    queue = list(initial_files)

    while queue:
        cur = queue.pop()
        if cur in scanned:
            continue
        text = cur.read_text(encoding="utf-8")
        scanned[cur] = text

        for delim, include_name in parse_includes(text):
            header_key = include_name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            if header_key in FORBIDDEN_HEADERS:
                rel = cur.relative_to(repo_root)
                print(f"FAIL: forbidden header '{header_key}' in {rel}")
                return 1

            if delim == '"':
                resolved = resolve_local_include(include_name, cur, synth_dir, include_dir, weights_dir)
                if resolved is not None and path_in_allowed_roots(resolved, allowed_roots):
                    queue.append(resolved)

    for path, text in scanned.items():
        rel = path.relative_to(repo_root)
        code = strip_comments(text)
        for patt in FORBIDDEN_SYMBOL_PATTERNS:
            if re.search(patt, code):
                print(f"FAIL: forbidden symbol '{patt}' in {rel}")
                return 1
        for patt in FORBIDDEN_ALLOC_PATTERNS:
            if re.search(patt, code):
                print(f"FAIL: forbidden alloc '{patt}' in {rel}")
                return 1
        if detect_division_operator(text):
            print(f"FAIL: division operator '/' found in {rel}")
            return 1

    print("PASS: check_ref_synth_rules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

