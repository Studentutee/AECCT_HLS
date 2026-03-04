#!/usr/bin/env python3
import re
from pathlib import Path


def extract_block(text: str, begin: str, end: str) -> str:
    i = text.find(begin)
    j = text.find(end)
    if i < 0 or j < 0 or j <= i:
        raise RuntimeError(f"marker missing: {begin}..{end}")
    return text[i:j + len(end)]


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*", "", text)
    return text


def assert_no(pattern: str, text: str, where: str) -> None:
    if re.search(pattern, text):
        raise RuntimeError(f"forbidden in {where}: {pattern}")


def main() -> int:
    p = Path("AECCT_ac_ref/src/RefModel.cpp")
    text = p.read_text(encoding="utf-8")

    for whole_forbidden in [r"#include\s*<cmath>", r"std::exp", r"std::sqrt", r"std::log", r"std::pow"]:
        assert_no(whole_forbidden, text, "RefModel.cpp")

    soft = strip_comments(extract_block(text, "// SOFTMAX_APPROX_BEGIN", "// SOFTMAX_APPROX_END"))
    ln = strip_comments(extract_block(text, "// LN_APPROX_BEGIN", "// LN_APPROX_END"))

    for blk_name, blk in [("softmax", soft), ("layernorm", ln)]:
        for patt in [r"std::exp", r"std::sqrt", r"std::log", r"std::pow"]:
            assert_no(patt, blk, blk_name)
        assert_no(r"(?<![*/])/ (?![*/])", blk, blk_name)
        assert_no(r"(?<![*/])/(?![*/])", blk, blk_name)

    print("PASS: ref approx rules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
