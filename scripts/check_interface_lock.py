#!/usr/bin/env python3
"""Check interface contract and protocol SSOT anchors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REQUIRED_AECCT_TYPES = [
    "typedef ac_int<16, false> u16_t;",
    "typedef ac_int<32, false> u32_t;",
    "typedef ac_channel<u16_t> ctrl_ch_t;",
    "typedef ac_channel<u32_t> data_ch_t;",
]

REQUIRED_PROTOCOL_FUNCS = [
    "pack_ctrl_cmd(",
    "unpack_ctrl_cmd_opcode(",
    "pack_ctrl_rsp_ok(",
    "pack_ctrl_rsp_done(",
    "pack_ctrl_rsp_err(",
    "unpack_ctrl_rsp_kind(",
    "unpack_ctrl_rsp_payload(",
]

REQUIRED_TOP_PARAMS = [
    "ac_channel<ac_int<16, false> >& ctrl_cmd",
    "ac_channel<ac_int<16, false> >& ctrl_rsp",
    "ac_channel<ac_int<32, false> >& data_in",
    "ac_channel<ac_int<32, false> >& data_out",
]


def require_strings(path: Path, checks: list[str], findings: list[str], tag: str) -> None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    for item in checks:
        if item not in text:
            findings.append(f"{tag}: missing `{item}` in {path.as_posix()}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    findings: list[str] = []

    types_h = repo / "include" / "AecctTypes.h"
    protocol_h = repo / "include" / "AecctProtocol.h"
    top_h = repo / "src" / "Top.h"

    for p in (types_h, protocol_h, top_h):
        if not p.exists():
            findings.append(f"missing file: {p.as_posix()}")

    if findings:
        print("FAIL: check_interface_lock")
        for item in findings:
            print(item)
        return 1

    require_strings(types_h, REQUIRED_AECCT_TYPES, findings, "type_lock")
    require_strings(protocol_h, REQUIRED_PROTOCOL_FUNCS, findings, "protocol_ssot")
    require_strings(top_h, REQUIRED_TOP_PARAMS, findings, "top_signature")

    if findings:
        print("FAIL: check_interface_lock")
        for item in findings:
            print(item)
        return 1

    print("PASS: check_interface_lock")
    return 0


if __name__ == "__main__":
    sys.exit(main())
