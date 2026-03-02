# M05 Report

## Goal/DoD
- Align `tb_top_m5` to v11.7.2 contract for `OP_SET_W_BASE` success response (`RSP_OK`).
- Remove mojibake comment corruption that swallowed executable statements.
- Produce `M05_artifacts` evidence pack.

## Changes
- `tb/tb_top_m5.cpp`:
  - `OP_SET_W_BASE` expectation updated from `RSP_DONE` to `RSP_OK` at both occurrences.
  - Corrupted comments replaced with English comments.
  - Swallowed `drive_debug_cfg(...)` call restored.

## Repro
- Build:
  - `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights tb\tb_top_m5.cpp /Fe:build\tmp_tb_top_m5.exe`
- Run:
  - `.\build\tmp_tb_top_m5.exe`
- Gates:
  - `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_gates.ps1`

## Result
- `tb_top_m5`: PASS
- Gates: PASS
- Verdict: `PASS`
