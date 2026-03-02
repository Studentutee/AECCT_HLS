# P00-002 Report

## Goal/DoD
- Fix compliance mismatch for `OP_SET_W_BASE` expectation in `tb_top_m4`/`tb_top_m5` from `RSP_DONE` to `RSP_OK`.
- Apply minimal unblock fixes for mojibake-comment-swallowed statements in `tb_top_m4.cpp` and `tb_top_m5.cpp`.
- Keep all edits within P00-002 Scope Lock.
- Produce patch-level evidence pack under `docs/milestones/P00-002_artifacts/`.
- Produce milestone evidence packs for `M04` and `M05`.

## What changed
- Updated `tb/tb_top_m4.cpp`:
  - Replaced both `expect_rsp(... RSP_DONE, OP_SET_W_BASE)` with `expect_rsp(... RSP_OK, OP_SET_W_BASE)`.
  - Converted corrupted header/case comments to English.
  - Restored executable `drive_cmd(... OP_SOFT_RESET)` statements that were swallowed by corrupted comment lines.
- Updated `tb/tb_top_m5.cpp`:
  - Replaced both `expect_rsp(... RSP_DONE, OP_SET_W_BASE)` with `expect_rsp(... RSP_OK, OP_SET_W_BASE)`.
  - Converted corrupted header/case comments to English.
  - Restored executable `drive_debug_cfg(...)` call swallowed by corrupted comment line.
- Updated `README.md` AUTO block only:
  - Added P00-002 build/run commands for `tb_top_m4` and `tb_top_m5`.
  - Added validation stamp for `2026-03-02 (P00-002)`.

## Repro steps
Working directory: `c:\Users\Peter\source\repos\AECCT_HLS`

1. Build:
- `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights tb\tb_top_m4.cpp /Fe:build\tmp_tb_top_m4.exe`
- `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights tb\tb_top_m5.cpp /Fe:build\tmp_tb_top_m5.exe`
2. Run TB:
- `.\build\tmp_tb_top_m4.exe`
- `.\build\tmp_tb_top_m5.exe`
3. Run gates:
- `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_gates.ps1`
4. Full command output:
- `docs/milestones/P00-002_artifacts/build.log`
- `docs/milestones/P00-002_artifacts/run_tb.log`

## Test results
- `tb_top_m4`: PASS (`PASS: tb_top_m4 (SET_W_BASE + LOAD_W PARAM_RX)`).
- `tb_top_m5`: PASS (`PASS: tb_top_m5 (DEBUG_CFG + HALTED + RESUME)`).
- Both compile commands returned exit code `0`.

## Gate results
- `check_design_purity`: PASS
- `check_interface_lock`: PASS
- `check_repo_hygiene`: PASS
- Wrapper result: `PASS: all gates`

## Design boundary check
- Scope lock respected for P00-002 ALLOW set.
- No edit in `src/**`, `include/**`, `gen/**`, or design behavior.
- No trace/golden files modified.
- README manual section unchanged; only AUTO block modified.

## Known limitations
- MSVC compile warnings from `third_party/ac_types` and HLS pragmas are pre-existing and unchanged.
- Validation used Windows/MSVC flow.
