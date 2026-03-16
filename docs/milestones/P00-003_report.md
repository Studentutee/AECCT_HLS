# P00-003 Report

## Goal/DoD
- Align `CFG_COMMIT` success response to v11.7.2: `RSP_OK(OP_CFG_COMMIT)`.
- Keep CFG failure behavior unchanged (`ERR_CFG_LEN_MISMATCH`, `ERR_CFG_ILLEGAL`).
- Align TB expectations (`m3/m6/m9/m11/m12/m13`) to `RSP_OK(OP_CFG_COMMIT)`.
- Add trace-optional guards for trace-dependent TBs with default `AECCT_HAS_TRACE=0`.
- Produce `docs/milestones/P00-003_artifacts/*` and `docs/milestones/M03_artifacts/*`.

## What changed
- `src/Top.h`:
  - In `ST_CFG_RX` success branch of `OP_CFG_COMMIT`, changed response from `pack_ctrl_rsp_done(OP_CFG_COMMIT)` to `pack_ctrl_rsp_ok(OP_CFG_COMMIT)`.
  - No other state transition or error handling logic changed.
- `tb/tb_top_m3.cpp`:
  - Updated CFG commit success expectations from `RSP_DONE` to `RSP_OK`.
  - Header comment converted to English.
- `tb/tb_top_m6.cpp`:
  - Updated CFG commit success expectation from `RSP_DONE` to `RSP_OK`.
  - Applied minimal unblock fix for mojibake-comment-swallowed `OP_SOFT_RESET` call in Case 5.
- `tb/tb_top_m9.cpp`, `tb/tb_layerloop_m11.cpp`, `tb/tb_mid_end_ln_m12.cpp`, `tb/tb_top_end2end_m13.cpp`:
  - Updated CFG commit success expectations from `RSP_DONE` to `RSP_OK`.
  - Added trace guards:
    - `AECCT_HAS_TRACE` default `0`.
    - `AECCT_TRACE_AVAILABLE` via `__has_include(...)` checks.
  - Trace compare paths are skipped when trace is disabled/unavailable.
- `README.md` AUTO block:
  - Added P00-003 build commands (m3/m6/m9/m11/m12/m13), mandatory run command (m3), and validation stamp.

## Repro steps
Working directory: `c:\Users\Peter\source\repos\AECCT_HLS`

1. Build:
- `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights /I data\trace tb\tb_top_m3.cpp /Fe:build\tmp_tb_top_m3.exe`
- `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights /I data\trace tb\tb_top_m6.cpp /Fe:build\tmp_tb_top_m6.exe`
- `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights /I data\trace tb\tb_top_m9.cpp /Fe:build\tmp_tb_top_m9.exe`
- `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights /I data\trace tb\tb_layerloop_m11.cpp /Fe:build\tmp_tb_layerloop_m11.exe`
- `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights /I data\trace tb\tb_mid_end_ln_m12.cpp /Fe:build\tmp_tb_mid_end_ln_m12.exe`
- `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights /I data\trace tb\tb_top_end2end_m13.cpp /Fe:build\tmp_tb_top_end2end_m13.exe`
2. Run mandatory TB:
- `.\build\tmp_tb_top_m3.exe`
3. Run gates:
- `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_gates.ps1`

## Test results
- `tb_top_m3`: PASS.
- `tb_top_m6`: build PASS (runtime SKIPPED by patch policy).
- `tb_top_m9`: build PASS (runtime SKIPPED by patch policy; trace compare guarded when `AECCT_HAS_TRACE=0`).
- `tb_layerloop_m11`: build PASS (runtime SKIPPED by patch policy; trace compare guarded when `AECCT_HAS_TRACE=0`).
- `tb_mid_end_ln_m12`: build PASS (runtime SKIPPED by patch policy; trace compare guarded when `AECCT_HAS_TRACE=0`).
- `tb_top_end2end_m13`: build PASS (runtime SKIPPED by patch policy; trace compare guarded when `AECCT_HAS_TRACE=0`).

## Gate results
- `check_design_purity`: PASS
- `check_interface_lock`: PASS
- `check_repo_hygiene`: PASS
- Wrapper result: `PASS: all gates`

## Design boundary check
- Top channel types unchanged.
- Ctrl bitfield pack/unpack unchanged.
- Scope lock respected for P00-003 ALLOW set.

## Notes
- Existing compiler warning noise from `third_party/ac_types` remains unchanged.