# P00-001 Report

## Goal/DoD
- Fix bring-up TB alignment issues for `tb_top_m0`, `tb_top_m1`, and `tb_top_m2` according to patch spell `P00-001`.
- Align SSOT opcode for `OP_SOFT_RESET` with spell v11.7.2.
- Align Top behavior for `OP_SET_W_BASE` response kind (`OK`, not `DONE`).
- Produce evidence pack under `docs/milestones/P00-001_artifacts/` with `file_manifest.txt`, `diff.patch`, `build.log`, `run_tb.log`, and `verdict.txt`.
- Reproduce baseline behavior, then verify patched behavior with required build/run/gate commands.

## What changed
- Updated `include/AecctProtocol.h`:
  - `OP_SOFT_RESET` changed from `0x0F` to `0x7F`.
  - Replaced non-English comments with English comments from patch spell E1.
  - Kept ctrl command/response bitfield helpers unchanged.
- Updated `src/Top.h`:
  - In `ST_IDLE` + `OP_SET_W_BASE` success path, changed response from `pack_ctrl_rsp_done` to `pack_ctrl_rsp_ok`.
  - No other opcode handler behavior changed.
- Replaced `tb/tb_top_m0.cpp` with spell E3 version.
- Replaced `tb/tb_top_m1.cpp` with spell E4 version.
- Replaced `tb/tb_top_m2.cpp` with spell E5 version.
- Updated only `README.md` AUTO block:
  - Added `tb_top_m0/m1/m2` build/run commands.
  - Added latest validation stamp for `2026-03-02 (P00-001)`.
- Added milestone artifacts and this report.

## Repro steps
Working directory: `c:\Users\Peter\source\repos\AECCT_HLS`

1. Build (baseline and patched used same commands):
- `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights tb\tb_top_m0.cpp /Fe:build\tmp_tb_top_m0.exe`
- `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights tb\tb_top_m1.cpp /Fe:build\tmp_tb_top_m1.exe`
- `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights tb\tb_top_m2.cpp /Fe:build\tmp_tb_top_m2.exe`
2. Run TB:
- `.\build\tmp_tb_top_m0.exe`
- `.\build\tmp_tb_top_m1.exe`
- `.\build\tmp_tb_top_m2.exe`
3. Run gates:
- `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_gates.ps1`
4. Full command output is captured in:
- `docs/milestones/P00-001_artifacts/build.log`
- `docs/milestones/P00-001_artifacts/run_tb.log`

## Test results
Baseline (`=== BASELINE ===`):
- `tb_top_m0`: FAIL (`ERROR: rsp mismatch. kind=0 payload=1 (expect ERR=2, err=1)`).
- `tb_top_m1`: FAIL (`ERROR: rsp mismatch. kind=2 payload=6 (expect kind=1 payload=2)`).
- `tb_top_m2`: FAIL (`ERROR: rsp mismatch. kind=2 payload=6 (expect kind=1 payload=2)`).
- Baseline compile succeeded for all three TB executables.

Patched (`=== PATCHED ===`):
- `tb_top_m0`: PASS.
- `tb_top_m1`: PASS.
- `tb_top_m2`: PASS.
- Patched compile succeeded for all three TB executables.

## Gate results
- `check_design_purity`: PASS
- `check_interface_lock`: PASS
- `check_repo_hygiene`: PASS
- Wrapper result: `PASS: all gates`

## Design boundary check
- Scope lock respected: only ALLOW files were modified.
- No changes to Top 4 channel types.
- No changes to ctrl_cmd/ctrl_rsp bitfield packing rules.
- No trace/golden/data files were modified.
- README Manual section was not modified; only AUTO block was updated.

## Known limitations
- Build emits existing third-party warning noise from `third_party/ac_types` and HLS pragmas under MSVC; these warnings are pre-existing and were not modified by this patch.
- Verification in this patch is Windows `cl` flow only (as specified in spell repro steps).

## Next step
- If desired, add a dedicated `check_tb_flow` gate in a separate patch to mechanically enforce bring-up TB command sequencing and prevent future protocol drift.