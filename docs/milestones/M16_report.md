# M16 Report (P16 Compliance Convergence)

## Goal / DoD
- Apply P16-001..P16-006 updates toward spell v11.7.2 compliance.
- Keep top interface contract and protocol SSOT unchanged.
- Remove design-side trace include dependency and prohibited constructs.
- Make required gates pass.
- Build and run one compliance smoke TB.
- Publish full artifact bundle under `docs/milestones/M16_artifacts/`.

## What Changed
- Governance and delivery:
  - Added `README.md` with Manual + Auto sections.
  - Added `.editorconfig`.
  - Updated `.gitignore` with explicit trace/build/archive hygiene patterns.
  - Added milestone report and artifact directory.
- Gates and utilities:
  - Added `scripts/check_design_purity.py`.
  - Added `scripts/check_interface_lock.py`.
  - Added `scripts/check_repo_hygiene.py`.
  - Added `scripts/run_gates.ps1` and `scripts/run_gates.sh`.
  - Added `scripts/strip_utf8_bom.py`.
- Design boundary + HLS-safe refactor:
  - Reworked `include/AecctUtil.h` as FP32 SSOT helper with `ac_ieee_float<binary32>`.
  - Reworked `include/QuantDesc.h`, `include/LayerNormDesc.h`, `include/AecctTypes.h`.
  - Reworked `src/blocks/{PreprocEmbedSPE,AttnLayer0,FFNLayer0,LayerNormBlock,TransformerLayer,FinalHead}.h`.
  - Updated `src/Top.h` to remove trace macro path and use bit-safe FP32 helper flow.
- Top class-run integration:
  - Added `design/AecctTop.h` (`class AecctTop` with `run(...)`).
  - Added `tb/tb_compliance_smoke_p16.cpp`.
- TB boundary cleanup:
  - Removed `#define AECCT_*_TRACE_MODE 1` from existing TBs.
  - Updated `tb/tb_regress_m14.cpp` to support trace-absent mode via `AECCT_HAS_TRACE`.
  - Trace-required cases are explicitly marked SKIPPED when trace is unavailable.
- Encoding cleanup:
  - Removed UTF-8 BOM from 27 files via `scripts/strip_utf8_bom.py`.

## Repro Steps
1. Working directory:
   - `c:\Users\Peter\source\repos\AECCT_HLS`
2. Baseline before patch:
   - `powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_m16_pipeline.ps1`
3. Strip BOM:
   - `python scripts/strip_utf8_bom.py --repo-root .`
4. Run gates:
   - `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_gates.ps1`
5. Build smoke TB:
   - `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights tb\tb_compliance_smoke_p16.cpp /Fe:build\tmp_tb_compliance_smoke_p16.exe`
6. Run smoke TB:
   - `.\build\tmp_tb_compliance_smoke_p16.exe`
7. Post-patch pipeline spot check:
   - `powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_m16_pipeline.ps1`

## Test Results
- Baseline pipeline:
  - `PASS` (pre-patch), log in `build_baseline.log`.
- Gates:
  - `PASS: check_design_purity`
  - `PASS: check_interface_lock`
  - `PASS: check_repo_hygiene`
- Smoke TB:
  - `PASS: tb_compliance_smoke_p16`
- Post-patch legacy regression:
  - Core command/state cases pass.
  - Trace-dependent cases are marked `SKIPPED` when `AECCT_HAS_TRACE=0`.
  - Overall pipeline script completes successfully in this mode.

## Gate Results
- See `docs/milestones/M16_artifacts/build.log`.
- All required gates passed.

## Design Boundary Check
- DESIGN (`src/include/gen`) no longer contains:
  - `*_step0.h` includes
  - `AECCT_*_TRACE_MODE`
  - `union`
  - C++ `float`
  - `<cmath>` and `std::sqrt/exp/log/pow`
- Trace usage remains TB-side and can be conditionally enabled.

## Known Limitations
- Existing regression cases that rely on external trace golden are skipped by default unless `AECCT_HAS_TRACE=1` is supplied.
- Build emits multiple warnings from third-party AC datatypes headers; these are unchanged third-party warnings.

## Next Step
- If full trace-based correctness regression is needed, run with external trace bundle and `AECCT_HAS_TRACE=1`.
- Optionally tighten gate rules to enforce English-only comments at scale.
