# P00-004 Report

## Goal/DoD
- Remove design-side `weights.h` dependency from allowed design files.
- Replace compile-time `w_*` reads with SRAM reads addressed by `param_base_word + kParamMeta[offset_w]`.
- Keep channel contract and ctrl bitfield contract unchanged.
- Update `check_design_purity.py` to fail on design-root `weights.h` include and `data/weights` include path usage.
- Verify by rebuild + `tb_top_m4` runtime + gates.

## What changed
- `src/blocks/FFNLayer0.h`:
  - Removed `weights.h` include.
  - Added SRAM-based weight/bias readers using `kParamMeta` offsets.
  - FFN bias/weight accesses now load from PARAM region (`param_base_word`) instead of compile-time arrays.
  - Extended `FFNLayer0(...)` signature to accept `param_base_word`.
- `src/blocks/TransformerLayer.h`:
  - Removed `weights.h` include.
  - Layer sublayer1 norm params now copied from PARAM SRAM via `kParamMeta` offsets.
  - Passed `pb.param_base_word` into `FFNLayer0(...)`.
- `src/blocks/FinalHead.h`:
  - Removed `weights.h` include.
  - Replaced compile-time `w_out_fc_bias[i]` access with SRAM read from `hp.out_fc_b_base_word + i`.
- `src/Top.h`:
  - Removed `weights.h` include.
  - Mid/end norm params now loaded from PARAM SRAM via `kParamMeta` IDs (mid: weight/bias 65/17, end: 64/16).
  - `run_mid_or_end_layernorm(...)` now receives `param_base_word`.
- `include/weights_streamer.h`:
  - Converted to design-safe shim without `weights.h`.
  - TB implementation moved to `tb/weights_streamer.h`.
- `tb/weights_streamer.h`:
  - Added TB-side streamer implementation (copied from previous include-side helper).
- `scripts/check_design_purity.py`:
  - Added fail rules:
    - `include_weights_h`
    - `include_data_weights_path`
- `README.md` AUTO block:
  - Added P00-004 build/run commands and validation stamp.

## Repro steps
Working directory: `c:\Users\Peter\source\repos\AECCT_HLS`

1. Build:
- `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights tb\tb_top_m4.cpp /Fe:build\tmp_tb_top_m4_p004.exe`
- `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights tb\tb_top_m5.cpp /Fe:build\tmp_tb_top_m5_p004.exe`
- `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights /I data\trace tb\tb_top_m3.cpp /Fe:build\tmp_tb_top_m3_p004.exe`
2. Run mandatory bring-up TB:
- `.\build\tmp_tb_top_m4_p004.exe`
3. Run gates:
- `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_gates.ps1`

## Test results
- `tb_top_m4`: build PASS, run PASS.
- `tb_top_m5`: build PASS, runtime SKIPPED (not mandatory in this patch).
- `tb_top_m3`: build PASS, runtime SKIPPED (not mandatory in this patch).

## Gate results
- `check_design_purity`: PASS
- `check_interface_lock`: PASS
- `check_repo_hygiene`: PASS
- Wrapper result: `PASS: all gates`

## New purity rule note
- `check_design_purity.py` now explicitly rejects in design roots:
  - `#include "weights.h"`
  - include paths containing `data/weights/` (or backslash variant)

## Design boundary check
- No Top channel type changes.
- No ctrl bitfield helper changes.
- Design-side `weights.h` include removed from allowed design files.