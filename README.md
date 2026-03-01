# AECCT_HLS M16 Fork

## Manual

### Contract Lock
- Top interface channels are fixed:
  - `ctrl_cmd`: `ac_channel<ac_int<16,false>>`
  - `ctrl_rsp`: `ac_channel<ac_int<16,false>>`
  - `data_in`: `ac_channel<ac_int<32,false>>`
  - `data_out`: `ac_channel<ac_int<32,false>>`
- `ctrl_cmd` / `ctrl_rsp` pack-unpack logic is SSOT in `include/AecctProtocol.h`.
- Response order remains `OK -> DONE`; error path returns `ERR`.

### Design Boundary
- Design code is restricted to `src/`, `include/`, and `gen/`.
- Design code must not include trace headers such as `*_step0.h`.
- Trace-based compare logic belongs to `tb/` only.

### HLS-Safe Subset
- Design code must avoid host `<cmath>` math calls.
- Design code must avoid C++ `union` bit-cast.
- Design code must avoid C++ `float` type and use AC types.

### Gate Entry
- `python scripts/check_design_purity.py --repo-root .`
- `python scripts/check_interface_lock.py --repo-root .`
- `python scripts/check_repo_hygiene.py --repo-root .`

<!-- AUTO-GENERATED BEGIN -->
## Auto

- Baseline pipeline command:
  - `powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_m16_pipeline.ps1`
- Gate wrapper:
  - `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_gates.ps1`
- Smoke TB build/run command:
  - `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights tb\tb_compliance_smoke_p16.cpp /Fe:build\tmp_tb_compliance_smoke_p16.exe`
  - `.\build\tmp_tb_compliance_smoke_p16.exe`
- Latest validation stamp:
  - `2026-03-02`: gates PASS, smoke TB PASS, legacy trace-required regression cases SKIPPED with `AECCT_HAS_TRACE=0`.
<!-- AUTO-GENERATED END -->
