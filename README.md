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
- Bring-up TB build commands (P00-001):
  - `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights tb\tb_top_m0.cpp /Fe:build\tmp_tb_top_m0.exe`
  - `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights tb\tb_top_m1.cpp /Fe:build\tmp_tb_top_m1.exe`
  - `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights tb\tb_top_m2.cpp /Fe:build\tmp_tb_top_m2.exe`
- Bring-up TB run commands (P00-001):
  - `.\build\tmp_tb_top_m0.exe`
  - `.\build\tmp_tb_top_m1.exe`
  - `.\build\tmp_tb_top_m2.exe`
- Bring-up TB build commands (P00-002):
  - `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights tb\tb_top_m4.cpp /Fe:build\tmp_tb_top_m4.exe`
  - `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights tb\tb_top_m5.cpp /Fe:build\tmp_tb_top_m5.exe`
- Bring-up TB run commands (P00-002):
  - `.\build\tmp_tb_top_m4.exe`
  - `.\build\tmp_tb_top_m5.exe`
- Bring-up TB build commands (P00-003):
  - `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights /I data\trace tb\tb_top_m3.cpp /Fe:build\tmp_tb_top_m3.exe`
  - `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights /I data\trace tb\tb_top_m6.cpp /Fe:build\tmp_tb_top_m6.exe`
  - `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights /I data\trace tb\tb_top_m9.cpp /Fe:build\tmp_tb_top_m9.exe`
  - `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights /I data\trace tb\tb_layerloop_m11.cpp /Fe:build\tmp_tb_layerloop_m11.exe`
  - `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights /I data\trace tb\tb_mid_end_ln_m12.cpp /Fe:build\tmp_tb_mid_end_ln_m12.exe`
  - `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights /I data\trace tb\tb_top_end2end_m13.cpp /Fe:build\tmp_tb_top_end2end_m13.exe`
- Bring-up TB run command (P00-003 mandatory):
  - `.\build\tmp_tb_top_m3.exe`
- Bring-up TB build commands (P00-004):
  - `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights tb\tb_top_m4.cpp /Fe:build\tmp_tb_top_m4_p004.exe`
  - `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights tb\tb_top_m5.cpp /Fe:build\tmp_tb_top_m5_p004.exe`
  - `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights /I data\trace tb\tb_top_m3.cpp /Fe:build\tmp_tb_top_m3_p004.exe`
- Bring-up TB run command (P00-004 mandatory):
  - `.\build\tmp_tb_top_m4_p004.exe`
- Bring-up TB build commands (P00-005 Step 2 governance evidence):
  - `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights tb\tb_top_m0.cpp /Fe:build\tmp_tb_top_m0_p005.exe`
  - `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights tb\tb_top_m1.cpp /Fe:build\tmp_tb_top_m1_p005.exe`
  - `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights tb\tb_top_m2.cpp /Fe:build\tmp_tb_top_m2_p005.exe`
- Bring-up TB run command (P00-005 mandatory):
  - `.\build\tmp_tb_top_m0_p005.exe`
- Bring-up TB runtime scope note (P00-005):
  - `tb_top_m1` and `tb_top_m2` are compile-compatible only in this step (runtime not required).
- Smoke TB build/run command:
  - `cl /nologo /std:c++20 /EHsc /utf-8 /I . /I include /I src /I third_party\ac_types /I data\weights tb\tb_compliance_smoke_p16.cpp /Fe:build\tmp_tb_compliance_smoke_p16.exe`
  - `.\build\tmp_tb_compliance_smoke_p16.exe`
- Latest validation stamp:
  - `2026-03-02 (P00-001)`: tb_top_m0 PASS, tb_top_m1 PASS, tb_top_m2 PASS, gates PASS.
  - `2026-03-02 (P00-002)`: tb_top_m4 PASS, tb_top_m5 PASS, gates PASS.
  - `2026-03-02 (P00-003)`: tb_top_m3 PASS, tb_top_m6/m9/m11/m12/m13 build PASS, gates PASS.
  - `2026-03-02 (P00-004)`: design-side weights.h removed, tb_top_m4 PASS, gates PASS.
  - `2026-03-12 (P00-005)`: Step 2 skeleton contract convergence tracked; tb_top_m0 smoke PASS; tb_top_m1/m2 build PASS (compile-compatible only); check_design_purity PASS; check_interface_lock PASS; check_repo_hygiene FAIL (pre-existing findings); governance closure pending.
<!-- AUTO-GENERATED END -->

