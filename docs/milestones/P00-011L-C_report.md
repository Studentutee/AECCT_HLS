# P00-011L-C Report — TB-Local Family Common Smoke for QKV Split-Interface Tops

## Goal / Scope
- 目標：以 TB-local common driver 整理 `L0_WQ / L0_WK / L0_WV` split-interface local top smoke。
- 範圍：commonization 只在 TB 內；不修改 production-side implementation。
- 本報告記錄的是 **local smoke scope accepted**，不是 full closure。

## Acceptance Status
- Implementation / local smoke: accepted
- Catapult / SCVerify: deferred

## Files Changed
- `tb/tb_ternary_live_leaf_top_smoke_p11l_c.cpp`

## Regression Baseline Not Modified
- `tb/tb_ternary_live_leaf_smoke_p11j.cpp`
- `tb/tb_ternary_live_leaf_top_smoke_p11k.cpp`
- `tb/tb_ternary_live_leaf_top_smoke_p11l_b.cpp`

## Exact Build Commands
1. `New-Item -ItemType Directory -Force -Path build\p11l_c > $null`
2. `cl /nologo /std:c++14 /EHsc /utf-8 /I . /I include /I src /I gen\include /I third_party\ac_types /I data\weights tb\tb_ternary_live_leaf_smoke_p11j.cpp /Fe:build\p11l_c\tb_ternary_live_leaf_smoke_p11j.exe > build\p11l_c\build_p11j.log 2>&1`
3. `cl /nologo /std:c++14 /EHsc /utf-8 /I . /I include /I src /I gen\include /I third_party\ac_types /I data\weights tb\tb_ternary_live_leaf_top_smoke_p11k.cpp /Fe:build\p11l_c\tb_ternary_live_leaf_top_smoke_p11k.exe > build\p11l_c\build_p11k.log 2>&1`
4. `cl /nologo /std:c++14 /EHsc /utf-8 /I . /I include /I src /I gen\include /I third_party\ac_types /I data\weights tb\tb_ternary_live_leaf_top_smoke_p11l_b.cpp /Fe:build\p11l_c\tb_ternary_live_leaf_top_smoke_p11l_b.exe > build\p11l_c\build_p11l_b.log 2>&1`
5. `cl /nologo /std:c++14 /EHsc /utf-8 /I . /I include /I src /I gen\include /I third_party\ac_types /I data\weights tb\tb_ternary_live_leaf_top_smoke_p11l_c.cpp /Fe:build\p11l_c\tb_ternary_live_leaf_top_smoke_p11l_c.exe > build\p11l_c\build_p11l_c.log 2>&1`

## Exact Run Commands
1. `cmd /c build\p11l_c\tb_ternary_live_leaf_smoke_p11j.exe > build\p11l_c\run_p11j.log 2>&1`
2. `cmd /c build\p11l_c\tb_ternary_live_leaf_top_smoke_p11k.exe > build\p11l_c\run_p11k.log 2>&1`
3. `cmd /c build\p11l_c\tb_ternary_live_leaf_top_smoke_p11l_b.exe > build\p11l_c\run_p11l_b.log 2>&1`
4. `cmd /c build\p11l_c\tb_ternary_live_leaf_top_smoke_p11l_c.exe > build\p11l_c\run_p11l_c.log 2>&1`

## Execution Evidence Excerpt (Accepted Local Evidence Summary)
- `PASS: tb_ternary_live_leaf_smoke_p11j`
- `PASS: tb_ternary_live_leaf_top_smoke_p11k`
- `PASS: tb_ternary_live_leaf_top_smoke_p11l_b`
- `L0_WQ split-interface top run() exact-match equivalent to direct kernel output`
- `L0_WK split-interface top run() exact-match equivalent to direct kernel output`
- `L0_WV split-interface top run() exact-match equivalent to direct kernel output`
- `PASS: tb_ternary_live_leaf_top_smoke_p11l_c`

## Deferred
- Catapult / SCVerify closure is deferred and not included in this acceptance scope.
