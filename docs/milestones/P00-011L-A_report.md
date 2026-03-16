# P00-011L-A Report — Repo-Tracked Split-Interface Local Smoke Repair

## Goal / Scope
- 目標：修復 repo-tracked `tb/tb_ternary_live_leaf_top_smoke_p11k.cpp` 與 `src/blocks/TernaryLiveQkvLeafKernelTop.h`，讓 split-interface P11K smoke 在無 `mc_scverify.h` 的本地環境可 build + run。
- 範圍：只做 P00-011L-A 所需最小修正；不回退已接受的 P00-011F ~ P00-011K。
- 本報告記錄的是 **local smoke scope acceptance**，不是 Catapult / SCVerify 正式 closure。

## Baseline / Authority Used
- `docs/process/GOVERNANCE_ENTRYPOINT_zhTW.txt`
- `docs/process/PROJECT_STATUS_zhTW.txt`
- `docs/process/AECCT_PROJECT_WORKFLOW_v1_zhTW.txt`
- `docs/reference/Catapult_C++_CodeGen_Guide_for_Codex_v3.txt`
- `docs/spec/AECCT_HLS_Spec_v12.1_zhTW.txt`
- `docs/architecture/AECCT_HLS_Architecture_Guide_v12.1_zhTW.txt`
- `docs/architecture/AECCT_module_interface_rules_v12.1_zhTW.txt`
- `docs/architecture/AECCT_SramMap_alias_freepoint_rules_v12.1_zhTW.txt`

## Pre-Patch Findings
- `src/blocks/TernaryLiveQkvLeafKernelTop.h` 已是 split-interface `run(...)` 介面，但在無 SCVerify header 的情況下缺 `CCS_BLOCK` fallback。
- `tb/tb_ternary_live_leaf_top_smoke_p11k.cpp` 仍停留在舊的 `run(sram, param_base, ...)` 呼叫模式，未對齊 split-interface row-kernel top。
- 這一輪的首要 blocker 是 `CCS_BLOCK(run)` 無 fallback 導致 `run` 解析失敗；本地 `mc_scverify.h` hard dependency 風險也需一起消除。

## Files Changed
- `src/blocks/TernaryLiveQkvLeafKernelTop.h`
- `tb/tb_ternary_live_leaf_top_smoke_p11k.cpp`

## Files Not Modified
- `tb/tb_ternary_live_leaf_smoke_p11j.cpp`
  - 作為回歸驗證基準，未修改。

## Implementation Summary
### `src/blocks/TernaryLiveQkvLeafKernelTop.h`
- 保持 split-interface `run(...)` public signature 不變。
- 保留 `mc_scverify.h` optional include（`__has_include`）。
- 新增 fallback：
  - `#ifndef CCS_BLOCK`
  - `#define CCS_BLOCK(name) name`
  - `#endif`

### `tb/tb_ternary_live_leaf_top_smoke_p11k.cpp`
- 呼叫介面改為 split-interface：
  - `x_row`
  - `payload_words`
  - `inv_sw_bits`
  - `out_row`
  - `out_act_q_row`
  - `out_inv_sw_bits`
- TB 內建立並填充 `x_row[32]`、`payload_words[64]`，並以 direct P11J kernel 產生 reference，對 `out_row / out_act_q_row / out_inv_sw_bits` 做 bit-exact 比對。
- 新增 dual-mode entry：
  - 有 `mc_scverify.h`：`CCS_MAIN / CCS_RETURN`
  - 無 `mc_scverify.h`：一般 `int main() / return`

## Exact Build Commands
1. `New-Item -ItemType Directory -Force -Path build\p11l_a > $null; cl /nologo /std:c++14 /EHsc /utf-8 /I . /I include /I src /I gen\include /I third_party\ac_types /I data\weights tb\tb_ternary_live_leaf_smoke_p11j.cpp /Fe:build\p11l_a\tb_ternary_live_leaf_smoke_p11j.exe > build\p11l_a\build_p11j.log 2>&1`
2. `cl /nologo /std:c++14 /EHsc /utf-8 /I . /I include /I src /I gen\include /I third_party\ac_types /I data\weights tb\tb_ternary_live_leaf_top_smoke_p11k.cpp /Fe:build\p11l_a\tb_ternary_live_leaf_top_smoke_p11k.exe > build\p11l_a\build_p11k.log 2>&1`

## Exact Run Commands
1. `cmd /c build\p11l_a\tb_ternary_live_leaf_smoke_p11j.exe > build\p11l_a\run_p11j.log 2>&1`
2. `cmd /c build\p11l_a\tb_ternary_live_leaf_top_smoke_p11k.exe > build\p11l_a\run_p11k.log 2>&1`

## Execution Evidence Excerpt
### Build (`build_p11k.log`)
- 無 `mc_scverify.h` 缺檔錯誤
- 無 `CCS_BLOCK` 未定義錯誤
- 無 `run` 宣告/成員缺失錯誤
- 僅見 warning，例如：
  - `src\blocks/TernaryLiveQkvLeafKernelTop.h(17): warning C4068: unknown pragma 'hls_design'`

### Run (`run_p11j.log`)
- `[p11j][PASS] kernel call succeeded for ternary_live_l0_wq_materialize_row_kernel`
- `PASS: tb_ternary_live_leaf_smoke_p11j`

### Run (`run_p11k.log`)
- `[p11k][PASS] split-interface top run() exact-match equivalent to direct P11J kernel output`
- `PASS: tb_ternary_live_leaf_top_smoke_p11k`

## Verified / Not Verified
### Verified
- repo-tracked `Top.h` 與 `P11K TB` 已對齊本輪目標。
- `TernaryLiveL0WqRowTop::run(...)` public signature 維持 split-interface，不靠改簽章解問題。
- local smoke scope 下，P11J / P11K 皆有 PASS evidence。
- `mc_scverify.h` 不再是本地 smoke 的 hard dependency。

### Not Verified
- 本輪未做 Catapult GUI / SCVerify 實跑。
- 本輪未擴大到 full live family closure，也未宣稱 numeric / runtime 全收斂。
- 本輪 raw local logs 未完整鏡像成正式 artifacts bundle；此報告記錄的是已接受的 local smoke evidence 摘要。

## Deferred Items
- Catapult / SCVerify validation：deferred，預計與後續同 family 項目一起批次驗證。
- 更大範圍的 ternary live migration / runtime closure：未納入本輪。

## Final Conclusion
- P00-011L-A 可接受為 **repo-tracked local smoke repair 完成**。
- 正式接受範圍限於本地 smoke；**不等同** Catapult / SCVerify 正式 closure。
