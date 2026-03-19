# TERNARY_LEAF_ROLEMAP_zhTW
Date: 2026-03-19

用途：快速區分 ternary leaf family 的角色邊界，降低 reviewer 重複比對成本。

<a id="file-role-comparison"></a>
## 1. File role comparison（核心對照）
| File | Ownership | Non-ownership | Input/Output surface | Relation to accepted local-only Q/KV path | Primary pointer |
| --- | --- | --- | --- | --- | --- |
| `src/blocks/TernaryLiveQkvLeafKernel.h` | row-kernel guard/decode/MAC；matrix-specific row materialization | 不擁有 SRAM policy、runtime scheduling、runtime-variable shape negotiation | 輸入 row + payload/meta，輸出 materialized row + `act_q` mirror | 核心 leaf compute 本體，供 generic entry 與 wrappers 委派 | `src/blocks/TernaryLiveQkvLeafKernel.h` + `REVIEWER_GUIDE` [6.4](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#ternary-role-section) |
| `src/blocks/TernaryLiveQkvLeafKernelTop.h` | fixed-shape local split-interface wrapper surface | 不擁有 SRAM policy 與 compile-prep policy | 輸入 fixed-shape wrapper payload，輸出委派到 split row kernels | AttnLayer0 在 local-only 路徑下優先使用的 wrapper 介面 | `src/blocks/TernaryLiveQkvLeafKernelTop.h` + `REVIEWER_GUIDE` [6.4](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#ternary-role-section) |
| `src/blocks/TernaryLiveQkvLeafKernelCatapultPrepTop.h` | compile-prep-facing wrapper surface | 不擁有 runtime policy 與外部 contract 擴張權 | 輸入 compile-prep wrapper payload，輸出委派到 leaf kernels | 用於 compile-prep family，不等於 runtime policy owner | `src/blocks/TernaryLiveQkvLeafKernelCatapultPrepTop.h` + `REVIEWER_GUIDE` [6.4](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#ternary-role-section) |
| `src/blocks/TernaryLiveQkvLeafKernelShapeConfig.h` | compile-time shape/payload constants SSOT | 不擁有 materialization logic 與 wrapper behavior | 輸入為 compile-time config；輸出為 constants 供其他檔案消費 | local-only path 的 shape expectation 基準，不含 runtime 行為 | `src/blocks/TernaryLiveQkvLeafKernelShapeConfig.h` + `REVIEWER_GUIDE` [6.4](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#ternary-role-section) |

<a id="loop-family-quick-map"></a>
## 2. Loop family quick map（GUI 快查）
- `TERNARY_QKV_IMPL_*`: generic SRAM-backed row kernel path。
- `TERNARY_WQ_SPLIT_*`: WQ fixed-shape split-interface row path。
- `TERNARY_WK_SPLIT_*`: WK fixed-shape split-interface row path。
- `TERNARY_WV_SPLIT_*`: WV fixed-shape split-interface row path。
- Primary pointer:
  - `REVIEWER_GUIDE` [7.4 TERNARY_*](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#ternary-loop-family)
  - `REVIEW_CHECKLIST` [ternary leaf family](./REVIEW_CHECKLIST_QKV_MAINLINE_zhTW.md#ternary-leaf-family)

## 3. Common misreadings（Reviewer warning）
- 誤讀 1: 把 `LeafKernelTop` 視為 SRAM policy owner。  
  正確: wrapper 只提供固定形狀入口，policy owner 在 Top。
- 誤讀 2: 把 `CatapultPrepTop` 視為 runtime contract 擴張點。  
  正確: compile-prep adapter，不主張 runtime policy。
- 誤讀 3: 把 `ShapeConfig` 當成含 materialization 行為的檔案。  
  正確: constants-only SSOT。
- Primary pointer:
  - `REVIEWER_GUIDE` [5.1 Lower-Level Ownership](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#lower-level-ownership)

## 4. PASS does / does-not prove（邊界聲明）
- PASS currently proves:
  - local-only path 下既有測試與流程可重現。
  - compile-prep family 在本地回歸範圍可通過。
- PASS currently does not prove:
  - Catapult closure
  - SCVerify closure
  - full runtime/global closure
- Primary pointer:
  - `REVIEWER_GUIDE` [9. PASS semantics](./REVIEWER_GUIDE_QKV_MAINLINE_zhTW.md#pass-semantics)
