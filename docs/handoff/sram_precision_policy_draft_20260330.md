# SRAM Precision Policy Draft (2026-03-30, updated after INT8 fixed-exp + FP16-global explore)

## 1. Draft Positioning
- 這是 task-local architecture/policy 草案，服務於 ref-only 量化探索。
- 不是 frozen spec，不是 HLS mainline closure，不是 Catapult/SCVerify closure。

## 2. Core Objective
在「SRAM 容量優先」前提下，同時兼顧：
1. 計算便利性（widen/unpack/rescale 複雜度可控）
2. compute-area 與平行化（小位寬單元更容易擴並行）
3. evaluator quality（trace-reference-aligned BER/FER）

## 3. Precision Policy Principles
1. 大容量物件優先壓低 storage bitwidth。
2. 小量但高敏感 scalar/local states 可保較高精度。
3. 必須分離 storage / on-read widen / compute / accumulator / write-back。
4. `W_REGION` ternary payload 已是 2-bit packed，不可誤解成要改 8-bit。
5. shared-exp INT8 是大頁面候選；FP16 是高敏感例外層候選。

## 4. Latest Ref Evidence Linkage
- `G2+embed_only`（敏感路徑）在 `begin=4,count=12`：
  - E4M3：`delta BER>0`, `delta FER>0`
  - INT8 fixed-exp zone3：`delta BER=0`, `delta FER=0`
- `FP16_REPLACE_FP32_GLOBAL`：
  - 0~3 / 4~15 / 16~31 都是 `delta BER=0`, `delta FER=0`
  - compare 中 x_pred/sign flips 都為 0
  - fp16 nonfinite/underflow counters 都為 0

## 5. Storage/Compute Policy Table

| Storage / object | Role / semantic meaning | Capacity driver | Current form | Proposed storage bitwidth | Proposed compute bitwidth | Proposed accumulator / scalar-state bitwidth | Allow widen-on-read? | Why not lower? / Why can lower? | Fragility evidence linkage | HW convenience / area / parallelism comment | Priority | Validation next step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `W_REGION` ternary payload | native linear quant weight payload | persistent heavy | ternary packed (2-bit class) | keep packed ternary | integer/ternary MAC | INT16 (native path) | Yes | 已比8-bit更省；再改8-bit收益低 | full stress 不指向 W payload | decode 固定後可高並行且不增 SRAM | P0 keep | 維持現狀，驗證 unpack throughput |
| `W_REGION` metadata (`inv_s_w`, matrix metadata) | dequant/scale metadata | small persistent | FP32/U32 mixed | 16-bit or mixed | FP16/FP32 | FP32 | Yes | metadata 量小但影響 scale 正確性 | 主要 fragility 在 activation side | 控制複雜度低，壓縮空間有限 | P1 | 盤點 metadata 精度敏感度 |
| `X_WORK` | main token×d working set | active working-set heavy | baseline FP32 | 8-bit storage (shared-exp INT8 candidate) | on-read widen to FP16/FP32 | FP32 on sensitive ops | Yes | 容量大，優先壓 storage | INT8 zone3 在敏感路徑優於 E4M3 | 可降 SRAM/面積並提升並行可能 | P0 candidate | 先沿敏感群組擴展驗證 |
| `SCR_K` | K scratch/cache | active working-set heavy | FP32-like scratch | 8-bit storage + shared exponent candidate | FP16/FP32 | FP32 for score state feed | Yes | softmax neighborhood 仍敏感 | interaction risk 在 G4/G3 | local widen 可控，較易硬體化 | P1 | 小範圍 tile A/B compare |
| `SCR_V` | V scratch/cache | active working-set heavy | FP32-like scratch | 8-bit storage + shared exponent candidate | FP16/FP32 | FP32 for context accum | Yes | accum path 不宜全程低位寬 | interaction-driven fragility | storage/compute decoupling 有利面積與並行 | P1 | 驗證 context path 分區策略 |
| `FINAL_SCALAR_BUF` | FinalHead `s_t` staging/readout local state | small scalar buffer | FP32 local staging | keep >=16-bit (FP16/BF16/FP32) | FP16/FP32 | FP32 for reduction accum | Optional | 量小但決策敏感，硬壓8-bit風險高 | FP16-global 穩定；FinalHead 仍特殊路徑 | 高精度例外成本小、收益高 | P0 protect | 先用 FP16 guard，不先壓8-bit |
| LN local states (`mean/var/invstd`) | normalization scalar states | tiny capacity | FP32 local | keep FP16/FP32 | FP16/FP32 | FP32 | No | scalar state 高敏感 | FP16-global 穩定，支持保高精度 | 面積成本小，穩定性收益大 | P0 protect | LN 演算法不改，先保 guard |
| softmax local states (`max/sumexp/reciprocal`) | online softmax states | tiny but high sensitivity | FP32 local | keep FP16/FP32 local | FP16/FP32 | FP32 | No | mask/online semantics 不能壞 | full stress early RED + FP16-global 穩定 | 非主要 SRAM 戰場，不宜冒險 | P0 protect | 維持高精度例外 |
| token-local tiles / FIFOs / local accum | block-local temporary states | medium scratch | mixed local arrays | 8~16 bit by role | kernel-specific mixed | >=16 or FP32 for accum | Yes | 純壓 storage 需避免 accum 崩壞 | interaction failure indicates careful partitioning | 適合做 layout-aware precision 分層 | P1 | 先做 profile 再定點優化 |
| IO staging | transport words | protocol-driven | 32-bit aligned | keep protocol width | N/A | N/A | No | 受介面契約限制 | no direct fragility signal | 不是 SRAM 壓縮主戰場 | P0 keep | 維持不變 |

## 6. Practical Conclusion
- 大頁面主策略優先：`X_WORK/SCR_K/SCR_V` 走 8-bit storage（shared-exp INT8 + widen-on-read）。
- 高敏感例外層：`FINAL_SCALAR_BUF`、LN/softmax local states 優先用 FP16（必要時 FP32）。
- `W_REGION` ternary payload 繼續維持 2-bit packed。
- 可落地分工：
  - INT8/shared-exp 負責容量主戰場
  - FP16 負責小量高敏感 guard layer

## 7. Governance Posture
- ref-only exploration input
- local-only evidence
- not HLS mainline closure
- not Catapult closure
- not SCVerify closure
- policy is draft, not frozen spec
