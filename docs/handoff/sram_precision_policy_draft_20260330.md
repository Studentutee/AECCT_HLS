# SRAM Precision Policy Draft (2026-03-30, updated after INT8 fixed-exp zone explore)

## 1. Draft Positioning
- 這是 task-local 的 architecture/policy 草案，用於指引 ref-only 量化探索。
- 不是 frozen spec，不是 HLS mainline closure，不是 Catapult/SCVerify closure。

## 2. Core Objective
在「SRAM容量優先」前提下，兼顧：
1. 計算便利性（widen/unpack/rescale 複雜度可控）
2. compute-area 與平行化（小位寬單元可提高並行）
3. evaluator quality（trace-reference-aligned BER/FER 不明顯惡化）

## 3. Precision Policy Principles
1. 大容量物件優先降低 **storage bitwidth**。
2. 少量但高敏感 scalar/local states 可維持較高精度。
3. 明確分離：storage / on-read widen / compute / accumulator / write-back。
4. `W_REGION` ternary payload 已是 2-bit packed，不能誤解成「要改成 8-bit」。
5. shared-exp INT8 是候選之一，不代表所有路徑都應強制套用。

## 4. Latest Ref Evidence Linkage (important)
- `G2 + embed_only`（目前最敏感路徑）比較：
  - E4M3 在 4~15 出現 `delta BER>0`, `delta FER>0`。
  - INT8 fixed-exp zone3 在 4~15 為 `delta BER=0`, `delta FER=0`，且無 x_pred/sign flips。
- 代表 shared-exp INT8 不是純理論假設，已有 ref evidence 支撐其可行性（至少在目前敏感路徑）。

## 5. Storage/Compute Policy Table

| Storage / object | Role / semantic meaning | Capacity driver | Current form (repo/spec view) | Proposed storage bitwidth | Proposed compute bitwidth | Proposed accumulator / scalar-state bitwidth | Allow widen-on-read? | Why not lower? / Why can lower? | Fragility evidence linkage | HW convenience / area / parallelism comment | Priority | Validation next step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `W_REGION` ternary payload | native linear quant weight payload | persistent capacity heavy | ternary packed (2-bit class) | Keep packed ternary | integer/ternary MAC path | INT16 (native path) | Yes (decode) | 已比8-bit更省；再改8-bit沒有主要收益 | full stress failure 不指向 W payload | decode 邏輯固定後可高並行，且不增加 SRAM | P0 keep | 維持現狀，僅驗證 unpack throughput |
| `W_REGION` metadata (`inv_s_w`, matrix metadata) | dequant/scale metadata | small persistent | FP32/U32 mixed | 16-bit or mixed (case-by-case) | FP16/FP32 | FP32 for scale chain | Yes | metadata 量小但影響 scale 正確性 | fragility 主訊號在 activation side | 保留可讀性與控制簡單，不追求極限壓縮 | P1 | 按模組盤點 metadata 節省潛力 |
| `X_WORK` | main token×d working set | active working-set heavy | baseline FP32 | 8-bit storage (shared-exp INT8 candidate) | on-read widen to FP16/FP32 | FP32 on sensitive ops | Yes | 容量大，先壓 storage 最有價值 | INT8 zone3 在敏感路徑優於 E4M3 | storage 壓低可減面積/功耗並提升並行可能 | P0 candidate | 先在敏感群組做 shared-exp 驗證再擴 |
| `SCR_K` | K scratch/cache | active working-set heavy | FP32-like scratch | 8-bit storage candidate + shared exponent | FP16/FP32 for compute | FP32 for score local state feed | Yes | 需保留 softmax neighborhood 穩定性 | G4/G3 仍屬 interaction risk | 透過 local widen 控制硬體複雜度 | P1 | 小範圍 tile-driven A/B compare |
| `SCR_V` | V scratch/cache | active working-set heavy | FP32-like scratch | 8-bit storage candidate + shared exponent | FP16/FP32 for compute | FP32 for context accum | Yes | accum path 對精度敏感，不宜全程低位寬 | current evidence shows interaction-driven risk | storage/compute decoupling 有利面積與可擴並行 | P1 | 驗證 context path 的 shared-exp 分區策略 |
| `FINAL_SCALAR_BUF` | FinalHead `s_t` staging/readout local state | small scalar buffer | FP32 local staging | keep >=16-bit (FP16/BF16/FP32) | FP16/FP32 | FP32 for reduction accum | Optional | 量小但決策敏感，硬壓8-bit風險高 | FinalHead partial 僅能保守外推 | 保高精度例外可降低大範圍 rollback 風險 | P0 protect | 維持 partial strategy，不先外擴 |
| LN local states (`mean/var/invstd`) | normalization scalar states | tiny capacity | FP32 local | keep FP32 (or cautious BF16) | FP32 | FP32 | No | scalar state 對穩定性高度敏感 | 不支持先改 LN 演算法 | 高精度 local state 對面積影響小、穩定收益大 | P0 protect | 先完成 attribution，再決定是否降精 |
| softmax local states (`max/sumexp/reciprocal`) | online softmax neighborhood | tiny capacity but high sensitivity | FP32 local | keep FP32 local | FP32 | FP32 | No | mask/online semantics 不能被破壞 | full stress early RED 與此區域交互高度相關 | 省 SRAM 空間有限，不值得冒語意風險 | P0 protect | 維持高精度，僅做局部觀測 |
| token-local tiles / FIFOs / local accum | block-local temporary states | medium scratch | mixed local arrays | 8~16 bit per role | kernel-specific mixed | >=16 or FP32 for accum | Yes | 純壓 storage 需避免 accum 崩壞 | interaction failure suggests careful partitioning | 最適合做 layout-aware precision 分層 | P1 | 先做 profile 再定點優化 |
| IO staging | interface transport words | protocol-driven | 32-bit word aligned | keep protocol width | N/A | N/A | No | IO 受介面契約限制 | no direct fragility signal | 不是主要 SRAM 壓縮戰場 | P0 keep | 維持不變 |

## 6. Practical Conclusion (current draft)
- 最值得先做 8-bit storage 的族群：`X_WORK`, `SCR_K`, `SCR_V`（搭配 shared-exp + widen-on-read）。
- 應先保高精度的例外集合：
  - `FINAL_SCALAR_BUF`
  - LN local scalar states
  - softmax local scalar states
- 已經比 8-bit 更省的類別：`W_REGION` ternary packed payload。
- 目前最可落地策略：
  - 大容量 working-set 走 shared-exp INT8 storage。
  - 少量敏感 scalar states 保高精度。
  - compute/accum 不與 storage 強綁定位寬。

## 7. Governance Posture
- ref-only exploration input
- local-only evidence
- not HLS mainline closure
- not Catapult closure
- not SCVerify closure
- policy is draft, not frozen spec
