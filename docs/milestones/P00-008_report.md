# P00-008 Report — Ternary Weight SRAM Storage Spec Update (docs-only)

## Goal / Scope
- 任務目標：將 ternary weight 在 SRAM 的存放規格正式補入現有 docs，並收斂為可審核的凍結條文。
- 任務範圍：僅 docs 更新；不修改 design/src/include/tb/scripts/gen。
- 任務性質：documentation closure，不含 implementation closure。

## Docs Baseline Used
- 本輪只使用本次 repo zip 內現有 docs 作為唯一基線。
- 未引用已刪除的舊外部規格檔。
- 目標文件存在性檢查結果：
- `docs/spec/AECCT_HLS_Spec_v12.1_zhTW.txt`：found
- `docs/architecture/AECCT_HLS_Architecture_Guide_v12.1_zhTW.txt`：found
- `docs/architecture/AECCT_SramMap_alias_freepoint_rules_v12.1_zhTW.txt`：found
- `docs/architecture/AECCT_module_interface_rules_v12.1_zhTW.txt`：found
- `docs/archive/spells/咒語v12.1_zhTW.txt`：found
- `docs/process/AECCT_v12_M0-M24_plan_zhTW.txt`：found

## Files Updated
- `docs/spec/AECCT_HLS_Spec_v12.1_zhTW.txt`
- `docs/architecture/AECCT_HLS_Architecture_Guide_v12.1_zhTW.txt`
- `docs/architecture/AECCT_SramMap_alias_freepoint_rules_v12.1_zhTW.txt`
- `docs/architecture/AECCT_module_interface_rules_v12.1_zhTW.txt`
- `docs/archive/spells/咒語v12.1_zhTW.txt`
- `docs/process/AECCT_v12_M0-M24_plan_zhTW.txt`

## Reconciliation Result (Current Run)
- 本次 reconciliation 以目前工作目錄作為 zip 基線。
- 結果為 content no-op：六份目標 docs 的 ternary SRAM 凍結條文已存在，未再追加新的正文條文。
- 依治理要求，P00-008 交付物仍保留並完成檢查：
  - `P00-008_report.md`
  - `file_manifest.txt`
  - `diff.patch`
  - `verdict.txt`

## Per-File Changes
- `AECCT_HLS_Spec_v12.1_zhTW.txt`
- 新增 ternary weight SRAM encoding 條文（codebook、2-bit packing、32-bit=16 weights、LSB-first）。
- 新增 `last_word_valid_count` 語意（1..16；滿載=16）與 tail padding=0。
- 明確化 per-matrix `inv_s_w` 為主流程 metadata 粒度；`s_w` 保留數學語意。
- 補 `LOAD_W/checker`：遇 `10`（illegal）必須回報格式錯誤。
- 補 matrix record 欄位，並明確為 logical section format，不代表手動固定 physical SRAM base offsets。
- 明確化 `W[out][in]`（數學）與 `in contiguous`（實體線性化）分離。

- `AECCT_HLS_Architecture_Guide_v12.1_zhTW.txt`
- 補 W_REGION 中 ternary packed payload + per-matrix `inv_s_w` + matrix record metadata 的使用語意。
- 補 decode 策略：先 tile-unpack/decode 到 local buffer，再消費 `{-1,0,+1}`。
- 補 per-matrix `inv_s_w` 讀取時機（matrix section 入口）。
- 明確 `W[out][in]` 為數學表示，`in contiguous` 僅為實體存放順序。

- `AECCT_SramMap_alias_freepoint_rules_v12.1_zhTW.txt`
- 補 W_REGION 包含 ternary packed payload、per-matrix `inv_s_w`、matrix record/section metadata。
- 強化 persistent/no-alias 語意：W_REGION 不得與 runtime scratch alias。
- 補 matrix record 邏輯層定位：只凍結 logical section，不手工固定 physical base offset。

- `AECCT_module_interface_rules_v12.1_zhTW.txt`
- 補 interface contract：ternary packed payload 格式唯一來源為 `W_REGION + WeightStreamOrder`。
- 補 block 限制：不得自訂另一套 ternary encoding。
- 補 `LOAD_W/parameter ingest` 必須支援 packed payload + per-matrix `inv_s_w` + metadata。

- `咒語v12.1_zhTW.txt`
- 新增 v12.1 freeze 硬釘子：codebook、16/word、LSB-first、`last_word_valid_count`（1..16；滿載=16）、padding=0、per-matrix `inv_s_w`、illegal `10` 必報錯、tile-unpack decode。
- 補 matrix record 為 logical section format，不等於手動固定 physical base offset。

- `AECCT_v12_M0-M24_plan_zhTW.txt`
- 在 M4/M16/M18 做最小補充，凍結 ternary packed SRAM format 與 per-matrix `inv_s_w` 對齊要求。
- 補 `LOAD_W / WeightStreamOrder / generator` 需與凍結格式一致的里程碑條文。

## Frozen Decisions Applied
- codebook：`00=0`、`01=+1`、`11=-1`、`10=reserved/illegal`。
- 2-bit packed ternary；32-bit word=16 weights；LSB-first。
- `last_word_valid_count`：合法範圍 `1..16`；最後一個 word 滿載時為 `16`。
- tail padding bits 一律為 `0`。
- 主流程 metadata 粒度：per-matrix `inv_s_w`；`s_w` 保留數學語意。
- `W[out][in]` 是數學表示；`in contiguous` 僅為實體線性化/存放順序。
- 遇 `10` illegal code：`LOAD_W / parameter ingest / checker` 必須報格式錯誤。
- matrix record 是 logical section format，不代表手動固定 physical base offsets。

## Validation Method
- `rg` 僅作快速篩檢（fast screening），用於定位候選段落。
- 最終判定採人工語意核對（semantic/manual），逐檔確認：
  - codebook、packing、`last_word_valid_count`（1..16，滿載=16）、padding=0
  - per-matrix `inv_s_w`
  - illegal `10` 必須報格式錯誤
  - matrix record 為 logical section format（非手動固定 physical base offset）
  - `W[out][in]`（數學）與 `in contiguous`（實體線性化）分離
  - tile-unpack/decode 建議

## Diff Policy / Tooling
- 本次為 content no-op reconciliation；`diff.patch` 採空檔策略（無額外 patch 內容）。
- 差異工具優先使用 `git diff`；本次環境可用，未觸發 file-based fallback。
- 若未來遇到 `git diff` 不可用，將改用檔案式比對並在報告明確記錄 fallback 與限制。

## Files Not Found in zip/docs
- none

## Verified / Not Verified
- Verified
- 6 份目標文件均存在，且已就地更新。
- 關鍵規則已落入 spec/architecture/srammap/interface/spells/plan 六份文件。
- 變更範圍維持 docs-only。

- Not Verified
- 本輪未做 code/generator/header/tb 實作驗證（依任務範圍排除）。
- 本輪未新增或驗證 runtime 數值正確性測試（依任務範圍排除）。

## Final Conclusion
- P00-008 已完成 docs-only 的 ternary weight SRAM storage 規格更新與凍結收斂。
- 本次結論僅代表文件收斂完成，不代表實作已完成。

## README Auto Policy
- This is a docs-only closure step and therefore does not trigger README Auto update.

## Recommended Next Step
- 以本輪凍結文件為單一來源，另立後續任務執行 `WeightStreamOrder / loader / checker` 的實作對齊與回歸驗證。
