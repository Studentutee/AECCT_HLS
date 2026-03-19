# READABILITY_GOVERNANCE_ADDENDUM_zhTW
Date: 2026-03-19

## 目的
把「design-side code 幾乎沒註解、Catapult 介面中 loop 無法辨識」這兩個 review 痛點，正式整理成可納入治理的更新提案。

## 本次建議納入治理的核心規則
1. design-side `.h/.cpp` 註解以 ASCII/英文為主，避免 Catapult / compiler / log / diff viewer 亂碼。
2. 中文解釋集中放在 repo-tracked docs、milestone report、reviewer guide / sidecar，不強塞進 HLS-sensitive design code。
3. 新增或修改 design block 時，至少補：
   - file / block intent
   - input -> intermediate -> output 資料流
   - ownership / fallback / bypass / write-back boundary
4. design-side `for` loop 原則上都要命名，使用 `LABEL_NAME: for (...) {}`。
5. 巢狀 loop 必須各自有唯一 label，名稱要能反映 row / col / mac / load / store / phase 角色。
6. reviewer-ready handoff 除執行證據外，建議再附 reviewer-facing sidecar，協助快速 review。

## 建議更新的文件
- `Catapult_C++_CodeGen_Guide_for_Codex_v3.1.txt`
  - 加入 Catapult-safe readability 規則
  - 加入 loop label 規範
  - 把 readability metadata 升級為產碼硬性規格
- `SYNTHESIS_RULES.md`
  - 加入 readability / Architecture visibility policy
  - 明確說明此規則目前先作 reviewer-ready handoff 要求，後續可再升級為自動 checker gate
- `EVIDENCE_BUNDLE_RULES.md`
  - 補 reviewer-facing sidecar 建議格式，不追溯要求既有歷史 bundle 重補
- `PROJECT_STATUS_zhTW.txt`
  - 把 readability governance 列入 open item 與 next recommended focus

## 不建議直接寫進 design code 的內容
- 大量中文行內註解
- 每行都註解
- 只重述語法表面的註解（例如 `// loop`、`// temp`）

## 建議 loop label 命名風格
- phase / 流程：`LOAD_W_LOOP`、`READ_MEM_LOOP`、`WRITEBACK_LOOP`
- 資料維度：`ROW_LOOP`、`COL_LOOP`、`HEAD_LOOP`、`TOKEN_LOOP`
- 運算角色：`MAC_LOOP`、`REDUCE_LOOP`、`PACK_LOOP`、`UNPACK_LOOP`

## 對未來 code review 的實際效果
- 在 Catapult Architecture / Schedule GUI 內更容易辨識哪個 loop 對應哪個 phase。
- reviewer 不用先硬啃原始碼，可以先看中文 sidecar 理解資料流。
- 有助於把「工具友善」與「人類可讀」分層，不互相傷害。
