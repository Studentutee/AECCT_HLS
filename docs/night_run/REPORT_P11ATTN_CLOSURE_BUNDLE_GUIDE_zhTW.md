# REPORT_P11ATTN_CLOSURE_BUNDLE_GUIDE_zhTW

Date: 2026-04-05  
Scope: attention mainline closure bundle guide (reviewer-facing, local-only)

## 這份文件在做什麼
這是 attention mainline closure 的導讀索引。  
目標是讓 reviewer 快速知道：
- 哪份文件回答哪個問題
- 哪些是「真的跑過的證據」
- 哪些是「依 code-path 做的推論」
- 為什麼還有 residual buckets 但不代表主線沒封口

## 建議閱讀順序
1. `REPORT_P11ATTN_MAINLINE_NO_DIRECT_SRAM_CLOSURE_STATEMENT.md`
2. `REPORT_P11OUT1_MAINLINE_CLASSIFICATION_AND_CLOSURE_PLAN.md`
3. `PROJECT_STATUS_zhTW.txt`（看整體定位）
4. `TASK_QUEUE_DONE_ARCHIVE.md`（看任務與證據鏈）

## 文件對照（問題 -> 文件）
1. 問題：attention mainline 現在是否可宣告 no-direct-SRAM fallback（local scope）？
- 文件：`docs/night_run/REPORT_P11ATTN_MAINLINE_NO_DIRECT_SRAM_CLOSURE_STATEMENT.md`
- 性質：實跑證據 + code-path inference（明確標示 not Catapult / not SCVerify）

2. 問題：out=1 residual buckets 為何沒有繼續 shrink？
- 文件：`docs/night_run/REPORT_P11OUT1_MAINLINE_CLASSIFICATION_AND_CLOSURE_PLAN.md`
- 性質：compile-backed audit + code-path inference（Class 1/2/3）

3. 問題：專案主狀態如何記錄這個 closure？
- 文件：`docs/process/PROJECT_STATUS_zhTW.txt`
- 性質：治理與狀態同步（非新證據產生點）

4. 問題：這些結論有沒有在 night-run 歷史裡留痕？
- 文件：`docs/night_run/TASK_QUEUE_DONE_ARCHIVE.md`
- 性質：任務履歷與證據連結

## 證據 vs 推論邊界（避免 overclaim）
### 實跑證據
- `p11aj` 日誌中的 `FULL_LOOP_MAINLINE_PATH_TAKEN PASS` / `FULL_LOOP_FALLBACK_NOT_TAKEN PASS`
- `p11anb` 日誌中的 selector/contract pass banners
- `check_design_purity` / `check_repo_hygiene` PASS

### 路徑推論
- `Top.h` 中 `out_prebuilt` 是由 AE/AF mainline 成功鏈派生
- `out=1` residual buckets 的 Class 2/3 定位（fallback/safety-net 或 likely unreachable）

## 為什麼 residual out=1 buckets 不等於主線未完成
- 目前沒有 compile-backed 強證據顯示 residual `out=1` unresolved buckets 屬於「真實 mainline 生產路徑」。
- 它們在現況主要屬於防禦面（safety-net）或不變量下不可達路徑。
- 所以它們應該被當成 fallback surface 管理，不應直接被解讀成 mainline closure 缺口。

## Scope 邊界（再次確認）
- local-only
- compile-first / evidence-first
- not Catapult closure
- not SCVerify closure
