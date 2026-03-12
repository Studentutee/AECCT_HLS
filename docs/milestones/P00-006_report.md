# P00-006 Report

## Goal / Scope
- 目標: 完成 repo hygiene triage 與 waiver baseline 初版，建立可審核治理基線。
- 範圍: 僅治理盤點與證據整理，不修改 P00-005 技術內容，不擴大到功能開發或架構重構。
- 範圍限制:
  - 不修改 `include/`、`src/`、`design/`、`tb/` 技術 contract。
  - 不做 repo-wide cleanup campaign。

## What Was Executed
- `python scripts/check_repo_hygiene.py --repo-root .`
- `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_gates.ps1`
- 補充證據命令:
  - `.gitignore` pattern 檢查 (`rg`, `Select-String`)
  - `git check-ignore -v` (ignored 路徑佐證)
  - `git ls-files --error-unmatch <path>` (tracked 狀態)
  - `git log --oneline -n 3 -- <path>` (歷史來源)
  - `git status --short --untracked-files=all | rg "^(.. )?(include/|src/|design/|tb/)" -N` (技術檔 scope lock 檢查)
  - `Get-Content docs/milestones/P00-005_artifacts/build.log | Select-String ...` (跨里程碑 pre-existing 佐證)
- 證據輸出:
  - `docs/milestones/P00-006_artifacts/hygiene.log`
  - `docs/milestones/P00-006_artifacts/supporting_checks.log`

## Hygiene Findings Summary
- `check_repo_hygiene`: FAIL
  - `.gitignore missing reports/`
  - `utf8-bom found: .vs/AECCT_HLS.slnx/v18/HierarchyCache.v1.txt`
  - `archive in repo: AECCT_ac_ref/AECCT_ac_ref.zip`
  - `utf8-bom found: AECCT_ac_ref/include/RefModel.h`
  - `utf8-bom found: AECCT_ac_ref/include/SoftmaxApprox.h`
  - `utf8-bom found: AECCT_ac_ref/src/RefModel.cpp`
  - `utf8-bom found: tools/check_ref_approx_rules.py`
  - `utf8-bom found: tools/compare_checkpoints.py`
  - `utf8-bom found: tools/gen_ref_lut.py`
  - `utf8-bom found: tools/run_algorithm_ref_step0.py`
- Re-run (after P00-006 docs creation): fail set unchanged，未新增本輪文件導致的新 hygiene finding。
- `run_gates.ps1`: aggregate runner，序列執行 individual checks；非獨立額外 gate 規則。
  - `check_design_purity`: PASS
  - `check_interface_lock`: PASS
  - `check_repo_hygiene`: FAIL

## Detailed Classification Policy
- Source classification:
  - `pre-existing`: 可由 `hygiene.log` + `supporting_checks.log` + git history 證明在 P00-006 前即存在。
  - `introduced-by-recent-work`: 問題路徑屬本輪新增/修改且無 pre-existing 證據。
  - `uncertain`: 證據不足，無法安全歸因。
- Action classification:
  - `cleanup-now`: 低風險、機械性、可小範圍修補。
  - `waiver`: 歷史遺留/成本較高/需治理核准後再處理。
  - `defer`: 目前資訊不足或需先定策略再動手。
- 額外硬規則:
  - 凡標 `pre-existing`，至少附一個具體證據來源 (log 或 git history)。

## Findings Table
| ID | Finding | Source | Action | Evidence |
|---|---|---|---|---|
| HYG-001 | `.gitignore missing reports/` | pre-existing | cleanup-now | `hygiene.log` (check_repo_hygiene FAIL); `supporting_checks.log` (`MISSING reports/`); `P00-005_artifacts/build.log` 同項目 |
| HYG-002 | `utf8-bom found: .vs/.../HierarchyCache.v1.txt` | pre-existing | waiver | `hygiene.log`；`supporting_checks.log` (`git check-ignore -v` 顯示 `.vs/` ignored)；`P00-005_artifacts/build.log` 同項目 |
| HYG-003 | `archive in repo: AECCT_ac_ref/AECCT_ac_ref.zip` | pre-existing | waiver | `hygiene.log`；`supporting_checks.log` (`git check-ignore -v` 顯示 `*.zip` ignored)；`P00-005_artifacts/build.log` 同項目 |
| HYG-004 | `utf8-bom found: AECCT_ac_ref/include/RefModel.h` | pre-existing | cleanup-now | `hygiene.log`；`supporting_checks.log` (`git ls-files` + `git log`)；`P00-005_artifacts/build.log` 同項目 |
| HYG-005 | `utf8-bom found: AECCT_ac_ref/include/SoftmaxApprox.h` | pre-existing | cleanup-now | `hygiene.log`；`supporting_checks.log` (`git ls-files` + `git log`)；`P00-005_artifacts/build.log` 同項目 |
| HYG-006 | `utf8-bom found: AECCT_ac_ref/src/RefModel.cpp` | pre-existing | cleanup-now | `hygiene.log`；`supporting_checks.log` (`git ls-files` + `git log`)；`P00-005_artifacts/build.log` 同項目 |
| HYG-007 | `utf8-bom found: tools/check_ref_approx_rules.py` | pre-existing | cleanup-now | `hygiene.log`；`supporting_checks.log` (`git ls-files` + `git log`)；`P00-005_artifacts/build.log` 同項目 |
| HYG-008 | `utf8-bom found: tools/compare_checkpoints.py` | pre-existing | cleanup-now | `hygiene.log`；`supporting_checks.log` (`git ls-files` + `git log`)；`P00-005_artifacts/build.log` 同項目 |
| HYG-009 | `utf8-bom found: tools/gen_ref_lut.py` | pre-existing | cleanup-now | `hygiene.log`；`supporting_checks.log` (`git ls-files` + `git log`)；`P00-005_artifacts/build.log` 同項目 |
| HYG-010 | `utf8-bom found: tools/run_algorithm_ref_step0.py` | pre-existing | cleanup-now | `hygiene.log`；`supporting_checks.log` (`git ls-files` + `git log`)；`P00-005_artifacts/build.log` 同項目 |
| HYG-011 | checker 行為: `check_repo_hygiene.py` 目前掃描 gitignored 路徑 | pre-existing | defer | `scripts/check_repo_hygiene.py` (`repo.rglob("*")` 無 ignore 過濾) + `supporting_checks.log` (`git check-ignore -v`) |

## Cleanup-Now Candidates
- HYG-001: `.gitignore` 補 `reports/`。
- HYG-004 ~ HYG-010: 指定檔案做 UTF-8 no-BOM 機械轉檔。

## Waiver Candidates
- HYG-002: `.vs` IDE cache BOM。
- HYG-003: `AECCT_ac_ref/AECCT_ac_ref.zip` 歷史壓縮檔。
- 以上已納入 `repo_hygiene_waiver_baseline.txt` 初版。

## Deferred Items
- HYG-011: 是否將 `check_repo_hygiene.py` 調整為尊重 `.gitignore`/tracked-only 需先定政策，再進行腳本修補。

## Modified Files
- `docs/milestones/P00-006_report.md` (new file)
- `docs/milestones/P00-006_artifacts/file_manifest.txt` (new file)
- `docs/milestones/P00-006_artifacts/verdict.txt` (new file)
- `docs/milestones/P00-006_artifacts/repo_hygiene_triage.txt` (new file)
- `docs/milestones/P00-006_artifacts/repo_hygiene_waiver_baseline.txt` (new file)
- `docs/milestones/P00-006_artifacts/hygiene.log` (new file)
- `docs/milestones/P00-006_artifacts/supporting_checks.log` (new file)
- `docs/milestones/P00-006_artifacts/diff.patch` (new file)

## Verified / Not Verified
- Verified:
  - hygiene triage 與雙維度分類完成。
  - waiver baseline 初版建立完成。
  - 每個 `pre-existing` 項目皆附具體證據來源。
  - `include/`、`src/`、`design/`、`tb/` 無本輪改動。
- Not verified:
  - 未執行 cleanup patch，repo hygiene gate 仍未關閉。
  - 未對 `check_repo_hygiene.py` 行為做策略性修訂。

## Governance Conclusion
- Technical direction: aligned (治理證據導向、無擴 scope)。
- Governance closure: closed for P00-006 triage deliverable only。
- Gate closure: not closed (`check_repo_hygiene` 仍 FAIL)。

## README Auto Handling
- 本輪未更新 `README.md` Auto。
- 理由: 依 governance 慣例，Auto 同步主要在 `include/`、`design/`、`tb/`、`scripts/` 改動時強制；P00-006 為 docs-only 治理盤點，不變更技術流程指令。

## Recommended Next Step
1. 開新小範圍 cleanup 任務，先處理 HYG-001 與 HYG-004~HYG-010。
2. 對 HYG-002/HYG-003 走 waiver 審批或替代方案決策。
3. 決策 HYG-011 後，再決定是否更新 hygiene checker 規則。
