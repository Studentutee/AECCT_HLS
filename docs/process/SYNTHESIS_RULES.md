# SYNTHESIS_RULES

## Purpose
- Define a local-only synthesis-safe gate for pre-synth hygiene checks and regression entry.
- Keep accepted P00-011M/P00-011N behavior stable while hardening structural checks.
- Clarify fail-fast policy and warning-summary policy.

## Scope
- Local smoke and local static checks only.
- Catapult and SCVerify are deferred unless explicitly requested.
- No public signature change, no Top contract change, no block-graph redesign.

## Pre-Synth Gate (Fail-Fast on Structural Violations)
- `scripts/check_design_purity.ps1`
- `scripts/check_interface_lock.ps1`
- `scripts/check_macro_hygiene.ps1`

If any script returns non-zero:
- Gate fails immediately.
- One-shot script exits non-zero.

## Warning Policy
- Warnings from build logs are summarized only.
- Warning summary must not change the exit code.
- Only structural violations are blockers.

## Readability / Architecture Visibility Policy
- 對 design-side synthesis path，review 可讀性屬正式治理項，不再視為純美化。
- design-side `.h/.cpp` 註解以 ASCII/英文為主；中文導讀請放 repo-tracked docs / report / reviewer guide，避免 Catapult/editor/log 亂碼。
- 新增或修改的 design code，應補最小 intent comment：block 角色、I/O 資料流、ownership / fallback / write-back boundary。
- design-side `for` loop 原則上需使用穩定 label（`NAME: for (...) {}`）；特別是會在 Catapult Architecture / Schedule GUI 中觀察的 loop，不應省略。
- 巢狀 loop label 不得重名，應反映 phase / row / col / mac / load / store 等角色。
- 本規則目前先作為 reviewer-ready handoff 要求；若後續新增自動 checker，才再升級為 fail-fast pre-synth gate。

## Macro Hygiene Policy
- Approved local macros:
  - `AECCT_LOCAL_P11M_WQ_SPLIT_TOP_ENABLE`
  - `AECCT_LOCAL_P11N_WK_WV_SPLIT_TOP_ENABLE`
- Local macros are allowed only in approved source/TB/script locations.
- Baseline build commands must not carry local macros.
- Macro build commands must use the approved `/D` form only.

## Interface Lock Policy
- Interface lock requires file existence and fixed anchor presence.
- Anchor matching is whitespace-tolerant.
- No whole-file hash lock is used in this gate version.

## One-Shot Entry
- Canonical command:
  - `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11l_local_regression.ps1 -BuildDir build\p11n *> build\p11n\run_p11o_regression.log`
- PASS criteria:
  - All three pre-synth checks print `PASS` banners.
  - Legacy regression coverage (`p11j/p11k/p11l_b/p11l_c/p11m`) passes.
  - Family dual-binary coverage (`p11n baseline/macro`) passes.
  - Final line includes `PASS: run_p11l_local_regression`.
