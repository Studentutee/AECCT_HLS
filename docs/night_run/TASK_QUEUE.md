# TASK_QUEUE

## Purpose
- Provide the executable queue source for night-run dispatch mode v1.1.
- Keep v1.1 bounded to local-only checker/runner tasks with explicit evidence links.
- Keep this file focused on **active** items only; completed history is archived in
  `docs/night_run/TASK_QUEUE_DONE_ARCHIVE.md`.

## Status Vocabulary
- `queued`
- `ready`
- `running`
- `blocked`
- `done`
- `dropped`

## Lane Vocabulary
- `checker`
- `runner`

## Executable Columns
- `task_id`: unique identifier in one queue file
- `status`: only `ready` rows are dispatch candidates
- `lane`: `checker` or `runner`
- `depends_on`: `-` or comma-separated task ids
- `runner`: v1.1 runner key (resolved by `run_night_pack.ps1`)
- `stop_on_fail`: `true` or `false`
- `objective`: short task purpose
- `acceptance`: local acceptance string for this row

## Queue Table
| task_id | status | lane | depends_on | runner | stop_on_fail | objective | acceptance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NR-CHECK-DESIGN-PURITY-ACTIVE-003 | done | checker | - | checker.design_purity | true | Active dispatch precheck for the next queue round after archive rollover. | Run 20260404_170432 task log contains `PASS: check_design_purity`; exit code = 0. |
| NR-RUNNER-NEXT-PARTIAL-BUCKET-AUDIT-004 | done | runner | NR-CHECK-DESIGN-PURITY-ACTIVE-003 | runner.local.p11aj | true | Continue next safest partial-bucket audit after q-ready/kv-not-prebuilt blocker verification. | Run 20260404_170432 runner `run.log` contains `Q_READY_KV_NOT_PREBUILT_SCORE_READY_TO_OUT_STAGE PASS` and `Q_READY_KV_NOT_PREBUILT_SCORE_READY_OUT_STAGE_MIGRATION PASS`. |
| NR-BLOCKER-QKV-STAGE-SHELL-001 | blocked | runner | - | runner.local.p11aj | true | Hold q-ready/kv-not-prebuilt shell contraction until bounded KV-materialization-compatible shell contract exists. | Blocker package tracked at `docs/night_run/BLOCKER_P11BE_QKV_STAGE_FEASIBILITY.md` with compile-backed p11aj/p11anb evidence. |
| NR-RUNNER-QKV-STAGE-MINCUT-PREP-001 | queued | runner | NR-BLOCKER-QKV-STAGE-SHELL-001 | runner.local.p11aj | true | Prepare bounded prework for next shell candidate after blocker clearance (no broad refactor). | Must preserve Top-owned shared-SRAM semantics and show compile-backed non-regression in p11aj/p11anb before selector remap. |
| NR-CHECK-DESIGN-PURITY-ACTIVE-004 | ready | checker | - | checker.design_purity | true | Active dispatch precheck for the next partial-bucket audit round. | Task log must contain `PASS: check_design_purity`; exit code = 0. |
| NR-RUNNER-NEXT-PARTIAL-BUCKET-AUDIT-005 | queued | runner | NR-CHECK-DESIGN-PURITY-ACTIVE-004 | runner.local.p11aj | true | Select next safest partial bucket after score-ready OUT-stage cut and audit via compile-backed truth-table. | Must emit case-specific PASS banner and preserve fully-prebuilt/scores-only/non-selected-bucket behavior. |

## Notes
- Keep one row per executable task.
- Keep completed rows in `TASK_QUEUE_DONE_ARCHIVE.md` (do not delete historical evidence).
- v1 dispatch supports only known runner keys; unknown key is fail-fast.
- v1.1 compile-backed keys currently include `runner.local.p11aj` and `runner.local.p11anb`.
- Do not claim Catapult/SCVerify closure from night-run v1.1 outputs.
