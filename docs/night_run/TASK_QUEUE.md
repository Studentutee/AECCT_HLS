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
| NR-BLOCKER-QKV-STAGE-SHELL-001 | done | runner | - | runner.local.p11aj | true | Resolve q-ready/kv-not-prebuilt shell contraction via bounded composed stage shell. | `build/p11aj/p11aj/run.log` contains `Q_READY_KV_NOT_PREBUILT_TO_QKV_SCORES_STAGE PASS` and `Q_READY_KV_NOT_PREBUILT_QKV_SCORES_STAGE_MIGRATION PASS`; `build/p11anb/attnlayer0_boundary_seam_contract/run.log` contains `P11ANB_TRANSFORMER_ATTN_SHELL_Q_READY_KV_NOT_PREBUILT_TO_QKV_SCORES_STAGE PASS`. |
| NR-RUNNER-QKV-STAGE-MINCUT-PREP-001 | done | runner | NR-BLOCKER-QKV-STAGE-SHELL-001 | runner.local.p11aj | true | Deliver bounded composed shell mincut for q-ready/kv-not-prebuilt/score-not-prebuilt bucket. | `src/blocks/TransformerLayer.h` adds `TRANSFORMER_ATTN_COMPAT_SHELL_QKV_SCORES_ONLY` and composed QKV+SCORES dispatch with explicit post->out writeback seam. |
| NR-CHECK-DESIGN-PURITY-ACTIVE-004 | done | checker | - | checker.design_purity | true | Active dispatch precheck for the next partial-bucket audit round. | Local precheck run (`scripts/check_design_purity.ps1`) reports `PASS: check_design_purity`; exit code = 0. |
| NR-RUNNER-NEXT-PARTIAL-BUCKET-AUDIT-005 | done | runner | NR-CHECK-DESIGN-PURITY-ACTIVE-004 | runner.local.p11aj | true | Select next safest partial bucket after q-ready/kv-not-prebuilt QKV+SCORES shell cut. | `build/p11aj/p11aj/run.log` contains `KV_READY_Q_NOT_PREBUILT_SCORE_READY_TO_OUT_STAGE PASS`; `build/p11anb/attnlayer0_boundary_seam_contract/run.log` contains `P11ANB_TRANSFORMER_ATTN_SHELL_KV_READY_Q_NOT_PREBUILT_SCORE_READY_TO_OUT_STAGE PASS`. |
| NR-CHECK-DESIGN-PURITY-ACTIVE-005 | done | checker | - | checker.design_purity | true | Active dispatch precheck for the next partial-bucket audit round after bucket-005 shrink. | Local precheck run (`scripts/check_design_purity.ps1`) reports `PASS: check_design_purity`; exit code = 0. |
| NR-RUNNER-NEXT-PARTIAL-BUCKET-AUDIT-006 | done | runner | NR-CHECK-DESIGN-PURITY-ACTIVE-005 | runner.local.p11aj | true | Select next safest partial bucket after kv-ready/q-not-prebuilt score-ready OUT-stage shrink. | `build/p11aj/p11aj/run.log` contains `KV_READY_Q_NOT_PREBUILT_TO_QKV_SCORES_STAGE PASS`; `build/p11anb/attnlayer0_boundary_seam_contract/run.log` contains `P11ANB_TRANSFORMER_ATTN_SHELL_KV_READY_Q_NOT_PREBUILT_TO_QKV_SCORES_STAGE PASS`. |
| NR-CHECK-DESIGN-PURITY-ACTIVE-006 | done | checker | - | checker.design_purity | true | Active dispatch precheck for the next partial-bucket audit round after bucket-006 shrink. | Local precheck run (`scripts/check_design_purity.ps1`) reports `PASS: check_design_purity`; exit code = 0. |
| NR-RUNNER-NEXT-PARTIAL-BUCKET-AUDIT-007 | done | runner | NR-CHECK-DESIGN-PURITY-ACTIVE-006 | runner.local.p11aj | true | Select next safest partial bucket after kv-ready/q-not-prebuilt QKV+SCORES shrink. | `build/p11aj/p11aj/run.log` contains `QKV_NOT_PREBUILT_SCORE_READY_TO_OUT_STAGE PASS`; `build/p11anb/attnlayer0_boundary_seam_contract/run.log` contains `P11ANB_TRANSFORMER_ATTN_SHELL_QKV_NOT_PREBUILT_SCORE_READY_TO_OUT_STAGE PASS`. |
| NR-CHECK-DESIGN-PURITY-ACTIVE-008 | ready | checker | - | checker.design_purity | true | Active dispatch precheck for the next partial-bucket audit round after bucket-008 shrink. | Task log must contain `PASS: check_design_purity`; exit code = 0. |
| NR-RUNNER-NEXT-PARTIAL-BUCKET-AUDIT-009 | queued | runner | NR-CHECK-DESIGN-PURITY-ACTIVE-008 | runner.local.p11aj | true | Select next safest partial bucket after non-prebuilt `00000` QKV+SCORES shrink. | Must emit case-specific PASS/blocked banner and preserve already-converged bucket behavior. |

## Notes
- Keep one row per executable task.
- Keep completed rows in `TASK_QUEUE_DONE_ARCHIVE.md` (do not delete historical evidence).
- v1 dispatch supports only known runner keys; unknown key is fail-fast.
- v1.1 compile-backed keys currently include `runner.local.p11aj` and `runner.local.p11anb`.
- Do not claim Catapult/SCVerify closure from night-run v1.1 outputs.
