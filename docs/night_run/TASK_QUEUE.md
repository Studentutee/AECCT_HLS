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
| NR-CHECK-DESIGN-PURITY-ACTIVE-010 | ready | checker | - | checker.design_purity | true | Active dispatch precheck for the next partial-bucket audit round after payload-enabled `10101` OUT-only shrink. | Task log must contain `PASS: check_design_purity`; exit code = 0. |
| NR-RUNNER-NEXT-PARTIAL-BUCKET-AUDIT-011 | queued | runner | NR-CHECK-DESIGN-PURITY-ACTIVE-010 | runner.local.p11aj | true | Select next safest partial bucket after payload-enabled `10101` OUT-only shrink. | Must emit case-specific PASS/blocked banner and preserve already-converged bucket behavior. |

## Notes
- Keep one row per executable task.
- Keep completed rows in `TASK_QUEUE_DONE_ARCHIVE.md` (do not delete historical evidence).
- v1 dispatch supports only known runner keys; unknown key is fail-fast.
- v1.1 compile-backed keys currently include `runner.local.p11aj` and `runner.local.p11anb`.
- Do not claim Catapult/SCVerify closure from night-run v1.1 outputs.
