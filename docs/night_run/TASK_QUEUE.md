# TASK_QUEUE

## Purpose
- Provide the executable queue source for night-run dispatch mode v1.
- Keep v1 bounded to local-only checker/runner tasks with explicit evidence links.

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
- `runner`: v1 runner key (resolved by `run_night_pack.ps1`)
- `stop_on_fail`: `true` or `false`
- `objective`: short task purpose
- `acceptance`: local acceptance string for this row

## Queue Table
| task_id | status | lane | depends_on | runner | stop_on_fail | objective | acceptance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NR-CHECK-DESIGN-PURITY-001 | ready | checker | - | checker.design_purity | true | Run design purity checker before any runner dispatch. | Log contains `PASS: check_design_purity`; exit code = 0. |
| NR-RUNNER-INIT-AGENT-STATE-001 | ready | runner | NR-CHECK-DESIGN-PURITY-001 | runner.init_agent_state | true | Execute one local-only runner task after checker passes. | Log contains `PASS: init_agent_state`; exit code = 0. |

## Notes
- Keep one row per executable task.
- v1 dispatch supports only known runner keys; unknown key is fail-fast.
- Do not claim Catapult/SCVerify closure from night-run v1 outputs.
