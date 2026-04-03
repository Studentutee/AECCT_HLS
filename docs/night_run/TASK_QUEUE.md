# TASK_QUEUE

## Purpose
- Provide a stable queue format for nightly multi-round local execution.
- Preserve clear ownership and governance posture per task row.

## Status Vocabulary
- `queued`
- `ready`
- `running`
- `blocked`
- `done`
- `dropped`

## Queue Table
| task_id | status | owner | scope | objective | acceptance |
| --- | --- | --- | --- | --- | --- |
| NR-SKELETON-SMOKE-001 | ready | local-agent | local-only | Verify skeleton can emit required evidence pack. | `PASS: run_night_pack` and required artifacts exist. |

## Notes
- Keep one row per task.
- Use explicit acceptance text, avoid implicit completion claims.
- Do not mark closure beyond available tool evidence.
