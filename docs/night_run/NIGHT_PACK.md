# NIGHT_PACK

## Purpose
- Define the minimal nightly automation dispatch v1.1 for multi-round local task execution.
- Standardize fixed inputs and fixed evidence outputs for reproducible handoff.
- Keep scope local-only unless explicitly expanded by a dedicated task.

## Fixed Inputs
- `docs/night_run/TASK_QUEUE.md`
- `docs/night_run/ACCEPTANCE_PACK.md`

## Canonical Entry
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_night_pack.ps1 -BuildDir build/night_run -Smoke`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_night_pack.ps1 -BuildDir build/night_run`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_night_pack.ps1 -BuildDir build/night_run -MaxReadyTasks 3`

## Queue Dispatch v1.1
- `run_night_pack.ps1` reads `TASK_QUEUE.md` and picks rows with `status=ready`.
- v1.1 supports bounded task mapping only:
  - `checker.design_purity`
  - `runner.init_agent_state`
  - `runner.local.p11aj` (compile-backed local runner)
- Dispatch order follows queue row order and honors `depends_on` and `stop_on_fail`.
- Per-task outputs are written to `build/night_run/<run_id>/tasks/<task_id>/`.
- Compile-backed runner may require `VsDevCmd`/`cl`; task summary must record toolchain note and evidence source.

## Required Output Contract
Each run must create a run folder under `build/night_run/<run_id>/` and include:
- `NIGHT_PACK_SUMMARY.txt`
- `NIGHT_PACK_EXECUTION.md`
- `NIGHT_PACK_VERDICT.json`
- `NIGHT_PACK_MANIFEST.txt`
- `ACCEPTANCE_PACK_FILLED.md`
- `tasks/<task_id>/task_execution.log`
- `tasks/<task_id>/task_summary.txt`

## Governance Posture
- Local evidence is local-only evidence.
- Closure posture must be explicit in outputs:
  - `not Catapult closure`
  - `not SCVerify closure`
- No production-readiness claim from skeleton-only runs.

## Design Boundary
- Skeleton setup must not change attention/design mainline code by default.
- Design changes are allowed only when explicitly queued as design tasks.

## Smoke Gate
- A minimal smoke run is valid when:
  - Script exits with code 0.
  - Stdout includes `PASS: run_night_pack`.
  - Required output contract files exist in the generated run folder.

## Dispatch Gate
- A minimal dispatch run is valid when:
  - At least one `checker` task and one `runner` task are executed from queue.
  - Each executed task has local evidence under `tasks/<task_id>/`.
  - `NIGHT_PACK_VERDICT.json` includes per-task PASS/FAIL and overall verdict.
