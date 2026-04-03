# NIGHT_PACK

## Purpose
- Define the minimal nightly automation skeleton for multi-round local task execution.
- Standardize fixed inputs and fixed evidence outputs for reproducible handoff.
- Keep scope local-only unless explicitly expanded by a dedicated task.

## Fixed Inputs
- `docs/night_run/TASK_QUEUE.md`
- `docs/night_run/ACCEPTANCE_PACK.md`

## Canonical Entry
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_night_pack.ps1 -BuildDir build/night_run -Smoke`

## Required Output Contract
Each run must create a run folder under `build/night_run/<run_id>/` and include:
- `NIGHT_PACK_SUMMARY.txt`
- `NIGHT_PACK_EXECUTION.md`
- `NIGHT_PACK_VERDICT.json`
- `NIGHT_PACK_MANIFEST.txt`
- `ACCEPTANCE_PACK_FILLED.md`

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
