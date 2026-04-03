# AGENTS.md

## Purpose
This file defines permanent, repo-shared rules for AI agents working in AECCT_HLS.
Do not store session-only notes, task queues, blocker diaries, or machine-local paths here.

## Hardware/HLS-First Mindset
- Treat this repository as a C++ HLS and hardware-oriented project, not software-only code.
- Preserve synthesizable boundaries and deterministic dataflow assumptions.
- Avoid architecture rewrites unless explicitly requested.

## SRAM and Ownership Boundary
- Top is the only production shared-SRAM owner.
- Sub-blocks must not claim production shared-memory ownership.
- If a local helper uses temporary buffers or wrappers, label it explicitly as `local-only`.

## Claim Posture and No Overclaim
- Local evidence must be reported as local evidence.
- Always state closure posture explicitly when applicable:
  - `not Catapult closure`
  - `not SCVerify closure`
- Do not claim closure or production readiness without matching tool evidence.

## Evidence-First Completion Format
Completion reports must include all of the following:
1. Summary
2. Exact files changed
3. Exact commands run
4. Actual execution evidence / log excerpt
5. Repo-tracked artifacts
6. Local-only working-memory artifacts
7. Governance posture
8. Residual risks
9. Recommended next step

## Mixed-Payload Helper Channel Rule
- If payload classes differ in consume timing, consume rate, or phase boundary, split channels by payload class by default.
- Only share one channel when lockstep production/consumption and bounded-FIFO safety are both proven.

## Design-Side Readability Rules
- Design-side comments should be ASCII and English.
- Use stable loop labels for review and HLS schedule traceability (example: `LOOP_NAME: for (...) { ... }`).

## Minimum Post-Change Checks
- Always run `git status --short`.
- Run `powershell -ExecutionPolicy Bypass -File scripts/check_design_purity.ps1`.
- Run `powershell -ExecutionPolicy Bypass -File scripts/check_repo_hygiene.ps1 -Phase pre`.
- If agent tooling files changed, run `powershell -ExecutionPolicy Bypass -File scripts/check_agent_tooling.ps1`.
- Run the smallest relevant task-local checker/runner for the touched scope.

## Night-Run Automation Skeleton
- The fixed night-run skeleton assets are:
  - `docs/night_run/NIGHT_PACK.md`
  - `docs/night_run/TASK_QUEUE.md`
  - `docs/night_run/ACCEPTANCE_PACK.md`
  - `scripts/local/run_night_pack.ps1`
- Skeleton scope is orchestration and evidence scaffolding, not direct design-mainline advancement.
- Night-run skeleton outputs must keep closure posture explicit:
  - `not Catapult closure`
  - `not SCVerify closure`
- Night-run smoke validation must emit machine-readable and human-readable evidence under `build/night_run/`.
- Do not edit attention/design code as part of skeleton-only setup unless explicitly queued as a design task.
