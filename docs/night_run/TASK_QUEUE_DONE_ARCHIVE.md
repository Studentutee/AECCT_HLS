# TASK_QUEUE_DONE_ARCHIVE

## Purpose
- Preserve completed queue history outside active dispatch queue.
- Keep machine-readable active queue slim while retaining evidence links.

## Archive Columns
- `task_id`
- `completed_at` (run_id/date)
- `lane`
- `runner`
- `objective`
- `evidence`
- `result`

## Done Archive Table
| task_id | completed_at | lane | runner | objective | evidence | result |
| --- | --- | --- | --- | --- | --- | --- |
| NR-CHECK-DESIGN-PURITY-001 | run_id=20260404_125258 | checker | checker.design_purity | Run design purity checker before any runner dispatch. | `build/night_run/20260404_125258/tasks/NR-CHECK-DESIGN-PURITY-001/task_execution.log` contains `PASS: check_design_purity`. | PASS |
| NR-RUNNER-INIT-AGENT-STATE-001 | run_id=20260404_125258 | runner | runner.init_agent_state | Execute one local-only runner task after checker passes. | `build/night_run/20260404_125258/tasks/NR-RUNNER-INIT-AGENT-STATE-001/task_execution.log` contains `PASS: init_agent_state`. | PASS |
| NR-RUNNER-LOCAL-P11AJ-001 | run_id=20260404_125258 | runner | runner.local.p11aj | Execute first compile-backed local runner in night-run queue. | `build/night_run/20260404_125258/tasks/NR-RUNNER-LOCAL-P11AJ-001/p11aj_runner_build/run.log` contains `PASS: tb_top_managed_sram_provenance_p11aj`. | PASS |
| NR-RUNNER-LOCAL-P11ANB-001 | run_id=20260404_125258 | runner | runner.local.p11anb | Execute selected partial-bucket attn seam regression in compile-backed night-run chain. | `build/night_run/20260404_125258/tasks/NR-RUNNER-LOCAL-P11ANB-001/p11anb_runner_build/run.log` contains `PASS: tb_p11anb_attnlayer0_boundary_seam_contract`. | PASS |
| NR-RUNNER-AUDIT-NEXT-PARTIAL-BUCKET-001 | date=2026-04-04 | runner | runner.local.p11aj | Audit partial bucket `q=1,kv=1,score=0` and gate feasibility into selector truth-table chain. | `build/p11aj/p11aj/run.log` contains `QKV_READY_SCORE_NOT_PREBUILT_TO_SCORES_STAGE PASS`. | PASS |
| NR-BLOCKER-SCORE-STAGE-SHELL-001 | date=2026-04-04 | runner | runner.local.p11aj | Resolve score-stage shell blocker via bounded enum+selector+call-site+writeback cut. | `docs/night_run/BLOCKER_P11BD_SCORE_STAGE_FEASIBILITY.md` status update marks resolved with p11aj/p11anb evidence. | PASS |
| NR-RUNNER-P11ANB-SCORE-FEASIBILITY-PROBE-001 | date=2026-04-04 | runner | runner.local.p11anb | Harden p11anb selector probe for `q=1,kv=1,score=0` score-stage contraction banner. | `build/p11anb/attnlayer0_boundary_seam_contract/run.log` contains `P11ANB_TRANSFORMER_ATTN_SHELL_QKV_READY_SCORE_NOT_PREBUILT_TO_SCORES_STAGE PASS`. | PASS |
| NR-RUNNER-SCORE-STAGE-SHELL-MINCUT-PREP-001 | date=2026-04-04 | runner | runner.local.p11aj | Deliver bounded score-stage shell mincut (`enum+selector+call-site+post->out writeback`). | `src/blocks/TransformerLayer.h` adds `TRANSFORMER_ATTN_COMPAT_SHELL_SCORES_ONLY`; p11aj/p11anb runners PASS. | PASS |
| NR-CHECK-DESIGN-PURITY-002 | run_id=20260404_160344 | checker | checker.design_purity | Re-run design purity checker before score-shell verify rows. | `build/night_run/20260404_160344/tasks/NR-CHECK-DESIGN-PURITY-002/task_execution.log` contains `PASS: check_design_purity`. | PASS |
| NR-RUNNER-LOCAL-P11AJ-SCORES-SHELL-VERIFY-002 | run_id=20260404_160344 | runner | runner.local.p11aj | Re-verify score-stage shell cut in queue-driven compile-backed chain. | `build/night_run/20260404_160344/tasks/NR-RUNNER-LOCAL-P11AJ-SCORES-SHELL-VERIFY-002/p11aj_runner_build/run.log` contains `QKV_READY_SCORE_NOT_PREBUILT_TO_SCORES_STAGE PASS`. | PASS |
| NR-RUNNER-LOCAL-P11ANB-SCORES-SHELL-VERIFY-002 | run_id=20260404_160344 | runner | runner.local.p11anb | Re-verify p11anb selector boundary seam for score-stage shell in queue-driven chain. | `build/night_run/20260404_160344/tasks/NR-RUNNER-LOCAL-P11ANB-SCORES-SHELL-VERIFY-002/p11anb_runner_build/run.log` contains `P11ANB_TRANSFORMER_ATTN_SHELL_QKV_READY_SCORE_NOT_PREBUILT_TO_SCORES_STAGE PASS`. | PASS |
| NR-RUNNER-NEXT-PARTIAL-BUCKET-AUDIT-002 | date=2026-04-04 | runner | runner.local.p11aj | Audit next safest partial bucket `q=1,kv=0,score=0`. | `build/p11aj/p11aj/run.log` contains `Q_READY_KV_NOT_PREBUILT_REMAINS_FULL PASS` and `Q_READY_KV_NOT_PREBUILT_QKV_STAGE_FEASIBILITY_BLOCKED PASS`. | PASS |
| NR-CHECK-DESIGN-PURITY-ACTIVE-001 | run_id=20260404_163302 | checker | checker.design_purity | Active dispatch precheck to keep queue consumption gate stable after archive split. | `build/night_run/20260404_163302/tasks/NR-CHECK-DESIGN-PURITY-ACTIVE-001/task_execution.log` contains `PASS: check_design_purity`. | PASS |
| NR-RUNNER-QKV-BLOCKER-VERIFY-001 | run_id=20260404_163302 | runner | runner.local.p11aj | Re-verify next-partial audit outcome (`q=1,kv=0,score=0`) on compile-backed p11aj chain after queue hygiene split. | `build/night_run/20260404_163302/tasks/NR-RUNNER-QKV-BLOCKER-VERIFY-001/p11aj_runner_build/run.log` contains `Q_READY_KV_NOT_PREBUILT_REMAINS_FULL PASS` and `Q_READY_KV_NOT_PREBUILT_QKV_STAGE_FEASIBILITY_BLOCKED PASS`. | PASS |
| NR-CHECK-DESIGN-PURITY-ACTIVE-002 | run_id=20260404_163447 | checker | checker.design_purity | Active dispatch precheck for the next queue round after archive rollover. | `build/night_run/20260404_163447/tasks/NR-CHECK-DESIGN-PURITY-ACTIVE-002/task_execution.log` contains `PASS: check_design_purity`. | PASS |
| NR-RUNNER-NEXT-PARTIAL-BUCKET-AUDIT-003 | run_id=20260404_163447 | runner | runner.local.p11aj | Audit next safest partial bucket after q-ready/kv-not-prebuilt blocker confirmation. | `build/night_run/20260404_163447/tasks/NR-RUNNER-NEXT-PARTIAL-BUCKET-AUDIT-003/p11aj_runner_build/run.log` contains `Q_READY_KV_NOT_PREBUILT_REMAINS_FULL PASS` and `Q_READY_KV_NOT_PREBUILT_QKV_STAGE_FEASIBILITY_BLOCKED PASS`. | PASS |
| NR-CHECK-DESIGN-PURITY-ACTIVE-004 | date=2026-04-04 | checker | checker.design_purity | Active dispatch precheck for the next partial-bucket audit round. | Local precheck `scripts/check_design_purity.ps1` output contains `PASS: check_design_purity`. | PASS |
| NR-RUNNER-NEXT-PARTIAL-BUCKET-AUDIT-005 | date=2026-04-04 | runner | runner.local.p11aj | Audit next safest partial bucket after q-ready/kv-not-prebuilt QKV+SCORES shell cut and apply bounded shrink when safe. | `build/p11aj/p11aj/run.log` contains `KV_READY_Q_NOT_PREBUILT_SCORE_READY_TO_OUT_STAGE PASS`; `build/p11anb/attnlayer0_boundary_seam_contract/run.log` contains `P11ANB_TRANSFORMER_ATTN_SHELL_KV_READY_Q_NOT_PREBUILT_SCORE_READY_TO_OUT_STAGE PASS`. | PASS |
| NR-CHECK-DESIGN-PURITY-ACTIVE-005 | date=2026-04-04 | checker | checker.design_purity | Active dispatch precheck for the next partial-bucket audit round after bucket-005 shrink. | Local precheck `scripts/check_design_purity.ps1` output contains `PASS: check_design_purity`. | PASS |
| NR-RUNNER-NEXT-PARTIAL-BUCKET-AUDIT-006 | date=2026-04-04 | runner | runner.local.p11aj | Audit next safest partial bucket after kv-ready/q-not-prebuilt score-ready OUT-stage shrink and apply bounded shrink when safe. | `build/p11aj/p11aj/run.log` contains `KV_READY_Q_NOT_PREBUILT_TO_QKV_SCORES_STAGE PASS`; `build/p11anb/attnlayer0_boundary_seam_contract/run.log` contains `P11ANB_TRANSFORMER_ATTN_SHELL_KV_READY_Q_NOT_PREBUILT_TO_QKV_SCORES_STAGE PASS`. | PASS |
| NR-CHECK-DESIGN-PURITY-ACTIVE-006 | date=2026-04-04 | checker | checker.design_purity | Active dispatch precheck for the next partial-bucket audit round after bucket-006 shrink. | Local precheck `scripts/check_design_purity.ps1` output contains `PASS: check_design_purity`. | PASS |
| NR-RUNNER-NEXT-PARTIAL-BUCKET-AUDIT-007 | date=2026-04-04 | runner | runner.local.p11aj | Audit next safest partial bucket after kv-ready/q-not-prebuilt QKV+SCORES shrink and apply bounded shrink when safe. | `build/p11aj/p11aj/run.log` contains `QKV_NOT_PREBUILT_SCORE_READY_TO_OUT_STAGE PASS`; `build/p11anb/attnlayer0_boundary_seam_contract/run.log` contains `P11ANB_TRANSFORMER_ATTN_SHELL_QKV_NOT_PREBUILT_SCORE_READY_TO_OUT_STAGE PASS`. | PASS |
