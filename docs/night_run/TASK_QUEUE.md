# TASK_QUEUE

## Purpose
- Provide the executable queue source for night-run dispatch mode v1.1.
- Keep v1.1 bounded to local-only checker/runner tasks with explicit evidence links.

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
| NR-CHECK-DESIGN-PURITY-001 | done | checker | - | checker.design_purity | true | Run design purity checker before any runner dispatch. | Run 20260404_125258 task log contains `PASS: check_design_purity`; exit code = 0. |
| NR-RUNNER-INIT-AGENT-STATE-001 | done | runner | NR-CHECK-DESIGN-PURITY-001 | runner.init_agent_state | true | Execute one local-only runner task after checker passes. | Run 20260404_125258 task log contains `PASS: init_agent_state`; exit code = 0. |
| NR-RUNNER-LOCAL-P11AJ-001 | done | runner | NR-RUNNER-INIT-AGENT-STATE-001 | runner.local.p11aj | true | Execute first compile-backed local runner in night-run queue. | Run 20260404_125258 runner `run.log` contains `PASS: tb_top_managed_sram_provenance_p11aj` and `SELECTED_PARTIAL_QKV_SCORE_NO_PAYLOAD_TO_OUT_STAGE PASS`. |
| NR-RUNNER-LOCAL-P11ANB-001 | done | runner | NR-RUNNER-LOCAL-P11AJ-001 | runner.local.p11anb | true | Execute selected partial-bucket attn seam regression in compile-backed night-run chain. | Run 20260404_125258 runner `run.log` contains `P11ANB_TRANSFORMER_ATTN_SHELL_SHRINK_SELECTED_PARTIAL_QKV_SCORE_NO_PAYLOAD_OUT_ONLY PASS` and `PASS: tb_p11anb_attnlayer0_boundary_seam_contract`. |
| NR-RUNNER-AUDIT-NEXT-PARTIAL-BUCKET-001 | done | runner | NR-RUNNER-LOCAL-P11ANB-001 | runner.local.p11aj | true | Audit next partial bucket (`q=1, kv=1, score=0`) for safe non-FULL stage contraction. | Run `build/p11aj/p11aj/run.log` contains `QKV_READY_SCORE_NOT_PREBUILT_REMAINS_FULL PASS` and `QKV_READY_SCORE_NOT_PREBUILT_SCORE_STAGE_FEASIBILITY_BLOCKED_ENUM_SURFACE PASS`. |
| NR-BLOCKER-SCORE-STAGE-SHELL-001 | blocked | runner | NR-RUNNER-AUDIT-NEXT-PARTIAL-BUCKET-001 | runner.local.p11aj | true | Hold score-stage shell code cut until compat-shell stage surface is added in bounded form. | Blocker package tracked at `docs/night_run/BLOCKER_P11BD_SCORE_STAGE_FEASIBILITY.md`; no safe score-stage shell path exists in current `TransformerLayer` selector/call-site surface. |
| NR-RUNNER-P11ANB-SCORE-FEASIBILITY-PROBE-001 | done | runner | NR-BLOCKER-SCORE-STAGE-SHELL-001 | runner.local.p11anb | true | Harden p11anb selector probe with explicit `q=1,kv=1,score=0` bucket feasibility banner and non-regression checks. | Run `build/p11anb/attnlayer0_boundary_seam_contract/run.log` contains `P11ANB_TRANSFORMER_ATTN_SHELL_QKV_READY_SCORE_NOT_PREBUILT_REMAINS_FULL PASS` and `PASS: tb_p11anb_attnlayer0_boundary_seam_contract`. |
| NR-RUNNER-SCORE-STAGE-SHELL-MINCUT-PREP-001 | queued | runner | NR-BLOCKER-SCORE-STAGE-SHELL-001,NR-RUNNER-P11ANB-SCORE-FEASIBILITY-PROBE-001 | runner.local.p11aj | true | Prepare bounded implementation slice for new score-stage compat shell (`enum + selector + call-site`) after blocker is explicitly cleared. | Must clear `NR-BLOCKER-SCORE-STAGE-SHELL-001` and then show compile-backed PASS lines for selected score-not-prebuilt bucket contraction plus non-regression banners. |

## Notes
- Keep one row per executable task.
- v1 dispatch supports only known runner keys; unknown key is fail-fast.
- v1.1 compile-backed keys currently include `runner.local.p11aj` and `runner.local.p11anb`.
- Do not claim Catapult/SCVerify closure from night-run v1.1 outputs.
