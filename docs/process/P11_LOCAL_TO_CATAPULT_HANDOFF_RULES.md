# P11 Local to Catapult Handoff Rules

## Purpose
- Freeze the accepted local-only handoff boundary for the QKV live-cut family.
- Provide a stable handoff surface for later Catapult-prep work without rewriting accepted history.
- Keep local acceptance meaningful while preventing overclaim of closure state.

## What This Document Is / Is Not
- This document is a boundary freeze and role map for accepted local-only artifacts.
- This document is not a Catapult closure claim.
- This document is not an SCVerify closure claim.
- This document is not an algorithm or interface redesign request.

## Current Accepted Local-Only Family Scope
- Accepted baseline tasks preserved as accepted:
- `P00-011M` accepted (`local smoke scope`)
- `P00-011N` accepted (`local smoke scope`)
- `P00-011O` accepted (`local smoke / local static checks scope`)
- `P00-011P` accepted (`local smoke / local static checks scope`)
- Mainline state remains local-only.
- local-only progress is valid.
- local smoke / local static checks != full Catapult closure.

## Accepted Handoff Surface
- Design-side source artifacts:
- `src/blocks/TernaryLiveQkvLeafKernel.h`
- `src/blocks/TernaryLiveQkvLeafKernelTop.h`
- `src/blocks/AttnLayer0.h`
- Local smoke and integration TB artifacts:
- `tb/tb_ternary_live_leaf_smoke_p11j.cpp`
- `tb/tb_ternary_live_leaf_top_smoke_p11k.cpp`
- `tb/tb_ternary_live_leaf_top_smoke_p11l_b.cpp`
- `tb/tb_ternary_live_leaf_top_smoke_p11l_c.cpp`
- `tb/tb_ternary_live_source_integration_smoke_p11m.cpp`
- `tb/tb_ternary_live_family_source_integration_smoke_p11n.cpp`
- Local gate scripts:
- `scripts/local/run_p11l_local_regression.ps1`
- `scripts/check_design_purity.ps1`
- `scripts/check_interface_lock.ps1`
- `scripts/check_macro_hygiene.ps1`
- `scripts/check_repo_hygiene.ps1`
- Governance and evidence references:
- `docs/process/PROJECT_STATUS_zhTW.txt`
- `docs/milestones/TRACEABILITY_MAP_v12.1.md`
- `docs/milestones/CLOSURE_MATRIX_v12.1.md`
- `docs/process/SYNTHESIS_RULES.md`
- `docs/process/EVIDENCE_BUNDLE_RULES.md`

## Role Classification
- design-side helper
- local top wrapper
- split-interface local top
- TB-only smoke drivers
- one-shot regression scripts
- governance / evidence documents

## Allowed Local-Only Macros / Knobs
- `AECCT_LOCAL_P11M_WQ_SPLIT_TOP_ENABLE`
- `AECCT_LOCAL_P11N_WK_WV_SPLIT_TOP_ENABLE`
- These macros are local-only knobs and do not upgrade acceptance to Catapult closure.
- Macro usage outside approved local contexts is boundary misuse.

## Explicit Deferred Items
- Catapult / SCVerify deferred by design.
- Full runtime closure remains deferred.
- Full numeric correctness closure remains deferred.
- Full family migration closure remains deferred.
- deferred items are intentional.

## Non-Goals
- No Catapult run in this task.
- No SCVerify run in this task.
- No new live migration slice.
- No algorithm change.
- No public signature or Top contract change.
- No dispatcher or block graph redesign.
- No broad refactor or unrelated cleanup.
- No fake formal-closure wording.

## How Later Catapult-Prep Work Should Interpret This Boundary
- Treat the accepted local-only chain as valid progress and a stable handoff baseline.
- Do not relabel local smoke/local static checks as full closure.
- Do not narrow already accepted scope of `P00-011M/N/O/P`.
- Use this boundary to stage later Catapult-prep tasks with explicit new evidence gates.
