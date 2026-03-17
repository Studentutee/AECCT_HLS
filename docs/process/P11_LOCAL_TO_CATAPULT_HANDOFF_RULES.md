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
- `P00-011Q` accepted (`local smoke / local static checks scope`, handoff freeze baseline)
- `P00-011R` accepted (`first Catapult-facing compile-prep probe`, `single-slice representative`, `local compiler evidence only`)
- `P00-011S` accepted (`WK/WV family compile-prep expansion`, `family representative`, `local compiler evidence only`)
- `P00-011T` accepted (`QKV shape SSOT consolidation`, `compile-time shape SSOT`, `runtime validation only`, `local compiler evidence only`)
- `P00-011U` accepted (`QKV payload-metadata SSOT bridge`, `local-only`, `runtime metadata guard expectations validated against compile-time SSOT chain`)
- `P00-011V` accepted (`QKV WeightStreamOrder continuity fence`, `local-only`, `validation-only continuity checks against authoritative local-build metadata`)
- `P00-011W` accepted (`QKV exported-artifact / loader-facing continuity fence`, `local-only`, `validation-only continuity checks against repo-tracked export artifact metadata`)
- `P00-011X` accepted (`QKV export-consumer semantic continuity fence`, `local-only`, `validation-only semantic continuity checks for repo-tracked export consumer interpretation`)
- `P00-011Y` accepted (`QKV local runtime-handoff fence`, `local-only`, `validation-only continuity checks for runtime-facing handoff expectations derived from the accepted authority chain`)
- `P00-011Z` accepted (`QKV local runtime-consume probe`, `local-only`, `read-only runtime-facing consume probe for L0_WQ/L0_WK/L0_WV derived from the accepted authority chain`)
- Mainline state remains local-only.
- local-only progress is valid.
- local smoke / local static checks != full Catapult closure.
- accepted local-only progress remains valid.
- P00-011Q handoff freeze remains authoritative.
- P00-011Q freeze boundary remains authoritative.
- P00-011R WQ compile-prep probe remains valid baseline.
- P00-011S WK/WV family compile-prep expansion remains valid baseline.
- P00-011T QKV shape SSOT consolidation remains valid baseline.
- P00-011U local-only payload-metadata SSOT bridge remains valid.
- P00-011V local-only WeightStreamOrder continuity fence remains valid.
- P00-011W local-only exported-artifact / loader-facing continuity fence remains valid.
- P00-011X local-only export-consumer semantic continuity fence remains valid.
- P00-011Y local-only runtime-handoff continuity fence remains valid.
- P00-011Z local-only runtime-consume probe remains valid.

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
- Compile-prep probe artifacts:
- `src/blocks/TernaryLiveQkvLeafKernelCatapultPrepTop.h`
- `tb/tb_ternary_live_leaf_top_compile_prep_p11r.cpp`
- `scripts/check_compile_prep_surface.ps1`
- `scripts/local/run_p11r_compile_prep.ps1`
- `docs/milestones/P00-011R_report.md`
- `tb/tb_ternary_live_leaf_top_compile_prep_family_p11s.cpp`
- `scripts/check_compile_prep_family_surface.ps1`
- `scripts/local/run_p11s_compile_prep_family.ps1`
- `docs/milestones/P00-011S_report.md`
- `src/blocks/TernaryLiveQkvLeafKernelShapeConfig.h`
- `src/blocks/TernaryLiveQkvWeightStreamOrderContinuityFence.h`
- `scripts/check_qkv_shape_ssot.ps1`
- `docs/milestones/P00-011T_report.md`
- `scripts/check_qkv_payload_metadata_ssot.ps1`
- `docs/milestones/P00-011U_report.md`
- `scripts/check_qkv_weightstreamorder_continuity.ps1`
- `docs/milestones/P00-011V_report.md`
- `scripts/check_qkv_export_artifact_continuity.ps1`
- `docs/milestones/P00-011W_report.md`
- `scripts/check_qkv_export_consumer_semantics.ps1`
- `docs/milestones/P00-011X_report.md`
- `scripts/check_qkv_runtime_handoff_continuity.ps1`
- `docs/milestones/P00-011Y_report.md`
- `tb/tb_ternary_qkv_runtime_probe_p11z.cpp`
- `scripts/local/run_p11z_runtime_probe.ps1`
- `docs/milestones/P00-011Z_report.md`
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
- P00-011R is not Catapult closure.
- P00-011R is not SCVerify closure.
- P00-011S is not Catapult closure.
- P00-011S is not SCVerify closure.
- P00-011T is not Catapult closure.
- P00-011T is not SCVerify closure.
- P00-011U is not Catapult closure.
- P00-011U is not SCVerify closure.
- P00-011V is not Catapult closure.
- P00-011V is not SCVerify closure.
- P00-011W is not Catapult closure.
- P00-011W is not SCVerify closure.
- P00-011X is not Catapult closure.
- P00-011X is not SCVerify closure.
- P00-011Y is not Catapult closure.
- P00-011Y is not SCVerify closure.
- P00-011Z is not Catapult closure.
- P00-011Z is not SCVerify closure.
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
- Keep `P00-011Q` as the authoritative freeze boundary while advancing compile-prep probes such as `P00-011R` and `P00-011S`.
- `P00-011S` is a WK/WV family compile-prep expansion with local compiler evidence only.
- `P00-011T` is a QKV shape SSOT consolidation task with compile-time shape SSOT and runtime validation only.
- `P00-011U` is a local-only payload-metadata SSOT bridge task that keeps runtime validation scoped to accepted metadata guards.
- `P00-011V` is a local-only WeightStreamOrder continuity fence task that checks compile-time continuity against the authoritative local-build metadata surface.
- `P00-011W` is a local-only exported-artifact / loader-facing continuity fence task that checks repo-tracked offline artifact metadata against the accepted SSOT + WeightStreamOrder continuity chain.
- `P00-011X` is a local-only export-consumer semantic continuity fence task that checks matrix_id-driven interpretation semantics on the repo-tracked export consumer surface against the accepted SSOT + WeightStreamOrder + export-artifact continuity chain.
- `P00-011Y` is a local-only runtime-handoff continuity fence task that derives a single runtime-facing QKV handoff expectation from the accepted SSOT + WeightStreamOrder + exported-artifact + export-consumer continuity chain, without introducing a second authority source.
- `P00-011Z` is a local-only runtime-consume probe task that performs read-only matrix_id-driven consumption checks for `L0_WQ/L0_WK/L0_WV` against the accepted runtime-handoff authority chain, without introducing a second authority source.
- Use this boundary to stage later Catapult-prep tasks with explicit new evidence gates.
