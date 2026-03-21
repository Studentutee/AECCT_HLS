# P00-011AR Report

## Scope
- Land active-chain boundary corrections first on the synth-facing/Catapult-facing path rooted at `TopManagedAttentionChainCatapultTop`.
- Keep Top as sole shared-SRAM owner/arbiter/writeback controller.
- Preserve local-only posture; this is not Catapult closure and not SCVerify closure.

## Landed This Round
- Phase B active helpers (`AttnPhaseBTopManagedQkScore`, `AttnPhaseBTopManagedSoftmaxOut`) now accept generic `SramView` and keep explicit `head_group_id` + `subphase_id` packet checks in the active path.
- Phase A active helpers (`AttnPhaseATopManagedQ`, `AttnPhaseATopManagedKv`) now accept generic `SramView` in the mainline and tile emit/writeback edges.
- Top active call edges (`run_p11ac`/`run_p11ad`/`run_p11ae`/`run_p11af`) now use templated `SramView` forwarding in `Top.h`.
- FFN active bridge now calls array-shaped core entry `FFNLayer0CoreWindow<..., u32_t(&)[SRAM_WORDS]>`; legacy pointer core entry is retained as `FFNLayer0CoreWindowDirect` for compatibility.

## Evidence
- `run_p11am_catapult_compile_surface`: PASS, mainline taken, fallback not taken.
- `run_p11an_attn_deep_boundary`: PASS.
- `run_p11ao_ffn_deep_boundary`: PASS.
- `run_p11ap_active_chain_residual_cleanup`: PASS (pre/post residual raw-pointer checker PASS on targeted active Attn+FFN chain).

## Governance Posture
- local-only progress
- not Catapult closure
- not SCVerify closure
- active-chain boundary correction only, not full repo migration
