# DIRECT_SRAM_ENDGAME_BLOCKERS_20260331

## Blocker B1: W4-C0 softmax bridge familyization is no longer a clean low-risk cut in this session

### Location
- `src/blocks/AttnPhaseBTopManagedSoftmaxOut.h::attn_phaseb_top_managed_softmax_out_mainline`

### Why blocked for this session
- C0 would need selector/family semantics inside the online softmax execution neighborhood (`INIT_ACC`, `RENORM`, `ACC`, `WRITEBACK` adjacency) rather than a pure capacity/span extension like B8/B9.
- To keep anti-fallback and no-spurious-touch guarantees, this likely requires synchronized expansion of observability signals and Top helper passthrough for multiple descriptors.
- Risk boundary is now close to core dataflow skeleton changes, which violates this session's bounded-safety stopping rule.

### Evidence basis
- B9 is clean PASS with all required checks, so stop is not due to regression.
- SoftmaxOut bridge hook is currently single-case selector at init-acc consume and tightly coupled to online path loops.
- See current hook anchors in:
  - `src/blocks/AttnPhaseBTopManagedSoftmaxOut.h` (`phase_tile_bridge_selected`, `ATTN_P11AF_MAINLINE_INIT_ACC_*`, `ATTN_P11AF_MAINLINE_RENORM_*`, `ATTN_P11AF_MAINLINE_ACC_*`, `ATTN_P11AF_MAINLINE_WRITEBACK_*`).

### Recommended next step
- Plan a dedicated C0 mini-campaign with isolated observability contract design and fresh task-local TB for multi-case softmax consume invariants before implementation.
