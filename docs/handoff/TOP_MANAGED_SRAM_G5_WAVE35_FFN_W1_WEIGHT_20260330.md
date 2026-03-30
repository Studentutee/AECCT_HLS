# TOP MANAGED SRAM G5 WAVE3.5 FFN W1 WEIGHT MIGRATION (2026-03-30)

## Summary
- Scope: bounded mincut on `FFNLayer0` W1 weight consume path only.
- Goal: push W1 weight tile consume from direct-SRAM-only to caller-fed descriptor path.
- External formal contract unchanged.
- local-only evidence only.
- not Catapult closure; not SCVerify closure.

## Reality Check
- At round start, `FFNLayer0` W1 tile load directly read weights from SRAM via:
  - `sram[ffn_param_addr_word(param_base, w1_weight_id, w1_idx)]`
- This is the core W1 weight consume loop for FFN first linear stage.

## Selected bounded cut
- Caller preload:
  - `TransformerLayer` and `TransformerLayerTopManagedAttnBridge` preload
    `topfed_ffn_w1_words[FFN_W1_WEIGHT_WORDS]`.
  - caller dispatches valid count via `w1_weight_words`.
- FFN consume:
  - `FFNLayer0CoreWindow` adds optional `topfed_w1_weight_words` and `topfed_w1_weight_words_valid`.
  - W1 loop consumes top-fed weight tile when provided.
- Fallback:
  - if top-fed pointer absent/out-of-range, fallback to legacy SRAM weight read.

## Exact files changed
- `include/FfnDescBringup.h`
- `src/blocks/FFNLayer0.h`
- `src/blocks/TransformerLayer.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `tb/tb_g5_wave35_ffn_w1_weight_migration_p11g5w35.cpp`
- `scripts/local/run_p11g5_wave35_ffn_w1_weight_migration.ps1`

## Deferred boundaries
- W2 weight path not migrated in this round.
- ReLU/W2 broader payload descriptorization not migrated in this round.
- Wave4 attention/phase migrations remain deferred.
