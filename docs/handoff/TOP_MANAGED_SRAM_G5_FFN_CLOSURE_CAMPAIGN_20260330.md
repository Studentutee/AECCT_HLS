# TOP MANAGED SRAM G5 FFN CLOSURE CAMPAIGN (2026-03-30)

## Scope
- Bounded local-only FFN closure push after Wave3/Wave3.5.
- No external formal contract change.
- No remote simulator / PLI / site-local flow touched.

## Subwave Plan and Result
1. Subwave A (W2 input `a` top-fed consume): DONE
- `FFNLayer0` W2 tile load now consumes caller-fed `topfed_w2_input_words` when provided.
- `TransformerLayer` stage-split dispatch preloads ReLU output into caller-fed W2 input descriptor.

2. Subwave B (W2 weight tile descriptor): DONE
- `FFNLayer0` W2 tile load now consumes caller-fed `topfed_w2_weight_words` when provided.
- `TransformerLayer` preloads W2 weights and dispatches valid descriptor window.

3. Subwave C (W2 bias descriptor): DONE
- `FFNLayer0` W2 out loop now consumes caller-fed `topfed_w2_bias_words` when provided.
- `TransformerLayer` preloads W2 bias words and dispatches valid descriptor window.

4. Subwave D (fallback boundary hardening): DONE (bounded)
- Legacy fallback retained for compatibility only.
- Added targeted validation proving top-fed path dominates when descriptors are provided.

## Exact Files Changed (campaign scope)
- `include/FfnDescBringup.h`
- `src/blocks/FFNLayer0.h`
- `src/blocks/TransformerLayer.h`
- `scripts/check_top_managed_sram_boundary_regression.ps1`
- `scripts/local/run_p11g5_ffn_closure_campaign.ps1`
- `tb/tb_g5_ffn_closure_campaign_p11g5fc.cpp`

## Validation Chain (local-only)
- `scripts/local/run_p11g5_ffn_closure_campaign.ps1`: PASS
- `scripts/check_top_managed_sram_boundary_regression.ps1`: PASS
- `scripts/local/run_p11g5_wave3_ffn_payload_migration.ps1`: PASS
- `scripts/local/run_p11g5_wave35_ffn_w1_weight_migration.ps1`: PASS
- `scripts/local/run_p11ah_full_loop_local_e2e.ps1 -BuildDir build/p11ah/g5_ffn_closure_campaign`: PASS
- `scripts/local/run_p11aj_top_managed_sram_provenance.ps1 -BuildDir build/p11aj/g5_ffn_closure_campaign`: PASS
- `scripts/check_helper_channel_split_regression.ps1`: PASS
- `scripts/check_design_purity.ps1`: PASS
- `scripts/check_repo_hygiene.ps1 -Phase pre`: PASS
- `scripts/check_repo_hygiene.ps1 -Phase post`: PASS

## Governance Posture
- local-only evidence
- compile-first / diagnostic-first
- not Catapult closure
- not SCVerify closure

## Deferred Boundary
- W1 fallback removal not in this round.
- W1/W2 complete fallback elimination not in this round.
- Wave4 Attn/Transformer/phase payload migration remains deferred.
