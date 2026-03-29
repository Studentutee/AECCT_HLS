# TOP MANAGED SRAM MIN CUTS (2026-03-29)

## Task C1: Top-Owned Contract Dispatch (Preproc / LN / FinalHead)
1. Summary
- Switched active Top infer path from wrapper-owned contract assembly to explicit Top-owned contract assembly and dispatch.
- `run_preproc_block`, `run_layernorm_block`, and `run_infer_pipeline` now build contracts in Top and dispatch core entries.

2. Exact files changed
- `src/Top.h`

3. Exact commands run
- `git status --short`
- `rg -n "PreprocEmbedSPECoreWindow|LayerNormBlockCoreWindow|FinalHeadCorePassABTopManaged|run_preproc_block|run_layernorm_block|run_infer_pipeline" src/Top.h src/blocks`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11ah_full_loop_local_e2e.ps1 -BuildDir build/p11ah/top_managed_sram_push`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11aj_top_managed_sram_provenance.ps1 -BuildDir build/p11aj/top_managed_sram_push`

4. Actual execution evidence / log excerpt
- `build/p11ah/top_managed_sram_push/run.log`:
  - `FULL_LOOP_MAINLINE_PATH_TAKEN PASS`
  - `FULL_LOOP_FALLBACK_NOT_TAKEN PASS`
  - `PASS: run_p11ah_full_loop_local_e2e`
- `build/p11aj/top_managed_sram_push/run.log`:
  - `PROVENANCE_STAGE_AC PASS`
  - `PROVENANCE_STAGE_AD PASS`
  - `PROVENANCE_STAGE_AE PASS`
  - `PROVENANCE_STAGE_AF PASS`
  - `PASS: run_p11aj_top_managed_sram_provenance`

5. Governance posture
- Local-only evidence.
- Top remains sole production shared-SRAM owner.
- not Catapult closure; not SCVerify closure.

6. Residual risks
- Compatibility wrappers still exist and can be used by other legacy callsites.
- Default infer loop is still pointer-facing orchestration, not fully switched to deep bridge path.

7. Next recommended step
- Extend static guard to forbid accidental reintroduction of wrapper-owned dispatch in Top active path.

## Task C2: Top Preload Of Transformer Sublayer1 Norm Params
1. Summary
- Added Top-side preload helper for sublayer1 LN gamma/beta and invoked it in both pointer and deep-bridge layer loops.
- Added `sublayer1_norm_preloaded_by_top` flag in `TransformerLayer` and `TransformerLayerTopManagedAttnBridge` with guarded fallback.

2. Exact files changed
- `src/Top.h`
- `src/blocks/TransformerLayer.h`

3. Exact commands run
- `rg -n "load_layer_sublayer1_norm_params|TransformerLayerTopManagedAttnBridge|TransformerLayer\(" src/Top.h src/blocks/TransformerLayer.h`
- `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11aj_top_managed_sram_provenance.ps1 -BuildDir build/p11aj/top_managed_sram_push`

4. Actual execution evidence / log excerpt
- `build/p11aj/top_managed_sram_push/run.log`:
  - `FULL_LOOP_MAINLINE_PATH_TAKEN PASS`
  - `FULL_LOOP_FALLBACK_NOT_TAKEN PASS`
  - `FINAL_X_EXPECTED_COMPARE_HARDENED PASS`
  - `PASS: tb_top_managed_sram_provenance_p11aj`

5. Governance posture
- Local-only evidence.
- Ownership shift is Top -> block consumption only.
- not Catapult closure; not SCVerify closure.

6. Residual risks
- Guarded fallback path still allows in-block preload when caller does not set the preload flag.

7. Next recommended step
- Gradually require preload flag in additional Top-managed callsites, then narrow legacy fallback usage.
