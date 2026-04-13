# FP16 rewrite scaffold round 1

## Summary
This round does not switch the active Top/mainline implementation yet.
It lays down the first code-side contracts for the clean rewrite path:

- `Fp16RewriteTypes.h`
- `TopManagedWindowTypes.h`
- `Fp16WeightProvider.h`
- `Fp16RewriteTopContract.h`
- `tb_fp16_rewrite_scaffold_compile.cpp`

## Why this round exists
The agreed rewrite direction is:

1. keep Top as the only shared-SRAM owner
2. keep the external 4-channel integration shell stable
3. rewrite the compute chain around fp16 RefModel semantics
4. stop letting compute code directly depend on `weights.h`
5. later swap the weight backend from header-backed dumps to Top-loaded W_REGION

This round only lays down the scaffolding for items 1/3/4.

## What changed
- archived current active `Top.h`, `AecctTop.h`, `PreprocEmbedSPE.h`, `TransformerLayer.h`, and `FinalHead.h`
- added common fp16 rewrite datatypes and helpers
- added explicit Top-managed window descriptor structs
- added a preproc-focused weight-provider abstraction with a temporary header-backed backend
- added one small compile/smoke test for the new scaffolding

## Closure posture
- not Catapult closure
- not SCVerify closure
- not functional rewrite closure
