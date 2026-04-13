# PREPROC_REWRITE_ROUND2

## Summary
This round starts the first real rewrite block on the active path: `PreprocEmbedSPE.h`.
The file keeps the original name and public entry points, but its active numeric path now uses fp16-first preproc composition and Top-managed contract metadata.

## Exact files changed
- `include/weights.h`
- `src/blocks/PreprocEmbedSPE.h`
- `tb/tb_fp16_preproc_refmodel_gap_probe.cpp`

## What the new bridge does
- keeps `PreprocEmbedSPE(...)` for old checkpoint/copy-style smoke tests
- restores `PreprocCfg`, `PreprocBlockContract`, and `clear_preproc_contract(...)`
- restores `PreprocEmbedSPECoreWindow(...)` so `Top.h` can call the block again
- restores `PreprocEmbedSPEWord16(...)` for focused word16 compare TBs
- moves clean compute into fp16-first helper code inside the same file

## What was actually run
- `tb_fp16_branch_preproc_xwork_smoke`
- `tb_g5_wave2_preproc_payload_migration_p11g5w2`
- `tb_fp16_preproc_u16_trace_ref_compare`
- `tb_fp16_preproc_refmodel_gap_probe`
- `python3 scripts/check_design_purity.py`
- `python3 scripts/check_repo_hygiene.py`

## Current result
- Top preproc path against branch-side fp16 stage-local expectation: PASS
- Top-fed payload migration pilot: PASS
- Authoritative fp16 RefModel compare: FAIL
- Gap probe shows earliest mismatch still lands at `sample=0 token=0 d=1`
  - clean rewrite / stage-local branch: `0xAA03`
  - fp16 RefModel output: `0xAA04`

## Interpretation
This round repaired the active Preproc API shape and connected it back to Top-owned orchestration.
It does not claim RefModel closure. The new clean preproc currently matches the existing fp16 branch-side convention, but that convention still differs from the authoritative fp16 RefModel by 1 ulp at the earliest observed element.

## Known temporary governance deviation
`check_design_purity.py` currently fails because this rewrite round still uses a header-backed weight bridge (`weights.h`) while the final Top-loaded W_REGION path has not been swapped in yet.
This is a temporary migration bridge, not the intended final contract.

## Recommended next step
Add one more split inside preproc compare and log the earliest mismatch at these boundaries:
- decoded input y
- node feature
- fp16-cast embed weight
- product output

Once the first 1-ulp mismatch is explained, keep the same public block API and replace only the internal preproc math/policy.
