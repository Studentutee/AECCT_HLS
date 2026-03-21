# P11 Corrected-Chain Catapult Launch Note (P00-011AS)

## Purpose
- This note defines the corrected active-chain Catapult launch entry for first true tool blocker capture.
- Canonical synth entry is fixed to `TopManagedAttentionChainCatapultTop::run`.

## Canonical Corrected Active Path
- `TopManagedAttentionChainCatapultTop::run`
- `run_transformer_layer_loop_top_managed_attn_bridge`
- `run_p11ad_layer0_top_managed_q`
- `run_p11ac_layer0_top_managed_kv`
- `run_p11ae_layer0_top_managed_qk_score`
- `run_p11af_layer0_top_managed_softmax_out`
- `TransformerLayerTopManagedAttnBridge`
- `FFNLayer0TopManagedWindowBridge`

## Launch Pack Files
- Runner: `scripts/catapult/run_p11as_corrected_chain_catapult_launch.ps1`
- Project Tcl: `scripts/catapult/p11as_corrected_chain_project.tcl`
- Filelist: `scripts/catapult/p11as_corrected_chain_filelist.f`
- Entry TU: `src/catapult/p11as_top_managed_attention_chain_entry.cpp`
- Preflight checker: `scripts/check_p11as_corrected_chain_launch_pack.ps1`

## One-Command Usage
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/catapult/run_p11as_corrected_chain_catapult_launch.ps1 -BuildDir build\p11as\catapult_launch
```

## Environment Requirements
- `catapult` command must be discoverable from `PATH`, or from `CATAPULT_HOME\bin\catapult.exe`, or from `MGC_HOME\bin\catapult.exe`.
- Technology/library setup is site-dependent and must be provided on the Catapult machine.

## Governance Posture
- tool-ready launch-prep only
- not Catapult closure
- not SCVerify closure
- corrected active-chain launch path only
