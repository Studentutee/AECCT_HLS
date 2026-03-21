# P11 Corrected-Chain Catapult Launch Note (P00-011AS)

## Purpose
- This note defines the corrected active-chain Catapult launch entry for first true tool blocker capture.
- Canonical synth entry is fixed to `TopManagedAttentionChainCatapultTop::run`.
- This note is a launch/handoff note, not a full Catapult Tcl user guide.

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
- Project Tcl draft: `scripts/catapult/p11as_corrected_chain_project.tcl`
- Filelist: `scripts/catapult/p11as_corrected_chain_filelist.f`
- Entry TU: `src/catapult/p11as_top_managed_attention_chain_entry.cpp`
- Preflight checker: `scripts/check_p11as_corrected_chain_launch_pack.ps1`
- Reference note: `docs/reference/Catapult_Tcl_Writing_Note_zhTW.md`

## One-Command Usage
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/catapult/run_p11as_corrected_chain_catapult_launch.ps1 -BuildDir build\p11as\catapult_launch
```

## Environment Requirements
- `catapult` command must be discoverable from `PATH`, or from `CATAPULT_HOME\bin\catapult.exe`, or from `MGC_HOME\bin\catapult.exe`.
- Technology/library setup is site-dependent and must be provided on the Catapult machine.

## Tcl Validation Status
- `scripts/catapult/p11as_corrected_chain_project.tcl` should currently be treated as a **launch-pack draft**, not as a fully verified Catapult project script.
- The launch intent is correct: corrected-chain canonical entry, filelist binding, preflight usage, and fail-fast environment probe are aligned with the current repo posture.
- Command-level compatibility on a real Catapult machine is **not yet verified**.
- Until real-tool validation exists, prefer the guidance in `docs/reference/Catapult_Tcl_Writing_Note_zhTW.md` for the minimum safe Tcl skeleton and for the list of commands that are already aligned with the uploaded Catapult training templates.

## Governance Posture
- tool-ready launch-prep only
- not Catapult closure
- not SCVerify closure
- corrected active-chain launch path only
