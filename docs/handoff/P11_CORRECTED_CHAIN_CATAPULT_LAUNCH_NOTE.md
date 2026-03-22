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
- For compile-flag policy, the project now intentionally **defaults to `options set Input/CompilerFlags -D__SYNTHESIS__`** because the user prefers explicit compiler-define form for first true-tool bring-up.
- `scripts/catapult/p11as_corrected_chain_project.tcl` should currently be treated as a **launch-pack draft**, not as a fully verified Catapult project script.
- The launch intent is correct: corrected-chain canonical entry, filelist binding, preflight usage, and fail-fast environment probe are aligned with the current repo posture.
- Command-level compatibility on a real Catapult machine is **not yet verified**.
- The current preferred direction is to rewrite the compile-first draft toward **`options set Input/...` style configuration**, based on the user-provided Catapult GUI transcript.

## GUI Transcript-Based Option Keys (new evidence)
The user has now confirmed these option-key families directly from Catapult transcript output:
- `options set Input/CppStandard c++20`
- `options set Input/SearchPath <path>` and `-append`
- `options set Input/LibPaths <path>` and `-append`
- `options set Input/CompilerFlags -D__SYNTHESIS__`
- `options set ComponentLibs/SearchPath ... -append`
- `options set ComponentLibs/TechLibSearchPath {...}`
- `options set Flows/QuestaSIM/Path ...`
- `options set Flows/DesignCompiler/Path ...`
- `options set Flows/VSCode/INSTALL ...`
- `options set Flows/VSCode/GDB_PATH ...`

## Recommended next Tcl correction
For the next corrected-chain Tcl revision, prefer this order:
1. `options defaults`
2. `project new`
3. `options set Input/CppStandard c++20`
4. `options set Input/SearchPath ...` for repo-local include/source paths
5. `options set Input/CompilerFlags -D__SYNTHESIS__`
6. optional `Input/LibPaths`
7. `solution file add ... -type C++`
8. `solution design set ... -top`
9. `go compile`

Do **not** re-expand the draft back to `solution new`, `-cflags`, `go elaborate`, or `go architecture` unless real-tool evidence on the target Catapult machine justifies it.

## Site-dependent settings posture
- `ComponentLibs/SearchPath` and `ComponentLibs/TechLibSearchPath` should be treated as site-dependent, not hard-coded single-machine truth.
- `Flows/.../Path` keys are now known, but they are not required for the corrected-chain compile-first draft unless the site explicitly needs them.
- Prefer environment-variable or user-supplied override style for these settings.

## Open items for first real-tool run
- Use `Input/CompilerFlags -D__SYNTHESIS__` as the current project default for first true compile; only switch away from this if real-tool transcript proves the target Catapult version auto-infers defines from bare tokens.
- Confirm whether corrected-chain compile first-pass truly needs `Input/LibPaths`, or whether `Input/SearchPath` alone is sufficient.
- Confirm whether repeated `-append` or brace-list style is more robust for repo-local multi-path setup on the target Catapult version.

## Governance Posture
- tool-ready launch-prep only
- not Catapult closure
- not SCVerify closure
- corrected active-chain launch path only
