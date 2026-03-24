# P11 Corrected-Chain Catapult Launch Note (P00-011AS)

## Purpose
- This note defines the corrected active-chain Catapult launch entry, launch-pack corrections, and latest compile-first transcript posture on a real Catapult machine.
- Canonical synth entry target（經真機 transcript 修正）為 `aecct::TopManagedAttentionChainCatapultTop`；`run` 是 class 內被 synthesize 的 interface method。
- This note is a launch/handoff note, not a full Catapult Tcl user guide.

## Canonical Corrected Active Path
- `aecct::TopManagedAttentionChainCatapultTop`
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

## Transcript-driven corrections to old assumptions
- Old assumption (deprecated): `solution design set TopManagedAttentionChainCatapultTop::run -top`
- Current corrected target: `solution design set aecct::TopManagedAttentionChainCatapultTop -top`
- Old assumption (deprecated): Catapult project Tcl should set `Input/CompilerFlags -D__SYNTHESIS__`
- Current corrected policy: do not user-define `__SYNTHESIS__`; empty macro list is legal.
- Current posture: class-level top target and `__SYNTHESIS__` policy are updated; historical blockers (`HIER-55`, `CIN-249`, `HIER-10`) should now be treated as prior iterations, and latest compile judgement must follow the newest solution transcript.

## Tcl Validation Status
- For compile-flag policy, Catapult project Tcl **must not** user-define `__SYNTHESIS__`; define macro list may be empty, and `Input/CompilerFlags` should only be set when non-empty user flags actually exist.
- `scripts/catapult/p11as_corrected_chain_project.tcl` should currently be treated as a **launch-pack draft with transcript-backed corrections**, not as a full closure script.
- The launch intent is correct: corrected-chain canonical entry, filelist binding, preflight usage, and fail-fast environment probe are aligned with the current repo posture.
- Basic command-level compatibility on a real Catapult machine is now evidenced by shared `go analyze -> go compile` transcript excerpts, but full closure is **not** verified.
- The current preferred direction remains to keep the compile-first draft in **`options set Input/...` style configuration**, based on the user-provided Catapult GUI transcript.

## Latest shared compile-first transcript refresh (2026-03-24)
- Latest user-shared solution transcript excerpt uses class-level target:
  - `solution design set aecct::TopManagedAttentionChainCatapultTop -top`
- The shared excerpt shows `go compile` has started on the corrected-chain solution.
- Keyword scan on the shared excerpt did **not** detect `# Error` or `Compilation aborted`.
- Current visible output is warning-only, mainly:
  - `CRD-549 / CRD-111 / CRD-68 / CRD-1 / CRD-186` from `third_party/ac_types` instantiation chains
  - `CIN-63` multiple-tops warnings for `TernaryLiveL0Wq/Wk/WvRowTop` and `TopManagedAttentionChainCatapultTop`
- Therefore this note should treat prior `HIER-55` / `CIN-249` / `HIER-10` blockers as historical transcript checkpoints, not as the current visible fatal.

## GUI Transcript-Based Option Keys (new evidence)
The user has now confirmed these option-key families directly from Catapult transcript output:
- `options set Input/CppStandard c++20`
- `options set Input/SearchPath <path>` and `-append`
- `options set Input/LibPaths <path>` and `-append`
- `Input/CompilerFlags` 只有在非空 user flags 存在時才設定；不可手動帶入 `-D__SYNTHESIS__`
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
5. optional non-empty `Input/CompilerFlags`（不得 user-define `__SYNTHESIS__`）
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
- Keep `Input/CompilerFlags` empty unless non-reserved user flags are actually needed; do not restore `-D__SYNTHESIS__` into Catapult project Tcl.
- Confirm whether corrected-chain compile first-pass truly needs `Input/LibPaths`, or whether `Input/SearchPath` alone is sufficient.
- Confirm whether repeated `-append` or brace-list style is more robust for repo-local multi-path setup on the target Catapult version.

## Governance Posture
- tool-ready launch-prep only
- not Catapult closure
- not SCVerify closure
- corrected active-chain launch path only
