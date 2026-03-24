# P00-011AS Report

## Scope
- Build a corrected active-chain only Catapult launch pack for first true tool-blocker capture on a machine with Catapult.
- Initial launch-pack baseline started from `TopManagedAttentionChainCatapultTop::run`, but real-tool transcript later corrected the top target to class-level `aecct::TopManagedAttentionChainCatapultTop`.
- Latest user-shared compile-first transcript excerpt (2026-03-24) shows `go compile` launched under the class-level target and, by keyword scan on the shared excerpt, no `# Error` / `Compilation aborted` was detected.
- Do not broaden into repo-wide cleanup/migration.

## Landed Artifacts
- `src/catapult/p11as_top_managed_attention_chain_entry.cpp`
- `scripts/catapult/p11as_corrected_chain_filelist.f`
- `scripts/catapult/p11as_corrected_chain_project.tcl`
- `scripts/check_p11as_corrected_chain_launch_pack.ps1`
- `scripts/catapult/run_p11as_corrected_chain_catapult_launch.ps1`
- `docs/handoff/P11_CORRECTED_CHAIN_CATAPULT_LAUNCH_NOTE.md`

## What This Enables
- One-command launch flow that performs:
- corrected-chain preflight checks (entry/filelist/include/macro/path/fallback exclusion)
- environment probe (`MGC_HOME` / `CATAPULT_HOME` / `PATH` / command detection)
- fail-fast when Catapult command is unavailable
- direct Catapult `analyze -> compile` attempt when available, with solution transcript capture
- optional deeper flow steps only after compile-first evidence justifies them

## Governance Posture
- tool-ready launch-prep
- not Catapult closure
- not SCVerify closure
- corrected active-chain launch-prep only
- not full repo migration

## Transcript-based corrections after initial landing
- `solution design set ... -top` should target class-level `aecct::TopManagedAttentionChainCatapultTop`, not `TopManagedAttentionChainCatapultTop::run`.
- Catapult project Tcl must not user-define `__SYNTHESIS__`; define macro list may be empty.
- Preflight/checker should validate reserved-macro-forbidden policy instead of requiring non-empty macro list.
- Historical real-tool blockers have included:
  - `HIER-55` from method-level top target mismatch
  - `CIN-249` from non-persistent `wq_top / wk_top / wv_top`
  - `HIER-10` from local `ac_channel` not crossing hierarchy boundary
- These should now be treated as historical transcript checkpoints; latest compile judgement must follow the newest solution transcript, not stale blocker snapshots.

## Latest shared compile-first transcript refresh (2026-03-24)
- Latest user-shared transcript excerpt shows:
  - `solution design set aecct::TopManagedAttentionChainCatapultTop -top`
  - `go compile` launched on the corrected-chain solution
- Keyword scan on the shared excerpt did not detect `# Error` or `Compilation aborted`.
- Current visible warning families in the shared excerpt are mainly:
  - `CRD-549 / CRD-111 / CRD-68 / CRD-1 / CRD-186` (predominantly from `third_party/ac_types` instantiation chains)
  - `CIN-63` multiple-tops warnings (`TernaryLiveL0Wq/Wk/WvRowTop`, `TopManagedAttentionChainCatapultTop`)
- This evidence is sufficient to update launch-pack posture and historical blocker ordering, but not sufficient to claim Catapult closure or SCVerify closure.
