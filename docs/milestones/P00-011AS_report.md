# P00-011AS Report

## Scope
- Build a corrected active-chain only Catapult launch pack for first true tool-blocker capture on a machine with Catapult.
- Initial launch-pack baseline started from `TopManagedAttentionChainCatapultTop::run`, but real-tool transcript later corrected the top target to class-level `aecct::TopManagedAttentionChainCatapultTop`.
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
- direct Catapult `analyze -> compile -> elaborate` attempt when available
- optional architecture surface step (`go architecture`) if supported by the installed flow

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
- Remaining compile issues must be judged by the final fatal lines of the real-tool transcript, not by earlier deprecated blockers alone.
