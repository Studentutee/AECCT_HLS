# P00-011AT Report

## Scope
- Minimal governance sync for corrected-chain `CIN-63` cleanup evidence.
- Record one next-stage Catapult run-only audit after compile-first (`go libraries`).
- Keep scope strict: compile-first + next-stage run-only audit only.

## Root Cause (CIN-63)
- Corrected-chain production compile graph pulled in `src/blocks/TernaryLiveQkvLeafKernelTop.h` top wrappers (`#pragma hls_design top`) via include path contamination.
- This made Catapult see multiple top candidates in the same compile graph.

## Minimal Patch (already landed before this sync)
- `src/blocks/AttnLayer0.h`
- `src/blocks/AttnPhaseATopManagedQ.h`
- `src/blocks/AttnPhaseATopManagedKv.h`
- Strategy:
  - stop production path from including top-wrapper header
  - keep leaf top wrappers for standalone/leaf-top usage
  - use kernel-split function calls on production path

## Compile-First Rerun Evidence (true machine)
- Canonical binary:
  - `/cad/mentor/Catapult/2025.3/Mgc_home/bin/catapult`
- Canonical project Tcl:
  - `scripts/catapult/p11as_corrected_chain_project.tcl`
- Canonical top:
  - `aecct::TopManagedAttentionChainCatapultTop`
- Exact Catapult launch command:
  - `/cad/mentor/Catapult/2025.3/Mgc_home/bin/catapult -shell -file scripts/catapult/p11as_corrected_chain_project.tcl`
- Catapult exit code:
  - `0`
- Authority transcript:
  - `/home/peter/AECCT/AECCT_HLS-master/Catapult_5/aecct_TopManagedAttentionChainCatapultTop.v1/messages.txt`
- Before/after:
  - before (`Catapult_3/.../messages.txt`): `CIN-63=7`, `CRD-549=12`, `# Error=0`, `Compilation aborted=0`, `Completed transformation 'compile'=1`
  - after (`Catapult_5/.../messages.txt`): `CIN-63=0`, `CRD-549=12`, `# Error=0`, `Compilation aborted=0`, `Completed transformation 'compile'=1`

## Next-Stage Run-Only Audit (go libraries)
- Method:
  - non-interactive remote shell via `ssh ... "bash -lc '...'"` with explicit canonical binary path
  - compile-first Tcl body reused and appended with `go libraries` in audit-only Tcl artifact
- Catapult launch command for this audit:
  - `/cad/mentor/Catapult/2025.3/Mgc_home/bin/catapult -shell -file /home/peter/AECCT/AECCT_HLS-master/build/catapult/p11as_runonly_libraries_20260327_035155/p11as_libraries_audit.tcl`
- Catapult exit code:
  - `2`
- Latest authority transcript:
  - `/home/peter/AECCT/AECCT_HLS-master/Catapult_6/aecct_TopManagedAttentionChainCatapultTop.v1/messages.txt`
- Transcript counters:
  - `# Error=0`
  - `Compilation aborted=0`
  - `Completed transformation 'compile'=1`
  - `Completed transformation 'libraries'=0`
  - `CIN-63=0`
  - `CRD-549=12`
- First true blocker after compile-first:
  - `go libraries` fails at techlib loading gate
  - key lines:
    - `Info: Please set ComponentLibs/TechLibSearchPath ... (LIB-220)`
    - `Error: Unable to load techlibs`

## Governance Posture
- compile-first scope completed with `CIN-63` cleanup evidence (`7 -> 0`)
- next-stage audit scope only (run-only)
- not Catapult closure
- not SCVerify closure
