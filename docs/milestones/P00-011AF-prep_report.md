# P00-011AF-prep Report - Softmax/Output Scaffold (Local-Only, Compile-Isolated)

## Summary
- `P00-011AF-prep` adds compile-isolated softmax/output scaffold TB/checker/runner.
- The prep scope is independent before AC merge and does not require AC-only design symbols.

## Scope
- In scope:
- softmax/output scaffold and validators
- compile/run independent before AC merge
- no dependency on non-landed AC headers
- Out of scope:
- no `src/` design edits
- no final softmax/output integration
- no Catapult/SCVerify claim

## Files changed
- `tb/tb_softmax_out_scaffold_p11af_prep.cpp`
- `scripts/check_p11af_prep_surface.ps1`
- `scripts/local/run_p11af_prep_softmax_out.ps1`
- `docs/milestones/P00-011AF-prep_report.md`

## Exact commands executed
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_p11af_prep_surface.ps1 -OutDir build\p11af_prep -Phase pre`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11af_prep_softmax_out.ps1 -BuildDir build\p11af_prep\p11af_prep`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_p11af_prep_surface.ps1 -OutDir build\p11af_prep -Phase post`

## Actual execution evidence excerpt
- `build\p11af_prep\check_p11af_prep_surface.log`
- `PASS: check_p11af_prep_surface`
- `build\p11af_prep\p11af_prep\run.log`
- `PASS: tb_softmax_out_scaffold_p11af_prep`
- `PASS: run_p11af_prep_softmax_out`

## Result / verdict wording
- local-only scaffold progress is valid.
- not Catapult closure.
- not SCVerify closure.
