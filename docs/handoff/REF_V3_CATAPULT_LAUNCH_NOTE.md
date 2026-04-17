# REF_V3 Catapult Launch Note

## Purpose
- Define the ref_v3 Catapult compile-entry launch wiring.
- Keep flow separation explicit:
  - Runbook handles how to run and capture evidence.
  - Project Tcl handles what to compile.
- This note is launch wiring only, not closure evidence.

## Canonical Entry Bundle
- Project Tcl: `scripts/catapult/ref_v3/project.tcl`
- Filelist: `scripts/catapult/ref_v3/filelist.f`
- Canonical synth top target: `aecct_ref::ref_v3::RefV3CatapultTop`
- Entry TU: `AECCT_ac_ref/catapult/ref_v3/ref_v3_catapult_top_entry.cpp`
- Catapult TB TU: `AECCT_ac_ref/tb_catapult/ref_v3/tb_ref_v3_catapult_scverify.cpp`

## Runbook Wiring
- Override example:
  - `scripts/catapult/ref_v3/project_override_ref_v3.env.example`
- Launch runner:
  - `scripts/catapult/ref_v3/run_ref_v3_catapult_launch.ps1`

## Canonical Binary
- Use fixed absolute path first:
  - `/cad/mentor/Catapult/2025.3/Mgc_home/bin/catapult`
- Do not rely on `catapult` from `PATH` unless explicitly allowed and version/path-equivalent is proven.

## Interactive Login Shell Requirement
- License result can differ by shell mode on the same machine.
- Non-interactive/non-login shell is known to risk:
  - `TERM=dumb`
  - empty `LM_LICENSE_FILE`
  - `mgls_errno=515`
- Preferred execution mode:
  - interactive login shell (example: `ssh -tt <host>`)
- If interactive login shell evidence is unavailable, treat as:
  - `LICENSE_ENV_MODE_MISMATCH=true`
  - `EXECUTION_DEFERRED_TTY_REQUIRED=true`
- Do not classify non-interactive 515 as a design compile blocker.

## Recommended Compile-Entry Command (interactive login shell)
```bash
export AECCT_REFV3_REPO_ROOT="<repo_root>"
export AECCT_REFV3_CATAPULT_OUTDIR="<outdir>/project"
/cad/mentor/Catapult/2025.3/Mgc_home/bin/catapult \
  -shell \
  -file "<repo_root>/scripts/catapult/ref_v3/project.tcl" \
  -logfile "<outdir>/catapult_internal.log" \
  2>&1 | tee "<outdir>/catapult_console.log"
```

## Governance Posture
- compile-entry check only
- run-only/flow-wiring scope
- not Catapult closure
- not SCVerify closure
- no design-core modification in launch-note scope
