# REF_V3 Catapult Launch Note

## Purpose
- Define the ref_v3 Catapult compile-entry launch wiring.
- Keep flow separation explicit:
  - Runbook handles how to run and capture evidence.
  - Project Tcl handles what to compile.
- This note is launch wiring + current blocker handoff, not closure evidence.

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
- Runbook:
  - `skills/catapult_shell_runbook/SKILL.md`

## Canonical Binary
- Use fixed absolute path first:
  - `/cad/mentor/Catapult/2025.3/Mgc_home/bin/catapult`
- Do not rely on `catapult` from `PATH` unless explicitly allowed and version/path-equivalent is proven.

## Validated Interactive Login Recipe
- Preferred remote target:
  - `peter@140.124.41.193`
  - port `2190`
- Preferred SSH mode:
  - `ssh -tt`
  - `-i .codex_ssh/id_codex`
  - `-o UserKnownHostsFile=.codex_ssh/known_hosts`
  - `-o StrictHostKeyChecking=yes`
- Remote default shell observed in the successful run:
  - `/bin/tcsh`
- Success-defining environment traits:
  - `TERM=xterm-256color`
  - `LM_LICENSE_FILE=5280@lshc:26585@lshc:1717@lshc`
  - `MGLS_LICENSE_FILE=`
  - `CDS_LIC_FILE=`

### Windows / PowerShell outer SSH command
```powershell
ssh -tt -i .codex_ssh/id_codex `
  -o UserKnownHostsFile=.codex_ssh/known_hosts `
  -o StrictHostKeyChecking=yes `
  -p 2190 peter@140.124.41.193
```

### tcsh-compatible remote commands
```tcsh
set REPO_ROOT=/home/peter/AECCT/AECCT_HLS-master
set TS=`date +%Y%m%d_%H%M%S`
set OUTDIR=$REPO_ROOT/build/ref_v3/catapult_compile_check_tty_$TS
set CATAPULT_BIN=/cad/mentor/Catapult/2025.3/Mgc_home/bin/catapult
set PROJECT_TCL=$REPO_ROOT/scripts/catapult/ref_v3/project.tcl

mkdir -p $OUTDIR/project
cd $REPO_ROOT
setenv AECCT_REFV3_REPO_ROOT $REPO_ROOT
setenv AECCT_REFV3_CATAPULT_OUTDIR $OUTDIR/project

echo "exit" | $CATAPULT_BIN -shell |& tee $OUTDIR/license_probe.log
$CATAPULT_BIN -shell -file $PROJECT_TCL -logfile $OUTDIR/catapult_internal.log |& tee $OUTDIR/catapult_console.log
```

## License Probe Success Criteria
Use the probe below before classifying any design blocker:

```bash
printf "exit\n" | /cad/mentor/Catapult/2025.3/Mgc_home/bin/catapult -shell
```

Must see at least one of:
- `Connected to license server ... (LIC-13)`
- `Catapult product license successfully checked out ... (LIC-14)`

## Current Known Status (updated 2026-04-17)
- Interactive login bring-up is now validated for `ref_v3`.
- `project.tcl` has been observed to run under true Catapult.
- `go compile` has been reached in interactive login mode.
- The first true blocker is no longer shell/license bring-up.
- The current first true blocker is source compile:
  - `RefModel.h(1): unrecognized token`
  - `Compilation aborted (CIN-5)`
- Latest known `messages.txt` path from the successful interactive run:
  - `/home/peter/AECCT/AECCT_HLS-master/Catapult_15/SIF/messages.txt`

## Blocker Classification Rule
- If shell mode is wrong and `LM_LICENSE_FILE` is missing, classify as runbook / env-mode blocker.
- If `LIC-13/LIC-14` is visible and transcript has `go compile`, do not keep reporting `TTY_REQUIRED_INTERACTIVE_LOGIN_UNAVAILABLE`.
- After `go compile`, the first blocker must move to compile-facing categories such as source compile / include path / wrapper-entry issues.

## Governance Posture
- compile-entry check only
- run-only/flow-wiring scope
- not Catapult closure
- not SCVerify closure
- no design-core modification in launch-note scope
