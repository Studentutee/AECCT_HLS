# P11AS Corrected-Chain Remote Diagnostic Handoff (2026-03-28)

## 1) Scope / posture
- This is a task-local handoff for recent remote Catapult diagnostics.
- Scope is audit-only / temporary-diagnostic preservation.
- `not Catapult closure`
- `not SCVerify closure`
- `sample memory is diagnostic only, not canonical policy`
- This note does **not** update core governance status files.

## 2) Proven baseline (temporary differential, non-canonical)
The following differential baseline is proven to advance corrected-chain flow through extract:
- `solution library add CBDK_TSMC40_Arm_f2.0`
- `solution library add ccs_sample_mem`
- `solution library add ccs_sample_rom`
- `directive set -CLOCKS {clk {-CLOCK_PERIOD 10}}`
- `directive set SCHED_USE_MULTICYCLE true` (must be set no later than memories / set early enough)

Under this temporary differential condition, the flow has evidence of passing:
- compile
- libraries
- assembly
- architect
- extract

## 3) Stage progression summary (diagnostic chain)
1. Extract bring-up baseline established: compile/libraries/assembly/architect/extract all pass; `DIR-5/SIF-4/CRAAS-6/q_bits:div` not present in that run.
2. Post-extract forward push reached switching and first failed with:
   - `No QuestaSIM installation found`
3. After site-local Questa path override (`Flows/QuestaSIM/Path` + `vsim` discoverable):
   - Switching failed on simulator license environment:
   - `Unable to find the license file ... SALT_LICENSE_SERVER ...`
   - `Unable to checkout a license`
   - `Invalid license environment`
4. After temporary build-local license override:
   - SALT-related failures were pushed away.
   - New first blocker became:
   - missing PLI object `novas_fli.so` (`vsim-PLI-3002`)
   - `default.fsdb` missing remained a downstream symptom.

## 4) Key diagnosis
- Earlier libraries-stage fatal remains high-suspicion linked to custom memory libraries (based on differential evidence that sample-memory baseline can pass libraries+).
- Sample-memory differential is diagnostic only and must not be treated as final policy.
- Divider extract blocker was pushed away under early-valid `SCHED_USE_MULTICYCLE true`.
- Current first blocker is **site-local simulator PLI dependency**:
  - `/share/PLI/MODELSIM/LINUX64/novas_fli.so` missing
  - error id includes `vsim-PLI-3002`
- `default.fsdb` absence is downstream from simulation/PLI failure, not root cause.

## 5) Evidence map (directory -> conclusion -> key files)
### A. `build/p11as_overnight_p1a_retry_20260327_235138/`
- Meaning: extract bring-up baseline with early multicycle placement validated.
- Key files:
  - `p11as_cbdk_samplemem_extract_audit_p1a.tcl`
  - `run_remote.sh`
  - `catapult_console.log`
  - `messages.txt`
  - `run_meta.txt`
- Key signals:
  - `COUNT_COMPILE=1`
  - `COUNT_LIBRARIES=1`
  - `COUNT_ASSEMBLY=1`
  - `COUNT_ARCHITECT=1`
  - `COUNT_EXTRACT=1`
  - `COUNT_DIR5=0`
  - `COUNT_SIF4=0`
  - `COUNT_CRAAS6=0`
  - `COUNT_QBITSDIV=0`

### B. `build/p11as_postextract_forwardpush2_20260328_075716/`
- Meaning: forward push beyond extract identified switching entry blocker.
- Key files:
  - `p11as_cbdk_samplemem_postextract_forwardpush2.tcl`
  - `run_postextract_forwardpush2.sh`
  - `console_remote.log`
  - `messages_remote.txt`
  - `run_meta_remote.txt`
- Key signals:
  - `COUNT_EXTRACT=1`
  - `COUNT_INSTANCE=1`
  - switching stage starts, then:
  - `No QuestaSIM installation found`

### C. `build/p11as_switching_questa_override_20260328_084955/`
- Meaning: site-local Questa path override fixed simulator discovery; blocker moved to license env.
- Key files:
  - `p11as_cbdk_samplemem_switching_questa_override.tcl`
  - `run_switching_questa_override.sh`
  - `precheck_remote.log`
  - `console_remote.log`
  - `messages_remote.txt`
  - `run_meta_remote.txt`
  - `phase1_which_vsim.txt`
  - `phase1_command_v_vsim.txt`
  - `phase3_fsdb_check.txt`
- Key signals:
  - `P11AS_OBS_QUESTA_PATH_SET_RC 0`
  - `P11AS_OBS_PRECHECK_WHICH_VSIM=/cad/mentor/Questa_Sim/2025.2_2/questasim/bin/vsim`
  - switching fatal:
  - `Unable to find the license file ... SALT_LICENSE_SERVER`
  - `Unable to checkout a license`
  - `Invalid license environment`

### D. `build/p11as_switching_license_override_20260328_094617/`
- Meaning: temporary build-local license override pushed beyond SALT fatal; new blocker is PLI dependency.
- Key files:
  - `p11as_cbdk_samplemem_switching_license_override.tcl`
  - `run_switching_license_override.sh`
  - `precheck_remote.log`
  - `console_remote.log`
  - `messages_remote.txt`
  - `run_meta_remote.txt`
  - `fsdb_check.txt`
  - `phase1_license_env_snapshot.txt`
  - `phase1_license_env_snapshot2.txt`
- Key signals:
  - `COUNT_UNABLE_FIND_LICENSE=0`
  - `COUNT_UNABLE_CHECKOUT_LICENSE=0`
  - `COUNT_INVALID_LICENSE_ENV=0`
  - first fatal:
  - `Load of "/share/PLI/MODELSIM/LINUX64/novas_fli.so" failed`
  - `(vsim-PLI-3002) Failed to load PLI object file`
  - `FSDB_EXISTS=0`

## 6) Minimal next action
Only one next step:
- Find and inject a verifiable site-local Novas PLI path (`novas_fli.so`) in the build-local switching run script, then rerun `go switching`.

## 7) Explicit non-goals
- Do not update core governance docs yet:
  - `docs/process/PROJECT_STATUS_zhTW.txt`
  - `docs/milestones/CLOSURE_MATRIX_v12.1.md`
  - `docs/milestones/TRACEABILITY_MAP_v12.1.md`
- Do not rewrite design code in this handoff step.
- Do not treat sample-memory differential as final/canonical solution.
