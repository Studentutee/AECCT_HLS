# REPORT_P11OUT1_MAINLINE_CLASSIFICATION_AND_CLOSURE_PLAN

Date: 2026-04-05  
Scope: attention mainline `out=1` family classification + closure planning (local-only)

Related closure statement:
- `docs/night_run/REPORT_P11ATTN_MAINLINE_NO_DIRECT_SRAM_CLOSURE_STATEMENT.md`
- `docs/night_run/REPORT_P11ATTN_CLOSURE_BUNDLE_GUIDE_zhTW.md`

## 1) Compile-backed audit basis
- `scripts/check_design_purity.ps1` -> `PASS: check_design_purity`
- `scripts/check_repo_hygiene.ps1 -Phase pre` -> `PASS: check_repo_hygiene`
- `scripts/local/run_p11aj_top_managed_sram_provenance.ps1` -> `PASS: run_p11aj_top_managed_sram_provenance`
- `scripts/local/run_p11anb_attnlayer0_boundary_seam_contract.ps1` -> `PASS: run_p11anb_attnlayer0_boundary_seam_contract`
- `build/p11aj/p11aj/run.log` contains:
  - `FULLY_PREBUILT_NO_PAYLOAD_DISABLED PASS`
  - `FULLY_PREBUILT_PAYLOAD_OUT_ONLY PASS`
  - `OTHER_PARTIAL_BUCKETS_REMAIN_FULL PASS`
  - `FULL_LOOP_MAINLINE_PATH_TAKEN PASS`
  - `FULL_LOOP_FALLBACK_NOT_TAKEN PASS`
- `build/p11anb/attnlayer0_boundary_seam_contract/run.log` contains:
  - `P11ANB_TRANSFORMER_ATTN_SHELL_SHRINK_FULLY_PREBUILT_OUT_ONLY PASS`
  - `P11ANB_TRANSFORMER_ATTN_SHELL_SHRINK_FULLY_PREBUILT_NO_PAYLOAD_DISABLED PASS`
  - `P11ANB_TRANSFORMER_ATTN_SHELL_SHRINK_OTHER_PARTIAL_STILL_FULL PASS`

## 2) Code-path invariants used for classification
- `Top.h` runloop initializes prebuilt flags per layer as false, then derives:
  - `q_prebuilt_from_top_managed` from `run_p11ad_layer0_top_managed_q`
  - `kv_prebuilt_from_top_managed` from `run_p11ac_layer0_top_managed_kv`
  - `score_prebuilt_from_top_managed` from `ae_mainline_score_path_taken`
  - `out_prebuilt_from_top_managed` from `af_mainline_softmax_output_path_taken`
- AE/AF mainline is entered only under `q_prebuilt_from_top_managed && kv_prebuilt_from_top_managed`.
- If AE score mainline or AF out mainline fails, AF path is not accepted and fallback markers are latched.
- Therefore in current Top-managed mainline:
  - `out_prebuilt_from_top_managed=true` implies `score_prebuilt_from_top_managed=true`
  - and implies `q_prebuilt_from_top_managed=true && kv_prebuilt_from_top_managed=true`

## 3) Remaining `out=1` buckets classification
Bit order: `kv, q, score, out, payload`

### Class 1: mainline-relevant and worth shrinking
- None found in this audit.

### Class 2: reachable but fallback-only / safety-net
- `10110`, `10111`, `01110`, `01111`, `00110`, `00111`
- Reason:
  - These are selector-input reachable (defensive API surface), but not reached by current Top mainline invariants.
  - They represent inconsistent partial-prebuilt descriptors with `out=1` and `score=1` while Q/KV readiness is partial.
  - Current policy keeping them in legacy `FULL` acts as safety-net behavior.

### Class 3: likely unreachable / not mainline-relevant
- `11010`, `11011`, `10010`, `10011`, `01010`, `01011`, `00010`, `00011`
- Reason:
  - `out=1` with `score=0` contradicts current AE/AF derivation chain in Top runloop.
  - No compile-backed evidence indicates these are produced by current mainline execution.

## 4) Phase B verdict driver
- Because Class 1 is empty, this round does not select any `out=1` bucket for shrink.
- Verdict: Phase B skipped (safe-stop, no forced cut).

## 5) Closure-planning conclusion
- Attention mainline currently does not require direct SRAM fallback for the already accepted mainline path evidence.
- Remaining `out=1` unresolved buckets are dominated by fallback/safety-net semantics, not an active mainline closure gap.
- Next round should prefer closure statement/sidecar framing over forced bucket shrink unless a new compile-backed path makes a Class 1 bucket reachable.

## Posture
- local-only evidence
- compile-first
- evidence-first
- not Catapult closure
- not SCVerify closure
