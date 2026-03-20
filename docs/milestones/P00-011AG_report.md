# P00-011AG Report - Attention-Chain Real-Mainline Correction Validator (Local-Only)

## Summary
- Landed a repo-tracked correction validator for AC->AD->AE->AF with real mainline orchestration evidence.
- Added a one-shot batch runner for tonight re-run.

## Scope
- local-only
- attention-chain correction validation focused on AC/AD/AE/AF
- optional Catapult-facing compile-prep progress chaining in same batch runner

## Files changed
- `tb/tb_attention_chain_correction_validator_p11ag.cpp`
- `scripts/local/run_p11ag_attention_chain_correction.ps1`
- `scripts/local/run_p11ag_attention_chain_batch.ps1`
- `docs/milestones/P00-011AG_report.md`

## Build command
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11ag_attention_chain_correction.ps1 -BuildDir build\p11ag\validator`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11ag_attention_chain_batch.ps1 -BuildDir build\p11ag`

## Run command
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11ag_attention_chain_correction.ps1 -BuildDir build\p11ag\validator`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11ag_attention_chain_batch.ps1 -BuildDir build\p11ag`

## Actual execution evidence
- `build\p11ag\p11ag_validator\run.log`
  - `REAL_MAINLINE_PATH_TAKEN PASS`
  - `STAGE_KV_STAGING PASS`
  - `STAGE_Q_PATH PASS`
  - `STAGE_SCORE_QK PASS`
  - `STAGE_ONLINE_SOFTMAX_NORM PASS`
  - `STAGE_FINAL_OUTPUT_WRITEBACK PASS`
  - `fallback_taken = false`
  - `FALLBACK_NOT_TAKEN PASS`
  - `PASS: tb_attention_chain_correction_validator_p11ag`
  - `PASS: run_p11ag_attention_chain_correction`
- `build\p11ag\p11ac\run.log`
  - `PASS: run_p11ac_phasea_top_managed`
- `build\p11ag\p11ad\run.log`
  - `PASS: run_p11ad_impl_q_path`
- `build\p11ag\p11ae\run.log`
  - `PASS: run_p11ae_impl_qk_score`
- `build\p11ag\p11af\run.log`
  - `PASS: run_p11af_impl_softmax_out`
- `build\p11ag\p11aeaf_e2e\run.log`
  - `PASS: tb_e2e_correction_smoke_p11aeaf`
  - `PASS: run_p11aeaf_e2e_smoke`
- `build\p11ag\catapult_progress\run.log`
  - `PASS: run_p11aeaf_catapult_progress`
- `build\p11ag\batch_summary.txt`
  - `status: PASS`
  - `PASS: run_p11ag_attention_chain_batch`

## Scope / limitation
- local-only
- not Catapult closure
- not SCVerify closure
- not full runtime closure
- not full algorithm closure
- validator focuses attention-chain correction (`AC/AD/AE/AF`) and does not claim full decoder/FFN/final-head correction closure
