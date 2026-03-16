# EVIDENCE_BUNDLE_RULES

## Purpose
- Define the local-only reproducible evidence bundle for P00-011P one-shot regression handoff.
- Keep raw execution logs as primary evidence while adding fixed machine-readable and handoff-friendly artifacts.

## Scope
- Local-only gate and artifacts.
- Catapult and SCVerify remain deferred unless explicitly requested.
- This bundle does not replace existing raw logs.

## Required Bundle Outputs
- `build\p11n\EVIDENCE_MANIFEST_p11p.txt`
- `build\p11n\EVIDENCE_SUMMARY_p11p.md`
- `build\p11n\warning_summary_p11p.txt`
- `build\p11n\verdict_p11p.json`

## Verdict JSON Contract
`verdict_p11p.json` must be valid JSON and include fixed top-level keys:
- `task_id`
- `overall`
- `prechecks`
- `regression`
- `compares`
- `artifacts`

## Warning Summary Policy
- Warning summary uses strict build-log allowlist only:
  - `build_p11j.log`
  - `build_p11k.log`
  - `build_p11l_b.log`
  - `build_p11l_c.log`
  - `build_p11m_baseline.log`
  - `build_p11m_macro.log`
  - `build_p11n_baseline.log`
  - `build_p11n_macro.log`
- Manual/sandbox logs are excluded from warning aggregation.
- Warning summary is non-blocking; structural violations remain fail-fast.

## Manifest Minimum Entries
`EVIDENCE_MANIFEST_p11p.txt` must list at least:
- one-shot run log (`build/p11n/run_p11p_regression.log`)
- `warning_summary_p11p.txt`
- `EVIDENCE_SUMMARY_p11p.md`
- `verdict_p11p.json`
- core raw run logs:
  - `run_p11j.log`
  - `run_p11k.log`
  - `run_p11l_b.log`
  - `run_p11l_c.log`
  - `run_p11m_baseline.log`
  - `run_p11m_macro.log`
  - `run_p11n_baseline.log`
  - `run_p11n_macro.log`

## One-Shot Entry
- Canonical command:
  - `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11l_local_regression.ps1 -BuildDir build\p11n *> build\p11n\run_p11p_regression.log`
- Success still requires final string:
  - `PASS: run_p11l_local_regression`
