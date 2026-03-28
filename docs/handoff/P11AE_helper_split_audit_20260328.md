# P11AE Helper Split Audit (2026-03-28)

## Scope
- local-only helper-path split audit and implementation
- target: `attn_phaseb_qk_pkt_ch_t in_ch` mixed `Q/K` payload risk

## Change Summary
- split helper input path into dedicated channels:
  - `attn_phaseb_q_pkt_ch_t q_ch`
  - `attn_phaseb_k_pkt_ch_t k_ch`
- kept output channel contract unchanged (`attn_phaseb_qk_pkt_ch_t out_ch`)
- kept packet metadata checks (`phase_id/subphase_id/token/head_group/tile`) in consume path

## Files Touched
- `src/blocks/AttnPhaseBTopManagedQkScore.h`
- `scripts/check_p11ae_impl_surface.ps1`

## Commands
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_p11ae_impl_surface.ps1 -OutDir build\p11ae_impl_night -Phase pre`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11ae_impl_qk_score.ps1 -BuildDir build\p11ae_impl_split`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_p11ae_impl_surface.ps1 -OutDir build\p11ae_impl_night -Phase post`

## Evidence Excerpt
- `PASS: check_p11ae_impl_surface` (pre/post)
- `QK_SCORE_MAINLINE PASS`
- `SCORE_EXPECTED_COMPARE PASS`
- `MAINLINE_SCORE_PATH_TAKEN PASS`
- `FALLBACK_NOT_TAKEN PASS`
- `fallback_taken = false`
- `PASS: run_p11ae_impl_qk_score`

## Governance Posture
- local-only
- helper-path risk reduction only
- not Catapult closure
- not SCVerify closure
