# P11AD Helper Split Audit (2026-03-28)

## Scope
- local-only helper-path split audit and implementation
- target: `attn_q_work_pkt_ch_t in_ch` mixed `X/WQ` payload risk

## Change Summary
- split helper input path into dedicated channels:
  - `attn_q_x_work_pkt_ch_t x_ch`
  - `attn_q_wq_work_pkt_ch_t wq_ch`
- kept output channel contract unchanged (`attn_q_work_pkt_ch_t out_ch`)
- preserved packet metadata checks in consume path

## Files Touched
- `src/blocks/AttnPhaseATopManagedQ.h`
- `scripts/check_p11ad_impl_surface.ps1`
- `tb/tb_q_path_impl_p11ad.cpp` (minimal compatibility fix for leaf-kernel symbol drift)

## Commands
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_p11ad_impl_surface.ps1 -OutDir build\p11ad_impl_night -Phase pre`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11ad_impl_q_path.ps1 -BuildDir build\p11ad_impl_split`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_p11ad_impl_surface.ps1 -OutDir build\p11ad_impl_night -Phase post`

## Evidence Excerpt
- `PASS: check_p11ad_impl_surface` (pre/post)
- `Q_PATH_MAINLINE PASS`
- `Q_EXPECTED_COMPARE PASS`
- `MAINLINE_Q_PATH_TAKEN PASS`
- `FALLBACK_NOT_TAKEN PASS`
- `fallback_taken = false`
- `PASS: run_p11ad_impl_q_path`

## Governance Posture
- local-only
- helper-path risk reduction only
- not Catapult closure
- not SCVerify closure
