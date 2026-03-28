# P11AC Helper Split Audit (2026-03-28)

## Scope
- local-only helper-path split audit and implementation
- target: `attn_work_pkt_ch_t in_ch` mixed `X/WK/WV` payload risk

## Change Summary
- split helper input path into dedicated channels:
  - `attn_x_work_pkt_ch_t x_ch`
  - `attn_wk_work_pkt_ch_t wk_ch`
  - `attn_wv_work_pkt_ch_t wv_ch`
- kept helper output channel unchanged (`attn_work_pkt_ch_t out_ch`)
- preserved metadata validation checks for token/tile/phase boundaries

## Files Touched
- `src/blocks/AttnPhaseATopManagedKv.h`
- `scripts/check_p11ac_phasea_surface.ps1`
- `tb/tb_kv_build_stream_stage_p11ac.cpp` (minimal compatibility fix for leaf-kernel symbol drift and pointer lvalue binding)
- `docs/milestones/P00-011AC_report.md` (minimal wording alignment to satisfy existing post-surface governance string gate)

## Commands
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_p11ac_phasea_surface.ps1 -OutDir build\p11ac_phasea_split_night -Phase pre`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local/run_p11ac_phasea_top_managed.ps1 -BuildDir build\p11ac_impl_split`
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_p11ac_phasea_surface.ps1 -OutDir build\p11ac_phasea_split_night -Phase post`

## Evidence Excerpt
- `PASS: check_p11ac_phasea_surface` (pre/post final pass)
- `STREAM_ORDER PASS`
- `MEMORY_ORDER PASS`
- `EXACT_SCR_KV_COMPARE PASS`
- `MAINLINE_PATH_TAKEN PASS`
- `FALLBACK_NOT_TAKEN PASS`
- `fallback_taken = false`
- `PASS: run_p11ac_phasea_top_managed`

## Governance Posture
- local-only
- helper-path risk reduction only
- not Catapult closure
- not SCVerify closure
