# TB Inventory (backup_io8, local-only)

## Scope
This inventory is a low-risk hygiene snapshot for the current backup_io8 debug turn.
It focuses on:
- `tb_backup_io8_loadw_infer_smoke` and directly related runners.
- quick visibility of runner-backed vs runnerless TB files.
- archive planning only (no bulk file moves in this turn).

## Columns
| TB file path | Has scripts/local/run_*.ps1 | Type | Recent usage | Recommended action | Why |
|---|---|---|---|---|---|
| `tb/tb_backup_io8_loadw_infer_smoke.cpp` | yes: `run_backup_io8_loadw_infer_smoke.ps1`, `run_backup_demo_smoke.ps1` | mainline | primary TB in this turn | keep | direct io8/loadw/infer mainline check |
| `tb/tb_backup_wave1_memory_packing_smoke.cpp` | yes: `run_backup_wave1_wave3_smoke.ps1`, `run_backup_demo_smoke.ps1` | module | backup wave1 smoke | keep | still isolates payload/packing boundary issues |
| `tb/tb_backup_wave2_quant_linear_smoke.cpp` | yes: `run_backup_wave1_wave3_smoke.ps1`, `run_backup_demo_smoke.ps1` | module | backup wave2 smoke | keep | separate quant-linear seam from backup_io8 mainline |
| `tb/tb_backup_wave3_io8_boundary_smoke.cpp` | yes: `run_backup_wave1_wave3_smoke.ps1`, `run_backup_demo_smoke.ps1` | seam | backup wave3 io8 boundary | keep | seam-level ownership is still useful |
| `tb/tb_backup_attn_qkv_live_contract.cpp` | yes: `run_backup_attn_qkv_live_contract.ps1` | seam | backup attn live contract | keep | active seam TB with dedicated runner |
| `tb/tb_p11anb_attnlayer0_boundary_seam_contract.cpp` | yes: `run_p11anb_attnlayer0_boundary_seam_contract.ps1` | seam | layer0 attn boundary seam | keep | active seam gate |
| `tb/tb_transformerlayer_ffn_higher_level_ownership_seam.cpp` | yes: `run_p11au_transformerlayer_ffn_higher_level_ownership_seam.ps1` | seam | FFN ownership seam | keep | complements backup_io8 scope without mega-merge |
| `tb/tb_top_ffn_handoff_assembly_smoke_p11av.cpp` | yes: `run_p11av_top_ffn_handoff_assembly_smoke.ps1` | mainline | FFN handoff assembly smoke | keep | active handoff assembly validation |
| `tb/tb_preproc_channel_trace_compare_pilot.cpp` | yes: `run_preproc_channel_trace_compare_pilot.ps1` | probe | preproc trace compare pilot | keep | useful probe-style comparator |
| `tb/tb_q_path_scaffold_p11ad_prep.cpp` | yes: `run_p11ad_prep_q_path.ps1` | milestone | q path prep scaffold | keep | still runner-backed |
| `tb/tb_qk_score_scaffold_p11ae_prep.cpp` | yes: `run_p11ae_prep_qk_score.ps1` | milestone | qk score prep scaffold | keep | still runner-backed |
| `tb/tb_softmax_out_scaffold_p11af_prep.cpp` | yes: `run_p11af_prep_softmax_out.ps1` | milestone | softmax out prep scaffold | keep | still runner-backed |
| `tb/tb_layernorm_affine_consume_trace_p11lna.cpp` | no (no scripts/local hit in this turn) | probe | historical LayerNorm affine consume trace | archive-candidate | runnerless and partially overlapped by current dual-compare probes |
| `tb/tb_top_m1.cpp` | no (no scripts/local hit in this turn) | milestone | legacy M1 bring-up TB | archive-candidate | no active runner reference; mostly historical docs |
| `tb/tb_top_m2.cpp` | no (no scripts/local hit in this turn) | milestone | legacy M2 bring-up TB | archive-candidate | no active runner reference; mostly historical docs |

## Runner-backed vs runnerless (this focused set)
- runner-backed: 12
- runnerless: 3

## Archive-candidate list (plan only, not moved)
- `tb/tb_layernorm_affine_consume_trace_p11lna.cpp`
- `tb/tb_top_m1.cpp`
- `tb/tb_top_m2.cpp`

## Notes
- No broad TB merge was performed in this turn.
- No bulk archive move was performed in this turn.
- Recommended next round: build a compact `active runner map` (tb -> runner) before any minimal archive execution.
