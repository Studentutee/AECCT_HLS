# M0~M16 Audit (v12.1 single-X_WORK, evidence-first)

- Audit date: 2026-03-10
- Baseline: latest repo state
- Scope: bookkeeping audit only (no code changes)

## Method and Decision Rules

1. SSOT priority used in this audit:
   - `docs/process/AECCT_v12_M0-M24_plan_zhTW.txt`
   - `docs/process/AECCT_HLS_Governance_v12_zhTW.txt`
   - `docs/spec/AECCT_HLS_Spec_v12.1_zhTW.txt`
   - `docs/architecture/AECCT_HLS_Architecture_Guide_v12.1_zhTW.txt`
   - `docs/changelog/AECCT_v12_single_X_WORK_change_summary_zhTW.txt`
   - supplementary spell docs (secondary only)
2. Formal PASS rule (strict): no formal PASS without both:
   - `docs/milestones/Mxx_report.md`
   - `docs/milestones/Mxx_artifacts/{diff.patch, build.log, run_tb.log, verdict.txt, file_manifest.txt}`
3. Evidence-first rule (explicit): if initial human hypothesis conflicts with actual repo evidence, output follows repo evidence.
4. Spec-drift rule: if latest single-X_WORK baseline changed expected semantics and repo still follows older semantics, status is `spec drift / needs re-baseline`.
5. Separation rule: implementation existence and formal milestone closure are evaluated separately.

## Initial Human Hypothesis (Non-Binding)

This hypothesis is recorded for traceability only and is overridden by repo evidence when inconsistent:

- M0: code exists but evidence missing
- M1: code exists but evidence missing
- M2: code exists but evidence missing
- M3: PASS with evidence
- M4: PASS with evidence
- M5: PASS with evidence
- M6: code exists but evidence missing
- M7: code exists but evidence missing
- M8: code exists but evidence missing
- M9: code exists but evidence missing
- M10: code exists but evidence missing
- M11: code exists but evidence missing
- M12: spec drift / needs re-baseline
- M13: code exists but evidence missing
- M14: code exists but evidence missing
- M15: code exists but evidence missing
- M16: spec drift / needs re-baseline

## Milestone Status Matrix

| Milestone | Latest plan expected scope | Repo evidence found | Formal evidence pack present? | Status | Main gap | Recommended next action |
|---|---|---|---|---|---|---|
| M0 | Project skeleton, contract/SSOT skeleton, `tb_top_m0` bring-up | `src/Top.h`, `include/AecctProtocol.h`, `tb/tb_top_m0.cpp`, gate scripts | No | code exists but evidence missing | Missing `M00_report.md` and `M00_artifacts/*` | Create M00 evidence pack from current repo and rerun `tb_top_m0` + gates |
| M1 | Top FSM mode separation and deterministic response ordering | FSM in `src/Top.h`, `tb/tb_top_m1.cpp` | No | code exists but evidence missing | Missing formal M01 evidence pack | Create M01 report/artifacts with FSM state/opcode matrix evidence |
| M2 | Single main SRAM logical regions with `X_WORK`; `READ_MEM` foundation | `tb/tb_top_m2.cpp` exists; `include/SramMap.h` and `src/Top.h` still use physical `X_PAGE0/X_PAGE1` and swap-like flow | No | spec drift / needs re-baseline | Current storage baseline remains dual-page centric versus single-X_WORK formal baseline | Re-baseline M2 storage semantics and produce M02 evidence pack under single-X_WORK rules |
| M3 | CFG_RX completion + ModelDesc/ModelShapes validation | `docs/milestones/M03_report.md`, `M03_artifacts/*`, `tb/tb_top_m3.cpp` includes len/shape checks | Yes | PASS with evidence | Minor hygiene noise only (encoding/comments) | Keep PASS; optionally add hygiene note in future maintenance patch |
| M4 | `SET_W_BASE` + `LOAD_W` + `WeightStreamOrder` checks | `docs/milestones/M04_report.md`, `M04_artifacts/*`, `tb/tb_top_m4.cpp` | Yes | PASS with evidence | Evidence is primarily TB-centered; design-side purity tightening appears in later patch evidence | Keep PASS; add cross-reference from M04 to later purity reinforcement evidence for traceability |
| M5 | in_fifo + IO_REGION + SET_OUTMODE + HALTED/DEBUG foundation | `docs/milestones/M05_report.md`, `M05_artifacts/*`, `tb/tb_top_m5.cpp`, `tb/tb_top_m6.cpp` | Yes | code exists but evidence missing | M05 evidence focuses DEBUG/HALTED path; full M5 scope (especially SET_OUTMODE/in_fifo/IO_REGION framing) is not fully closed in M05 pack | Regenerate M05 closure evidence with explicit full-scope checklist coverage |
| M6 | INFER stub end-to-end flow (outmode length/flow first, not numeric correctness) | `tb/tb_top_m6.cpp`, INFER flow in `src/Top.h` | No | code exists but evidence missing | Missing M06 report/artifacts | Create M06 evidence pack from current flow and expected outmode behavior |
| M7 | PreprocEmbedSPE writes preproc result to X_WORK | `src/blocks/PreprocEmbedSPE.h`, `include/PreprocDescBringup.h`, `tb/tb_preproc_m7.cpp` | No | code exists but evidence missing | Missing formal M07 evidence pack | Create M07 report/artifacts and explicitly document X_WORK mapping |
| M8 | Synth-safe two-pass LayerNormBlock | `src/blocks/LayerNormBlock.h`, `tb/tb_layernorm_m8.cpp` | No | code exists but evidence missing | Missing M08 closure evidence | Create M08 checkpoint-based evidence pack |
| M9 | Attention bring-up (Q/K/V, SCR_K/SCR_V path, checkpointability) | `src/blocks/AttnLayer0.h`, `tb/tb_attn_m9a_qkv.cpp`, `tb/tb_attn_m9b_scores.cpp`, `tb/tb_attn_m9c_out.cpp`, `tb/tb_top_m9.cpp` | No | code exists but evidence missing | Missing M09 report/artifacts | Produce M09 checkpoint evidence pack (QKV/cache/liveness checkpoints) |
| M10 | SoftmaxApprox + attention completion | `include/SoftmaxApprox.h`, attention full-path code and TBs (`tb_attn_m9*`, `tb_softmax_m15a.cpp`) | No | code exists but evidence missing | Missing M10 formal evidence pack; no M10-labeled closure proof | Produce M10 report/artifacts and document current softmax path versus v12.1 direction |
| M11 | FFN + residual + reusable TransformerLayer | `src/blocks/FFNLayer0.h`, `src/blocks/TransformerLayer.h`, `tb/tb_ffn_m10*.cpp`, `tb/tb_layerloop_m11.cpp` | No | code exists but evidence missing | Missing M11 formal closure pack | Create M11 evidence pack with FFN/layer-loop checkpoints |
| M12 | Multi-layer scheduling + mid/end LN under single-X_WORK overwrite/liveness rules | `tb/tb_mid_end_ln_m12.cpp` exists; `src/Top.h` uses alternate page base and out-of-place mid/end LN; `include/SramMap.h` dual-page map | No | spec drift / needs re-baseline | Runtime schedule still relies on dual-page assumptions conflicting with single-X_WORK formal baseline | Re-baseline schedule/storage semantics, then generate M12 closure evidence |
| M13 | FinalHead official Pass A/B semantics with `FINAL_SCALAR_BUF`, no PAGE_NEXT/TEMP_PAGE staging | `src/blocks/FinalHead.h` exists but is placeholder-style (`xor`-like logits/x_pred path), no explicit Pass A/B + `FINAL_SCALAR_BUF` semantics; `tb/tb_top_end2end_m13.cpp` exists | No | not done | Core M13 algorithm contract is not implemented to latest definition; no M13 evidence pack | Implement formal M13 behavior and then produce M13 report/artifacts |
| M14 | Top end-to-end regression + overlap basic verification with outmode coverage | `tb/tb_regress_m14.cpp` has cases incl. outmode/overlap/determinism; trace-dependent checks are skipped when `AECCT_HAS_TRACE=0` | No | code exists but evidence missing | No `M14_report.md`/`M14_artifacts`; latest expected coverage not formally evidenced | Produce M14 closure pack with explicit outmode/overlap evidence policy (trace-on/trace-off) |
| M15 | Native AECCT quant freeze (scope/boundary documentation and SSOT convergence) | `include/QuantDesc.h`, `include/VerifyTolerance.h`, quant entries in `include/WeightStreamOrder.h`, `tb/tb_softmax_m15a.cpp` | No | code exists but evidence missing | Missing M15 freeze report/evidence; current evidence is fragmented and not milestone-scoped | Create M15 freeze closure docs + artifacts (boundary table, checks, verdict) |
| M16 | Offline generator integration for `SramMap`/`WeightStreamOrder`/native quant metadata | `docs/milestones/M16_report.md` + `M16_artifacts/*`; `tools/gen_headers.py`, `tools/run_m16_pipeline.ps1`; no `scripts/gen_sram_map.py` or `scripts/gen_weight_stream_order.py` | Yes | spec drift / needs re-baseline | Existing M16 evidence is compliance/synth-safe convergence oriented; does not cleanly match latest-plan generator-integration definition | Remap current closure as compliance pre-M17 item, then close latest-plan M16 with generator-integration evidence |

## Evidence Gaps

1. Missing milestone evidence packs:
   - `M00`, `M01`, `M02`, `M06`, `M07`, `M08`, `M09`, `M10`, `M11`, `M12`, `M13`, `M14`, `M15`
2. Scope-coverage mismatch within existing pack:
   - `M05`: pack exists but does not fully demonstrate the latest expected full scope in one closure artifact set
3. Drift-sensitive closure missing:
   - `M02`, `M12`, `M16` (single-X_WORK/generator-definition impact)
4. FinalHead formal closure missing:
   - `M13`: no formal Pass A/B + `FINAL_SCALAR_BUF` milestone evidence
5. Quant freeze closure missing:
   - `M15`: no dedicated milestone freeze report/artifacts

## Spec-Drift / Renamed-Scope Items

1. Single-X_WORK baseline supersedes historical dual-page assumptions:
   - plan/spec/changelog now define single `X_WORK` formal scheduling baseline
   - current `include/SramMap.h` and parts of `src/Top.h` still encode `X_PAGE0/X_PAGE1` physical ping-pong semantics
2. M12 definition drift:
   - latest M12 expects `X_WORK` token-safe overwrite and liveness rules
   - current runtime remains page-alternating in top-level flow
3. M16 scope/name drift:
   - current `M16_report.md` is explicitly "P16 Compliance Convergence"
   - latest plan M16 requires offline generator integration closure content
4. FinalHead contract drift pressure:
   - latest v12.1 formalizes Pass A/B with `FINAL_SCALAR_BUF` and no next-page staging
   - current design-side FinalHead implementation does not reflect that formal structure

## Recommended Buckets

- PASS with evidence:
  - `M3`, `M4`
- Code exists but evidence missing:
  - `M0`, `M1`, `M5`, `M6`, `M7`, `M8`, `M9`, `M10`, `M11`, `M14`, `M15`
- Spec drift / needs re-baseline:
  - `M2`, `M12`, `M16`
- Not done:
  - `M13`

## Next Concrete Action

Start with a re-baseline closeout batch for `M2 + M12` (single-X_WORK alignment) and in parallel create missing formal evidence packs for already-implemented milestones (`M0`, `M1`, `M6`~`M11`, `M14`, `M15`) before attempting final milestone closure claims.
