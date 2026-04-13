# REFMODEL_PUREFP16_NEW_ROUND2

## Summary
- Wrote a new single-mode pure-fp16 reference path by replacing the active `AECCT_ac_ref/src/RefModel.cpp` main carrier with `ac_std_float<16,5>` semantics throughout the core tensor flow.
- Kept the public `RefModel` API stable so existing ref-side callers do not need a new class name.
- Archived the prior mixed-precision implementation under `archive/refmodel_mixed_precision_legacy/AECCT_ac_ref/...` before rewriting.

## Exact files changed
- `archive/refmodel_mixed_precision_legacy/AECCT_ac_ref/include/RefModel.h`
- `archive/refmodel_mixed_precision_legacy/AECCT_ac_ref/include/InvSqrtApprox.h`
- `archive/refmodel_mixed_precision_legacy/AECCT_ac_ref/include/SoftmaxApprox.h`
- `archive/refmodel_mixed_precision_legacy/AECCT_ac_ref/include/RefE4M3Helpers.h`
- `archive/refmodel_mixed_precision_legacy/AECCT_ac_ref/src/RefModel.cpp`
- `archive/refmodel_mixed_precision_legacy/AECCT_ac_ref/src/ref_main.cpp`
- `AECCT_ac_ref/include/RefModel.h`
- `AECCT_ac_ref/src/RefModel.cpp`
- `tb/tb_refmodel_purefp16_smoke.cpp`
- `tb/tb_refmodel_purefp16_trace_compare.cpp`
- `tb/tb_refmodel_purefp16_trace_compare_batch16.cpp`

## What changed in the new active RefModel
1. Main carrier stays on `ac_std_float<16,5>` rather than `float`/`binary32` containers.
2. Legacy precision stress roundtrip helpers are disabled on the active path; the pure-fp16 ref path now treats those hooks as identity.
3. LayerNorm token processing was rewritten to use fp16 carrier arithmetic end-to-end in the main math path.
4. Model-boundary input conversion now consumes host-side `double` trace values and converts them once at entry.
5. Existing API surface (`RefModelIO`, `RefRunConfig`, `infer_step0`) remains intact for ref-side bring-up continuity.

## Exact commands run
```bash
cd /mnt/data/purefp16_new/AECCT_HLS-backup-fp16-io8-inline-ln1p

g++ -std=c++17 -O0 -I. -IAECCT_ac_ref/include -Iinclude -Igen -Igen/include -Ithird_party/ac_types -I/mnt/data \
  AECCT_ac_ref/src/ref_main.cpp AECCT_ac_ref/src/RefModel.cpp \
  -o build/refmodel_newpure/ref_main

g++ -std=c++17 -O2 -I. -IAECCT_ac_ref/include -Iinclude -Igen -Igen/include -Ithird_party/ac_types -I/mnt/data \
  tb/tb_refmodel_purefp16_smoke.cpp AECCT_ac_ref/src/RefModel.cpp \
  -o build/refmodel_newpure/tb_refmodel_purefp16_smoke_repo

g++ -std=c++17 -O2 -I. -IAECCT_ac_ref/include -Iinclude -Igen -Igen/include -Ithird_party/ac_types -I/mnt/data \
  tb/tb_refmodel_purefp16_trace_compare.cpp AECCT_ac_ref/src/RefModel.cpp \
  -o build/refmodel_newpure/tb_refmodel_purefp16_trace_compare_repo

g++ -std=c++17 -O2 -I. -IAECCT_ac_ref/include -Iinclude -Igen -Igen/include -Ithird_party/ac_types -I/mnt/data \
  tb/tb_refmodel_purefp16_trace_compare_batch16.cpp AECCT_ac_ref/src/RefModel.cpp \
  -o build/refmodel_newpure/tb_refmodel_purefp16_trace_compare_batch16_repo

python3 scripts/check_design_purity.py
python3 scripts/check_repo_hygiene.py
```

## Execution evidence
### RefModel-only smoke
```text
[refmodel_purefp16_smoke] logits0=24.234375 logits1=26.078125 xpred0=0 xpred1=0
```

### Sample0 trace compare
```text
[refmodel_purefp16_trace_compare] xpred_mismatch=0 first_xpred=-1 max_logit_abs=1.995368958 max_logit_idx=49
```

### Batch16 trace compare
```text
[refmodel_purefp16_trace_compare_batch] B=16 total_xpred_mismatch=0 first=(-1,-1) max_logit_abs=28.106781006 max_logit=(1,35)
```

## Important scope notes
- This round validates the new RefModel with ref-only tests, not full design-side compare TBs.
- Existing design-side compare TB compilation is still blocked by unrelated active design headers, so that evidence is intentionally not overclaimed here.
- `check_design_purity.py` and `check_repo_hygiene.py` still fail due pre-existing repo/design conditions outside the narrow RefModel rewrite scope.

## Current status
- Pure-fp16 RefModel active path: **brought up**
- RefModel-only smoke: **PASS**
- Sample0 x_pred compare vs provided trace: **exact**
- Batch16 x_pred compare vs provided trace: **exact**
- Full evaluator / BER-FER closure: **not run in this round**
