# RefModel Pure-FP16 Round 1

## Summary
- Start converting `AECCT_ac_ref` from hybrid fp32-container/fp16-roundtrip behavior to a pure-fp16 main compute path.
- Keep trace buffers as host-side `double` / `float` sources; cast only on ingress to the model.
- Do not change Top contract or DUT block graph in this round.

## Exact files changed
- `AECCT_ac_ref/include/RefE4M3Helpers.h`
- `AECCT_ac_ref/include/InvSqrtApprox.h`
- `AECCT_ac_ref/include/SoftmaxApprox.h`
- `AECCT_ac_ref/src/RefModel.cpp`

## Main forward progress
1. `RefModel.cpp` internal alias now uses `ref_fp16_t` for the main compute carrier.
2. Preproc path now matches the fp16 authoritative contract used by `tb_fp16_preproc_u16_trace_ref_compare`.
3. Generic helper islands (`inv_sqrt`, `softmax exp/rcp`, `e4m3 roundtrip`) were made fp16-compatible enough to compile under the pure-fp16 carrier.
4. `dump_2d` / `dump_3d` no longer touch tensor contents when dump is disabled.

## Evidence
- `tb_fp16_preproc_u16_trace_ref_compare`: PASS
- `AECCT_ac_ref/src/ref_main.cpp + RefModel.cpp`: compile PASS
- `tb_fp16_trace_chain_layer0_ref_compare`: still aborts in the composite full-chain run; this is not closed in this round.

## Known open item
- Composite layer0 compare still aborts with an AC finite-value assert after the pure-fp16 RefModel finishes its own `infer_step0()` path. This needs a second round to separate RefModel-side residual nonfinite islands from any DUT-side full-chain issue.
