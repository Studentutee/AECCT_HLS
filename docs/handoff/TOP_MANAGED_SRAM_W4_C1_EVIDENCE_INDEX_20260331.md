# TOP_MANAGED_SRAM_W4_C1_EVIDENCE_INDEX_20260331

## Evidence Root
- `build/evidence/w4c1_contract_campaign_20260331/`

## Structural Gates
1. `powershell -ExecutionPolicy Bypass -File scripts/check_design_purity.ps1`
2. `powershell -ExecutionPolicy Bypass -File scripts/check_interface_lock.ps1`
3. `powershell -ExecutionPolicy Bypass -File scripts/check_macro_hygiene.ps1`
4. `powershell -ExecutionPolicy Bypass -File scripts/check_repo_hygiene.ps1 -Phase pre`
5. `powershell -ExecutionPolicy Bypass -File scripts/check_top_managed_sram_boundary_regression.ps1`
6. `powershell -ExecutionPolicy Bypass -File scripts/check_helper_channel_split_regression.ps1`
7. `powershell -ExecutionPolicy Bypass -File scripts/check_repo_hygiene.ps1 -Phase post`

Logs:
- `build/evidence/w4c1_contract_campaign_20260331/check_design_purity.log`
- `build/evidence/w4c1_contract_campaign_20260331/check_interface_lock.log`
- `build/evidence/w4c1_contract_campaign_20260331/check_macro_hygiene.log`
- `build/evidence/w4c1_contract_campaign_20260331/check_repo_hygiene_pre.log`
- `build/evidence/w4c1_contract_campaign_20260331/check_top_managed_sram_boundary_regression.log`
- `build/evidence/w4c1_contract_campaign_20260331/check_helper_channel_split_regression.log`
- `build/evidence/w4c1_contract_campaign_20260331/check_repo_hygiene_post.log`

## Targeted Probe (C1)
- Command:
  - `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11w4c1_softmaxout_head_token_contract_probe.ps1`
- Runner log:
  - `build/evidence/w4c1_contract_campaign_20260331/run_p11w4c1_softmaxout_head_token_contract_probe.log`
- TB raw run log:
  - `build/p11w4c1/softmaxout_head_token_contract_probe/run.log`
- Build log:
  - `build/p11w4c1/softmaxout_head_token_contract_probe/build.log`
- Verdict:
  - `build/p11w4c1/softmaxout_head_token_contract_probe/verdict.txt`
- Manifest:
  - `build/p11w4c1/softmaxout_head_token_contract_probe/file_manifest.txt`

Expected PASS lines from TB raw run log:
- `W4C1_SOFTMAXOUT_HEAD_TOKEN_CONTRACT_BRIDGE_VISIBLE PASS`
- `W4C1_SOFTMAXOUT_HEAD_TOKEN_CONTRACT_OWNERSHIP_CHECK PASS`
- `W4C1_SOFTMAXOUT_HEAD_TOKEN_CONTRACT_DESC_VISIBLE PASS`
- `W4C1_SOFTMAXOUT_HEAD_TOKEN_CONTRACT_CONSUME_COUNT_EXACT PASS`
- `W4C1_SOFTMAXOUT_HEAD_TOKEN_CONTRACT_EXPECTED_COMPARE PASS`
- `W4C1_SOFTMAXOUT_HEAD_TOKEN_CONTRACT_LEGACY_COMPARE PASS`
- `W4C1_SOFTMAXOUT_HEAD_TOKEN_CONTRACT_NO_SPURIOUS_TOUCH PASS`
- `W4C1_SOFTMAXOUT_HEAD_TOKEN_CONTRACT_MISMATCH_REJECT PASS`
- `W4C1_SOFTMAXOUT_HEAD_TOKEN_CONTRACT_ANTI_FALLBACK PASS`
- `PASS: tb_w4c1_softmaxout_head_token_contract_probe`

## Baseline Recheck
Commands:
1. `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11w4c0_softmaxout_contract_probe.ps1`
2. `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11w4b9_qkscore_longspan_bridge.ps1`
3. `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11w4b8_qkscore_family_fullhead_bridge.ps1`
4. `powershell -ExecutionPolicy Bypass -File scripts/local/run_p11w4b1_phaseb_tile_bridge.ps1`

Logs:
- `build/evidence/w4c1_contract_campaign_20260331/run_p11w4c0_softmaxout_contract_probe.log`
- `build/evidence/w4c1_contract_campaign_20260331/run_p11w4b9_qkscore_longspan_bridge.log`
- `build/evidence/w4c1_contract_campaign_20260331/run_p11w4b8_qkscore_family_fullhead_bridge.log`
- `build/evidence/w4c1_contract_campaign_20260331/run_p11w4b1_phaseb_tile_bridge.log`

## Highlight Index
- `build/evidence/w4c1_contract_campaign_20260331/evidence_highlights.txt`
- `build/evidence/w4c1_contract_campaign_20260331/authority_extract_20260331.txt`
- `build/evidence/w4c1_contract_campaign_20260331/command_manifest.txt`
- `build/evidence/w4c1_contract_campaign_20260331/verdict_summary.txt`
- `build/evidence/w4c1_contract_campaign_20260331/git_status_pre.txt`
- `build/evidence/w4c1_contract_campaign_20260331/git_status_post.txt`
