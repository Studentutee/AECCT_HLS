# PREPROC_MAINLINE_REHOME_ROUND2

## 一句話
本輪把 `PreprocEmbedSPE` 的舊版本封存到 `archive/src_legacy_fp32bridge/blocks/PreprocEmbedSPE.h`，並把新的 fp16 bring-up 主線放回原本檔名 `src/blocks/PreprocEmbedSPE.h`。

## 這輪做了什麼
- 新增 archive 版本：`archive/src_legacy_fp32bridge/blocks/PreprocEmbedSPE.h`
- active path 改回原檔名：`src/blocks/PreprocEmbedSPE.h`
- **沒有** 修改 `AECCT_ac_ref/include/RefModel.h`
- **沒有** 修改 `AECCT_ac_ref/src/RefModel.cpp`
- package 不包含 trace headers

## 為什麼這樣做
- 舊檔退出主線，但保留 reference
- active path 沿用原檔名，避免之後 include / filelist / handoff 持續雙軌
- RefModel 保持單一 authoritative math path，不為了 debug compare 另外複製 helper 計算鏈

## 目前狀態
- 這份 package 的重點是檔案重整與主線 re-home。
- 本輪**沒有**重新附帶 trace compare harness，也**沒有**新增 RefModel preproc-only helper。
- 若要做下一輪逐段 compare，建議改成：
  - 保持 `infer_step0()` 為唯一主計算路徑
  - 只增加 read-only tap / dump / wrapper
  - 不建立第二條 RefModel math path
