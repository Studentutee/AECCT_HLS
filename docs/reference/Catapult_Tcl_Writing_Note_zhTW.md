# Catapult Tcl Writing Note（zhTW）

## 文件定位
- 這份文件是 **AECCT 專案目前 corrected active chain 的 Catapult Tcl 寫法備忘錄**。
- 目的不是取代官方 user guide，也不是宣稱 repo 內現有 Tcl 已經在真 Catapult 環境完全驗證。
- 這份文件只固定三件事：
  - 目前 repo 在 Catapult 啟動時要抓哪個 synth entry。
  - 哪些 Tcl 指令骨架已和你提供的 `Catapult coding範本.zip` 內範本風格對齊。
  - 哪些指令/寫法目前仍屬 **待真機驗證**。

## 目前專案的 canonical synth entry
- Canonical synth entry：`TopManagedAttentionChainCatapultTop::run`
- 對應 entry TU：`src/catapult/p11as_top_managed_attention_chain_entry.cpp`
- 對應 filelist：`scripts/catapult/p11as_corrected_chain_filelist.f`
- 對應 launch runner：`scripts/catapult/run_p11as_corrected_chain_catapult_launch.ps1`
- 對應 launch note：`docs/handoff/P11_CORRECTED_CHAIN_CATAPULT_LAUNCH_NOTE.md`

## 你提供的 Catapult 範本中，可直接拿來當基線的 Tcl 風格
以下骨架在你提供的範本中有明確對應，可視為目前最保守、最安全的基線：

```tcl
set sfd [file dirname [info script]]
options defaults
project new
solution file add <source> -type C++
solution design set <top> -top
go compile
```

若要往後走到 library / clock / architecture 相關步驟，範本常見的順序是：

```tcl
solution library add <library>
go libraries
directive set -CLOCKS {clk {-CLOCK_PERIOD <period> }}
go assembly
go architect
go extract
```

## 目前已對齊範本風格的項目
這些項目可以視為「方向正確，值得保留」：
- 使用 `project new` 建立 Catapult project。
- 以 `solution file add ... -type C++` 載入 C++ source。
- 以 `solution design set <top> -top` 指定 synth top。
- 以 `go compile` 做最小 compile 驗證。
- 需要更完整 synthesis flow 時，再補 library / clock / assembly / architect / extract。

## 目前**不要直接當成已驗證寫法**的項目
下列項目目前 **不能** 因為 repo 內已有 draft Tcl 就當成已驗證：
- `solution new <name>`
- `solution file add ... -cflags ...`
- `go elaborate`
- `go architecture`

原因不是說它們一定錯，而是：
- 在你這次提供的 Catapult training 範本裡，我們沒有找到可直接對應的基線寫法。
- 因此它們目前只能算 **launch-pack draft 寫法**，尚未完成「命令級相容性」驗證。

## AECCT corrected-chain 建議的最小安全 Tcl 骨架
如果你要先在 GUI 或真 Catapult 機器上做 **最小 compile 驗證**，優先用下面這種保守骨架：

```tcl
set sfd [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $sfd ".." ".."]]

options defaults
project new

solution file add [file join $repo_root src catapult p11as_top_managed_attention_chain_entry.cpp] -type C++
solution design set TopManagedAttentionChainCatapultTop::run -top

go compile
```

### 中文說明
- `options defaults`：先把專案選項重設到預設狀態，避免被舊 project/session 殘留值污染。
- `project new`：建立新 project。
- `solution file add`：加入設計來源檔；這裡先用最小 entry TU 驗證 flow 能不能起來。
- `solution design set ... -top`：指定真正要綜合的 top。
- `go compile`：先做最基本的 compile；這一步通過後，再談後續 library / assembly / architect。

## 如果你是在 Catapult GUI 裡手動操作
建議順序：
1. 開 Catapult。
2. 用 GUI 或 `File -> Run script` 載入最小 Tcl。
3. 確認 synth entry 是 `TopManagedAttentionChainCatapultTop::run`。
4. 如果站點需要，再補 technology/library 路徑。
5. 先讓 `go compile` 成功，再往下補 `go libraries` / `go assembly` / `go architect`。

## 目前 repo 內 draft Tcl 的定位
`scripts/catapult/p11as_corrected_chain_project.tcl` 目前定位是：
- 已對齊 corrected-chain launch intent。
- 已對齊 filelist / entry / preflight / env probe 的 repo 內 launch-pack 需求。
- **尚未** 對齊到「真 Catapult 機器上已驗證無誤」的正式 project Tcl。

換句話說：
- 它可以拿來當 launch-pack 草案與自動化入口。
- 但在真機驗證前，不能把它當成完全定稿的 Catapult project script。

## 建議的驗證策略
1. **先驗最小 compile**
   - entry TU
   - top 設定
   - include path / macro / filelist
2. **再驗 library / clock / assembly**
   - 這一步通常帶有站點相依（site-dependent）設定。
3. **最後才擴到更完整腳本**
   - 包含 project script 自動化、更多 checker、或更完整 architecture/library flow。

## 目前文件姿態
- 這份文件是 reference note，不是 closure claim。
- 這份文件不宣稱 Catapult closure。
- 這份文件不宣稱 SCVerify closure。
- 這份文件只固定 corrected active chain 的 Tcl 寫法基線與驗證邊界。
