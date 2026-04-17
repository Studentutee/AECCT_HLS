# Catapult Tcl Writing Note（zhTW）

## 文件定位
- 這份文件是 **AECCT 專案目前 corrected active chain 的 Catapult Tcl 寫法備忘錄**。
- 目的不是取代官方 user guide，也不是宣稱 repo 內現有 Tcl 已經在真 Catapult 環境完全驗證。
- 這份文件只固定四件事：
  - 目前 repo 在 Catapult 啟動時要抓哪個 synth entry。
  - 哪些 Tcl 指令骨架已和你提供的 `Catapult coding範本.zip` 內範本風格對齊。
  - 哪些 `options set ...` 寫法已由使用者在 Catapult GUI transcript 中實際觀察到。
  - 哪些指令/寫法目前仍屬 **待真機 compile 驗證**。

## 與 Catapult Shell Runbook 的關係
- 通用執行文件建議命名為 **Catapult Shell Runbook**，檔名仍可維持 `SKILL.md` 以相容 agent / loader 的固定入口。
- Runbook 的責任是：
  - 怎麼在遠端 shell 啟動 Catapult
  - 怎麼保存 console log / internal log / `messages.txt`
  - 怎麼 grep success / fail / blocker / warning
  - 怎麼用固定治理格式回報
- 本 note 的責任是：
  - 說明 **AECCT 這個專案** 的 project Tcl 應該怎麼寫
  - 固定 canonical synth top / entry TU / search path policy / compiler flag policy
- 一句話分工：
  - **Runbook 管怎麼跑**
  - **本 note 管 AECCT 的 Tcl 要跑什麼**

## 2026-04 執行模式差異經驗（cross-reference）
- 這次實機經驗確認：同一台主機、同一個 binary，`catapult -shell` 是否能 checkout license，會受到 shell 啟動模式影響。
- 非互動/非 login shell 可能出現：
  - `TERM=dumb`
  - `LM_LICENSE_FILE` 空值
  - `mgls_errno=515`
- 互動 login shell（例如 `ssh -tt`）可能出現：
  - 站點初始化完成
  - `LM_LICENSE_FILE` 正常
  - `LIC-13` / `LIC-14` 成功
- 這是**執行環境與 runbook 問題**，不是 Tcl 語法本身的設計規則。
- 因此本 note 只保留 cross-reference；具體檢查與處置流程請以 `skills/catapult_shell_runbook/SKILL.md` 為準。

## 目前專案的 canonical synth entry
- Canonical synth entry（經真機 transcript 修正後）：`aecct::TopManagedAttentionChainCatapultTop`
- `run` 仍是實際被 synthesize 的 interface method，但 `solution design set ... -top` 應以 class-level target 為主，而不是 `TopManagedAttentionChainCatapultTop::run`。
- 舊寫法 `TopManagedAttentionChainCatapultTop::run` 曾在真機 transcript 觸發 `HIER-55`，因此現在應視為 **deprecated target string**。
- 對應 entry TU：`src/catapult/p11as_top_managed_attention_chain_entry.cpp`
- 對應 filelist：`scripts/catapult/p11as_corrected_chain_filelist.f`
- 對應 launch runner：`scripts/catapult/run_p11as_corrected_chain_catapult_launch.ps1`
- 對應 launch note：`docs/handoff/P11_CORRECTED_CHAIN_CATAPULT_LAUNCH_NOTE.md`
- 若搭配通用 Runbook，建議在 AECCT override 中額外記錄：
  - `PROJECT_TCL=scripts/catapult/p11as_corrected_chain_project.tcl`
  - `PROJECT_ENTRY_DESC="corrected-chain compile-first project tcl"`
  - `TOP_TARGET="aecct::TopManagedAttentionChainCatapultTop"`
  - `ENTRY_TU="src/catapult/p11as_top_managed_attention_chain_entry.cpp"`

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

## 使用者在 Catapult GUI transcript 中已確認的 `options set` 寫法
以下寫法是使用者直接從 Catapult transcript 截圖提供的，可當成 **目前這台 Catapult 的 GUI/Tcl 對應語法證據**：

```tcl
options set Input/CppStandard c++20
options set Input/SearchPath /usr
options set Input/SearchPath /usr/tmp -append
options set Input/SearchPath {/usr/share/alsa /usr/share/akonadi} -append
options set Input/LibPaths /usr
options set Input/LibPaths /usr/share -append
options set Input/CompilerFlags <non-empty user flags only>
options set ComponentLibs/SearchPath /home/peter/Catapult_Library_Builder -append
options set ComponentLibs/TechLibSearchPath {/cad/PDK/CBDK_TSMC40_Arm_f2.0/CIC/SynopsysDC/db/sc9_base_hvt /cad/PDK/CBDK_TSMC40_Arm_f2.0/CIC/SynopsysDC/db/sc9_base_lvt /cad/PDK/CBDK_TSMC40_Arm_f2.0/CIC/SynopsysDC/db/sc9_base_rvt}
options set Flows/QuestaSIM/Path /cad/mentor/Questa_Sim/2025.2_2/questasim/bin
options set Flows/DesignCompiler/Path /cad/synopsys/synthesis/2025.06/bin/
options set Flows/VSCode/INSTALL /usr/bin/
options set Flows/VSCode/GDB_PATH /usr/bin/
```

### 中文解釋
- `Input/CompilerFlags` 的舊建議值 `-D__SYNTHESIS__` 已被真機 transcript 推翻：`__SYNTHESIS__` 在 Catapult project Tcl 中不可 user-define。
- 因此目前 corrected-chain 的 Catapult Tcl 應採：**define macro list 可空；只有在存在其他非保留的 user flags 時，才設定 `Input/CompilerFlags`**。
- `Input/CppStandard`：設定 Catapult 的 C++ 標準版本。以目前使用者回報，**先以 `c++20` 為專案預設嘗試值**。
- `Input/SearchPath`：設定 include / source 搜尋路徑；可單筆設定，也可用 `-append` 持續追加。
- `Input/LibPaths`：設定一般 library 搜尋路徑；compile-first draft 可先帶入。
- `Input/CompilerFlags`：可用來承載其他非保留的 user-defined compiler flags；對 corrected-chain Catapult Tcl 而言，define macro list 可空。
- `ComponentLibs/SearchPath`、`ComponentLibs/TechLibSearchPath`：屬於 component / tech library 路徑，通常站點相依。
- `Flows/.../Path`：屬於外部工具整合路徑，非最小 compile 必要，但 GUI transcript 已顯示其 option key。

## 目前已對齊範本或 GUI 證據的項目
這些項目可以視為「方向正確，值得保留」：
- 使用 `project new` 建立 Catapult project。
- 以 `solution file add ... -type C++` 載入 C++ source。
- 以 `solution design set <top> -top` 指定 synth top。
- 以 `go compile` 做最小 compile 驗證。
- 使用 `options set Input/CppStandard ...` 設定 C++ 標準。
- 使用 `options set Input/SearchPath ...` / `-append` 設定 include/source 搜尋路徑。
- 使用 `options set Input/LibPaths ...` 設定 library 路徑。
- `Input/CompilerFlags` 僅在存在其他非保留 user flags 時才設定；**不可**在 Catapult project Tcl 中手動加 `-D__SYNTHESIS__`。
- 需要更完整 synthesis flow 時，再補 component libs / tech libs / library / clock / assembly / architect / extract。

## 目前**不要直接當成已驗證寫法**的項目
下列項目目前 **不能** 因為 repo 內已有 draft Tcl 就當成已驗證：
- `solution new <name>`
- `solution file add ... -cflags ...`
- `go elaborate`
- `go architecture`

原因不是說它們一定錯，而是：
- 在你這次提供的 Catapult training 範本與 GUI transcript 中，我們沒有找到足夠直接的基線證據。
- 因此它們目前只能算 **launch-pack draft 寫法**，尚未完成「命令級相容性」驗證。

## AECCT corrected-chain 建議的最小安全 Tcl 骨架（v4）
如果你要先在 GUI 或真 Catapult 機器上做 **最小 compile 驗證**，優先用下面這種保守骨架：

```tcl
set sfd [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $sfd ".." ".."]]

options defaults
project new

options set Input/CppStandard c++20
options set Input/SearchPath $repo_root
options set Input/SearchPath [file join $repo_root "include"] -append
options set Input/SearchPath [file join $repo_root "src"] -append
options set Input/SearchPath [file join $repo_root "gen" "include"] -append
options set Input/SearchPath [file join $repo_root "third_party" "ac_types"] -append
options set Input/SearchPath [file join $repo_root "data" "weights"] -append
if {$p11as_compiler_flags ne ""} {
    options set Input/CompilerFlags $p11as_compiler_flags
}

solution file add [file join $repo_root src catapult p11as_top_managed_attention_chain_entry.cpp] -type C++
solution design set aecct::TopManagedAttentionChainCatapultTop -top

go compile
```

### 中文說明
- `options defaults`：先把專案選項重設到預設狀態，避免被舊 project/session 殘留值污染。
- `project new`：建立新 project。
- `Input/CppStandard`：先用 `c++20`，因為使用者明確表示目前本地端也是用 `C++20`。
- `Input/SearchPath`：這一步非常重要，因為 entry TU 會 include repo 內 header；**不要只把 include dir 放在 Tcl 變數裡卻不真正餵給工具**。
- `Input/CompilerFlags`：只在非空時才設；**不要**在 Catapult Tcl 中 user-define `__SYNTHESIS__`。
- `solution file add`：加入設計來源檔；這裡先用最小 entry TU 驗證 flow 能不能起來。
- `solution design set ... -top`：指定真正要綜合的 class-level top；目前 corrected-chain 應指向 `aecct::TopManagedAttentionChainCatapultTop`。
- `go compile`：先做最基本的 compile；這一步通過後，再談後續 library / assembly / architect。

## `Input/SearchPath` / `Input/LibPaths` 的實作筆記
- 單一路徑可直接：`options set Input/SearchPath /path/to/dir`
- 後續追加可用：`options set Input/SearchPath /another/path -append`
- 也可一次追加 brace-list：`options set Input/SearchPath {/p1 /p2} -append`
- `Input/LibPaths` 也觀察到相同的單筆 + `-append` 模式。
- 對 AECCT 這種 repo-local corrected-chain compile-first draft，至少應先把 repo root、`include`、`src`、`gen/include`、必要的第三方 datatype/include 路徑補進 `Input/SearchPath`。

## `ComponentLibs/...` 與 `Flows/...` 的定位
- `ComponentLibs/SearchPath`、`ComponentLibs/TechLibSearchPath`：建議保留為 **站點相依（site-dependent）** 設定，不要把單一機器的絕對路徑硬寫成唯一值。
- `Flows/QuestaSIM/Path`、`Flows/DesignCompiler/Path`、`Flows/VSCode/...`：可記錄 option key，但不應在 corrected-chain compile-first 草案裡一開始就強綁。
- 若要放進 repo script，較佳做法是：**以環境變數或可覆寫參數帶入**，而不是直接硬碼到 Tcl。

## 如果你是在 Catapult GUI 裡手動操作
建議順序：
1. 開 Catapult。
2. 先設定 `Input/CppStandard`、`Input/SearchPath`、必要時的 `Input/LibPaths`。
3. 用 GUI 或 `File -> Run script` 載入最小 Tcl。
4. 確認 synth entry target 是 class-level `aecct::TopManagedAttentionChainCatapultTop`；`run` 仍是 class 內被綜合的方法。
5. 若站點需要，再補 technology/component library 路徑。
6. 先讓 `go compile` 成功，再往下補 `go libraries` / `go assembly` / `go architect`。

## 目前 repo 內 draft Tcl 的定位
`scripts/catapult/p11as_corrected_chain_project.tcl` 目前定位是：
- 已對齊 corrected-chain launch intent。
- 已對齊 filelist / entry / preflight / env probe 的 repo 內 launch-pack 需求。
- 應優先往 **`options set Input/...` 版的 compile-first Tcl** 收斂。
- **尚未** 對齊到「真 Catapult 機器上已驗證無誤」的正式 project Tcl。

換句話說：
- 它可以拿來當 launch-pack 草案與自動化入口。
- 但在真機驗證前，不能把它當成完全定稿的 Catapult project script。
- 若由通用 Runbook 啟動，`PROJECT_TCL` 就應指向這份檔案。

## 建議的驗證策略
1. **先驗最小 compile**
   - entry TU
   - top 設定
   - `Input/CppStandard`
   - `Input/SearchPath`
   - `Input/CompilerFlags`
2. **再看是否需要 library path / component libs**
   - `Input/LibPaths`
   - `ComponentLibs/SearchPath`
   - `ComponentLibs/TechLibSearchPath`
3. **最後才擴到更完整腳本**
   - 包含 project script 自動化、library / clock / assembly / architect、或更完整 flow 串接。

## 目前仍待使用者從 GUI / 真機補充確認的點
- corrected-chain 已確認 `solution design set` 應採 class-level target；但 include graph 內仍可見 multiple tops warning，尚未代表 top-related 問題全數收斂。
- 若一次塞多個 repo-local search path，實機是否偏好 repeated `-append`，還是 brace-list 一次設定。
- corrected-chain compile 第一關是否真的需要 `Input/LibPaths`，或只設 `Input/SearchPath` 即可先過。
- 當 top-target 與 `__SYNTHESIS__` policy 修正後，historical blocker 曾依序出現過 `HIER-55`、`CIN-249 (wq_top / wk_top / wv_top)`、`HIER-10`；這些目前應視為 **歷史收斂軌跡**，不應再覆蓋最新 transcript 判讀。
- 最新使用者提供的 compile-first transcript excerpt（2026-03-24）以關鍵字檢索未命中 `# Error` / `Compilation aborted`；目前可見輸出以 warning 為主。

## 2026-03-24 compile-first transcript refresh（latest shared excerpt）
- 最新使用者提供的 Catapult compile transcript excerpt 已顯示：
  - `solution design set aecct::TopManagedAttentionChainCatapultTop -top`
  - `go compile` 已真正啟動
  - 目前共享 excerpt 以關鍵字檢索未命中 `# Error` 或 `Compilation aborted`
- 目前可見 warning 主要分兩群：
  - `CRD-549 / CRD-111 / CRD-68 / CRD-1 / CRD-186`（多數來自 `third_party/ac_types` 模板實例化鏈）
  - `CIN-63` multiple tops warning（`TernaryLiveL0Wq/Wk/WvRowTop` 與 `TopManagedAttentionChainCatapultTop`）
- 這代表 corrected-chain compile-first 真機 bring-up 已從「持續追第一個 fatal」推進到「最新共享 excerpt 為 warning-only」；但這仍不等同 Catapult closure，也不等同 SCVerify closure。

## 目前已收斂 / 未收斂邊界
### 已收斂
- `solution design set ... -top` 的 corrected-chain target 不應再使用 `TopManagedAttentionChainCatapultTop::run`。
- Catapult project Tcl 不應 user-define `__SYNTHESIS__`。
- define macro list 可空；`Input/CompilerFlags` 只有非空才設。

### 尚未收斂
- include graph 中仍可能出現 multiple tops warning，這不應和 class-level target 修正混為一談。
- 最新共享 transcript excerpt 雖未檢出 fatal keyword，但仍應以實際 solution transcript 的最後幾行作為正式判讀依據；本 note 不宣稱問題已完全收斂。

## 目前文件姿態
- 這份文件是 reference note，不是 closure claim。
- 這份文件不宣稱 Catapult closure。
- 這份文件不宣稱 SCVerify closure。
- 這份文件只固定 corrected active chain 的 Tcl 寫法基線與驗證邊界。
