# Catapult Shell Runbook

> 檔名仍建議保留為 `SKILL.md`，因為很多 agent / loader 會把它當成固定入口；但**對人閱讀與維護時**，建議把這份文件稱為 **Runbook** 或 **Execution Playbook**，不要只叫它「skill」。

## 文件定位

這份文件是 **Catapult shell execution runbook**，負責規範：

1. 如何在**遠端終端**啟動 Catapult
2. 如何保存 exact command / console log / internal log / `messages.txt`
3. 如何做最小必要的 success / failure / blocker / warning 檢查
4. 如何用固定治理格式回報結果

這份文件**不是**：

- project Tcl 寫法教學
- GUI 操作手冊
- SSH 金鑰發放文件
- 專案 closure 聲明

一句話分工：

- **Runbook 管怎麼跑**
- **Project Tcl 管跑什麼**

---

## 命名建議

### 對機器/框架的檔名
- `SKILL.md`

### 對人閱讀的正式名稱
- **Catapult Shell Runbook**

### 資料夾名稱建議
- `skills/catapult_shell_runbook/`

若想再短一點，也可以用：
- `catapult_shell_runner`
- `catapult_shell_playbook`

我最推薦的是：
- **目錄名**：`catapult_shell_runbook`
- **檔名**：`SKILL.md`
- **文件標題**：`Catapult Shell Runbook`

---

## 與專案 Tcl / 入口檔案的關係

這份 Runbook 不直接硬碼某個單一專案的 top、entry TU 或 Tcl 細節。
它透過 `project_override.env` 接收**本輪執行入口**，再去呼叫 project-specific Tcl。

### Runbook 的執行入口
- `project_override.env`

### 專案實際執行入口
- `PROJECT_TCL`

### 建議在 override 中額外記錄的說明欄位

```bash
PROJECT_NAME="my_project"
PROJECT_TCL="/abs/path/to/repo/scripts/catapult/my_project.tcl"
PROJECT_ENTRY_DESC="compile-first corrected-chain project tcl"
TOP_TARGET="my_namespace::MyTop"
ENTRY_TU="src/catapult/my_top_entry.cpp"
```

### 中文解釋
- `PROJECT_TCL`：Catapult 本輪真正要跑的 Tcl 主入口。
- `PROJECT_ENTRY_DESC`：給人看的摘要，例如「corrected-chain compile-first project Tcl」。
- `TOP_TARGET`：方便在報告中寫清楚 canonical synth top。
- `ENTRY_TU`：方便在報告中寫清楚 entry translation unit。

這樣做的好處是：
- Runbook 可跨專案重用。
- project-specific 的 top / entry / Tcl 差異，交給 override 管理。
- 不會把 AECCT 專案規則硬寫死在通用 Runbook 內。

---

## 遠端伺服器連線前提

本 Runbook 假設執行環境**已經可以進入安裝 Catapult 的 Linux 伺服器 shell**。

### 建議前提
- 使用 SSH 連線到遠端伺服器
- 建議使用公鑰驗證（public key authentication）
- 可使用 `ssh-ed25519` 類型金鑰
- 使用者需先自行確認遠端登入可用，再啟動本 Runbook

### 本 Runbook 不負責
- 產生 SSH 金鑰
- 發放或保存私鑰
- 設定首次主機信任（known_hosts onboarding）
- 代替使用者完成伺服器 onboarding

### 邊界說明
一旦進入遠端 shell，本 Runbook 才開始負責：
- 檢查 Catapult binary
- 檢查 repo root / project Tcl
- 建立 outdir
- 用 shell mode 啟動 Catapult
- 保存 console log / internal log
- 搜尋並檢查最新 `messages.txt`

### optional：外層 wrapper 可記錄的遠端欄位

```bash
REMOTE_HOST="hls-server"
REMOTE_USER="peter"
REMOTE_REPO_ROOT="/abs/path/on/server/to/repo"
```

> 這些欄位可供**外層 launcher** 使用；核心 Runbook 不應假設自己負責 SSH 登入。

---

## 互動模式與 License 環境守則（2026-04 經驗）

在同一台主機上，`catapult -shell` 的 license 結果可能因為「執行模式」不同而不同：

- 非互動、非 login shell（例如單純 `ssh ... "bash -s"`）常見現象：
  - `TERM=dumb`
  - `LM_LICENSE_FILE` / `MGLS_LICENSE_FILE` / `CDS_LIC_FILE` 為空
  - 出現 `mgls_errno = 515`、`License server machine is down or not responding`
- 互動 login shell（例如 `ssh -tt` + 站點預設 shell 初始化）常見現象：
  - `TERM=xterm-256color`（或其他互動終端值）
  - `LM_LICENSE_FILE` 被站點初始化腳本正確設置
  - 可看到 `LIC-13` / `LIC-14` 成功訊息

因此遇到 license 失敗時，先判定是否為**環境模式不一致**，不要直接定性為 license server 故障。

最低限度必記錄：

```bash
echo "SHELL=$SHELL"
echo "TERM=${TERM:-}"
echo "LM_LICENSE_FILE=${LM_LICENSE_FILE:-}"
echo "MGLS_LICENSE_FILE=${MGLS_LICENSE_FILE:-}"
echo "CDS_LIC_FILE=${CDS_LIC_FILE:-}"
```

若上述 license 變數都為空，且本輪不是互動 login shell，應先改用互動 login 模式重跑，再做 blocker 結論。

---

## 官方依據（Catapult 2025.3）

以下是這份 Runbook 採用 shell flow 的依據：

- Catapult 支援 GUI 與 command-line/scripts 兩種互動方式，適合用 Tcl script 在命令列或 batch mode 執行。
- `catapult` 指令支援 `-shell`、`-file <Tcl_script_pathname>`、`-logfile <logfile_pathname>`。
- 在 interactive command line shell 中，可用 `source <script>` 或 `dofile <script>`；`dofile` 會把 Tcl 命令送到標準輸出，方便 debug。
- 以 shell 啟動時，可直接用 `-file` source Tcl script。
- `application report -transcript true` 會把報告送到 transcript。
- Catapult 會在 solution 目錄建立/更新 `messages.txt`，可作為系統訊息檢查依據。

> 建議在專案內保留對應的 user guide / reference manual 章節頁碼，但不要把 chat 專用 citation 語法硬寫進這份可移植文件。

---

## 嚴格規範

### 必守原則
- **不可依賴 GUI 點擊**
- **優先使用固定絕對路徑 binary**：
  `/cad/mentor/Catapult/2025.3/Mgc_home/bin/catapult`
- **不可只寫 `catapult`**，除非先驗證 `PATH` 中解析出的版本就是同一支 binary
- **不可 overclaim**
- **沒有 exact commands 與 actual log excerpt，不可宣稱 stage PASS**
- 若發生以下任一情況，必須停止並誠實回報，不得宣稱 PASS：
  - launch 失敗
  - `command not found`
  - license checkout / feature 失敗
  - Tcl error
  - `Compilation aborted`
  - `messages.txt` 中存在 `# Error`
- 若本輪是 **run-only audit**，則 `Exact files changed` 必須明寫：`無`
- 若本輪是 **patch + rerun**，必須明列每一個實際修改檔案

### 模式
- **run-only audit**：只執行、只收證據、不改檔
- **patch + rerun**：允許小幅修改後重跑，但仍需逐項列出改檔與重跑證據

---

## 建議目錄結構

```text
<repo>/
  build/
    catapult/
      <run_tag>/
        catapult_console.log
        catapult_internal.log
        catapult_version.txt
        command.env
        exact_command.sh
        grep_summary.txt
        grep_blockers.txt
        grep_warnings.txt
        grep_messages_tail.txt
        report_stub.md
```

若這份 Runbook 要被 agent / tool loader 直接讀取，建議整體放在：

```text
skills/
  catapult_shell_runbook/
    SKILL.md
    templates/
      catapult_shell_run.sh
      project_override.env.example
```

---

## Project-specific override 區塊

每個專案都要先準備一個 override 檔，例如：`project_override.env`。

可直接從 `templates/project_override.env.example` 複製。

### 必填欄位

```bash
MODE="run-only"                      # run-only | patch-rerun
REPO_ROOT="/abs/path/to/repo"
PROJECT_NAME="my_project"
PROJECT_TCL="/abs/path/to/repo/scripts/catapult/my_project.tcl"
RUN_TAG="my_project_$(date +%Y%m%d_%H%M%S)"
CATAPULT_OUTDIR="/abs/path/to/repo/build/catapult/${RUN_TAG}"
```

### 建議補充欄位

```bash
PROJECT_ENTRY_DESC="compile-first project tcl"
TOP_TARGET="my_namespace::MyTop"
ENTRY_TU="src/catapult/my_top_entry.cpp"
```

### 可選欄位

```bash
CATAPULT_BIN="/cad/mentor/Catapult/2025.3/Mgc_home/bin/catapult"
ALLOW_PATH_CATAPULT=0

# 找 messages.txt 時額外搜尋的根目錄
MESSAGE_SEARCH_ROOTS=(
  "/abs/path/to/repo/build"
  "/abs/path/to/repo/out"
)

# 呼叫者指定 blocker / warning 關鍵字
BLOCKER_KEYWORDS=(
  "license"
  "Segmentation fault"
  "invalid command name"
)

WARNING_KEYWORDS=(
  "Warning"
  "loop not unrolled"
)

# optional：外層 launcher 用的遠端資訊
REMOTE_HOST=""
REMOTE_USER=""
REMOTE_REPO_ROOT=""

# optional：執行前後 hook
PRE_RUN_HOOK=""
POST_RUN_HOOK=""
```

---

## 預設 shell command 模板

```bash
export CATAPULT_BIN="/cad/mentor/Catapult/2025.3/Mgc_home/bin/catapult"
export REPO_ROOT="<abs_repo_root>"
export PROJECT_TCL="<abs_repo_root>/scripts/catapult/<project>.tcl"
export CATAPULT_OUTDIR="<abs_repo_root>/build/catapult/<run_tag>"
test -x "$CATAPULT_BIN"
test -d "$REPO_ROOT"
test -f "$PROJECT_TCL"
mkdir -p "$CATAPULT_OUTDIR"
cd "$REPO_ROOT"
"$CATAPULT_BIN" -shell -file "$PROJECT_TCL" -logfile "$CATAPULT_OUTDIR/catapult_internal.log" 2>&1 | tee "$CATAPULT_OUTDIR/catapult_console.log"
```

---

## 最小 transcript / messages 檢查模板

```bash
grep -c '# Error' <messages.txt>
grep -c 'Compilation aborted' <messages.txt>
grep -c "Completed transformation 'compile'" <messages.txt>
grep -n '# Error\|Compilation aborted\|Completed transformation' <messages.txt> | tail -n 50
```

---

## Codex 執行流程

### Step 0：讀 override
- 先讀 `project_override.env`
- 若缺必填欄位，停止並回報缺少哪些欄位
- 回報時應帶出：`PROJECT_TCL`、`PROJECT_ENTRY_DESC`、`TOP_TARGET`、`ENTRY_TU`（若有提供）

### Step 1：binary 驗證
- 預設直接使用：
  `/cad/mentor/Catapult/2025.3/Mgc_home/bin/catapult`
- 只有在 `ALLOW_PATH_CATAPULT=1` 時，才允許用 `command -v catapult`
- 但即使允許，也必須驗證：
  - `command -v catapult` 不為空
  - `readlink -f "$(command -v catapult)"` 與固定絕對路徑相同
- 否則回退到固定絕對路徑

### Step 2：preflight
必須執行並保存證據：

```bash
test -x "$CATAPULT_BIN"
test -d "$REPO_ROOT"
test -f "$PROJECT_TCL"
mkdir -p "$CATAPULT_OUTDIR"
"$CATAPULT_BIN" -version | tee "$CATAPULT_OUTDIR/catapult_version.txt"
echo "SHELL=$SHELL" | tee -a "$CATAPULT_OUTDIR/catapult_version.txt"
echo "TERM=${TERM:-}" | tee -a "$CATAPULT_OUTDIR/catapult_version.txt"
echo "LM_LICENSE_FILE=${LM_LICENSE_FILE:-}" | tee -a "$CATAPULT_OUTDIR/catapult_version.txt"
echo "MGLS_LICENSE_FILE=${MGLS_LICENSE_FILE:-}" | tee -a "$CATAPULT_OUTDIR/catapult_version.txt"
echo "CDS_LIC_FILE=${CDS_LIC_FILE:-}" | tee -a "$CATAPULT_OUTDIR/catapult_version.txt"
```

若任何一步失敗，停止並回報，不可進入 PASS。
若 license 變數全空且執行模式為非互動 shell，先改用互動 login shell 重跑，再決定是否屬於真正 license blocker。

### Step 3：保存 exact command
把實際要執行的命令完整寫入：
- `$CATAPULT_OUTDIR/exact_command.sh`

內容必須是**可重播**的完整 shell 指令，而不是摘要。

### Step 4：執行 Catapult shell run
固定使用：

```bash
"$CATAPULT_BIN" -shell -file "$PROJECT_TCL" -logfile "$CATAPULT_OUTDIR/catapult_internal.log"
```

外層再用：

```bash
2>&1 | tee "$CATAPULT_OUTDIR/catapult_console.log"
```

並且 shell 必須開啟：

```bash
set -euo pipefail
```

### Step 5：搜尋最新 `messages.txt`
優先在：
1. `$CATAPULT_OUTDIR`
2. `MESSAGE_SEARCH_ROOTS[*]`
3. `$REPO_ROOT`

內搜尋 `messages.txt`，並用 **mtime 最新** 作為本輪判定目標。

若找不到 `messages.txt`：
- 不可直接宣稱 compile PASS
- 必須回報：Catapult console 有無成功訊號、`messages.txt` 缺失、因此只能做有限結論

### Step 6：grep 檢查
#### 固定 success / fail 關鍵字
- `# Error`
- `Compilation aborted`
- `Completed transformation 'compile'`

#### 額外 blocker / warning 關鍵字
- 由 `BLOCKER_KEYWORDS` / `WARNING_KEYWORDS` 提供
- 必須分開輸出到獨立檔案

### Step 7：停止條件
只要出現以下任何一項，就不得宣稱 PASS：
- Catapult 主命令 exit code 非 0
- console log 出現 license 失敗跡象
- console log 或 internal log 出現 Tcl error / invalid command / can't open / can't read
- `messages.txt` 中 `# Error` 計數 > 0
- `messages.txt` 中 `Compilation aborted` 計數 > 0
- 找不到可支撐 PASS 的 exact log excerpt

### Step 8：固定格式回報
回報必須包含下列區段，而且名稱固定：

#### Summary
- 說明本輪是 `run-only audit` 還是 `patch + rerun`
- 說明是 preflight fail、launch fail、compile fail、或 compile evidence present
- 若無法判定 PASS，要明說「無法宣稱 PASS 的原因」
- 若有提供 `PROJECT_ENTRY_DESC` / `TOP_TARGET` / `ENTRY_TU`，應在此區簡短帶出

#### Exact files changed
- 若為 run-only，明寫 `無`
- 若有 patch，逐一列出修改檔案

#### Exact commands run
- 逐條列出實際執行命令
- 不可只寫「執行 Catapult」

#### Actual execution evidence / log excerpt
- 貼出關鍵 log excerpt
- 至少包含：
  - 啟動 Catapult 的完整命令
  - compile 相關關鍵字
  - `# Error` / `Compilation aborted` 檢查結果
  - blocker / warning 檢查結果

#### Governance posture
- 明確區分：
  - compile-first PASS
  - warning-only
  - local-only
  - not Catapult closure
  - not SCVerify closure

---

## AECCT_HLS 專案的對接建議

若這份 Runbook 要服務 AECCT_HLS，建議在專案層另外保留：

```text
scripts/catapult/p11as_corrected_chain_project.tcl
scripts/catapult/overrides/p11as_project_override.env
docs/reference/Catapult_Tcl_Writing_Note_zhTW.md
```

分工如下：
- `SKILL.md`：通用執行 Runbook
- `p11as_corrected_chain_project.tcl`：AECCT 的 project Tcl 入口
- `p11as_project_override.env`：本輪執行參數
- `Catapult_Tcl_Writing_Note_zhTW.md`：AECCT 專案的 Tcl 寫法基線與邊界

---

## 禁止事項

- 不可把 GUI 點擊流程當成本 Runbook 的主流程
- 不可把單一專案的 top / entry TU / search path 硬寫成所有專案共用規則
- 不可在沒有證據時宣稱 compile / architect / extract PASS
- 不可把 SSH 私鑰、帳號密碼、站點專屬敏感資訊寫進 repo 版本控制
- 不可把 project Tcl 與 Runbook 合成同一份檔案；兩者應維持**弱耦合**而非混寫

---

## 最後建議

若你之後要交給別的 agent / 專案維護，請統一用下面這句話說明：

**這份 `SKILL.md` 是 Catapult Shell Runbook；它規範怎麼在遠端 shell 啟動、檢查、記錄與回報 Catapult 執行，但不取代 project-specific Tcl。**
