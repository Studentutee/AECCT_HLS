param(
    [string]$RunRoot = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($RunRoot -eq "") {
    $repoRoot = (Resolve-Path (Join-Path (Join-Path $PSScriptRoot "..\..\..") "..")).Path
    $RunRoot = Join-Path $repoRoot "build\\ref_v3\\blocks"
}
$RunRoot = [System.IO.Path]::GetFullPath($RunRoot)

if (-not (Test-Path -LiteralPath $RunRoot)) {
    throw "Run root not found: $RunRoot"
}

function Parse-KeyValueFile {
    param([string]$Path)
    $map = @{}
    if (-not (Test-Path -LiteralPath $Path)) {
        return $map
    }
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match "^[\\s#]") { continue }
        $idx = $line.IndexOf("=")
        if ($idx -lt 1) { continue }
        $k = $line.Substring(0, $idx).Trim()
        $v = $line.Substring($idx + 1).Trim()
        $map[$k] = $v
    }
    return $map
}

function Resolve-SearchFiles {
    param(
        [string]$WorkDir,
        [string]$HintPath
    )
    $files = New-Object System.Collections.Generic.List[string]
    if ($HintPath -ne "" -and (Test-Path -LiteralPath $HintPath)) {
        Get-ChildItem -LiteralPath $HintPath -File -Recurse -ErrorAction SilentlyContinue |
            ForEach-Object { $files.Add($_.FullName) }
    }
    if (Test-Path -LiteralPath $WorkDir) {
        Get-ChildItem -LiteralPath $WorkDir -File -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.Extension -in @(".rpt", ".txt", ".log", ".xml") } |
            ForEach-Object {
                if (-not $files.Contains($_.FullName)) {
                    $files.Add($_.FullName)
                }
            }
    }
    return $files
}

function Find-FirstMatchLine {
    param(
        [string[]]$Files,
        [string[]]$Patterns
    )
    foreach ($f in $Files) {
        $hit = Select-String -Path $f -Pattern $Patterns -SimpleMatch -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($null -ne $hit) {
            return @{ file = $f; line = $hit.Line.Trim() }
        }
    }
    return @{ file = ""; line = "" }
}

function Extract-FirstNumber {
    param([string]$Text)
    if ($Text -match "[-+]?\\d+(?:\\.\\d+)?(?:[eE][-+]?\\d+)?") {
        return [double]$matches[0]
    }
    return $null
}

$manifestFiles = Get-ChildItem -LiteralPath $RunRoot -Recurse -File -Filter "refv3_block_manifest.txt" -ErrorAction SilentlyContinue
$rows = New-Object System.Collections.Generic.List[object]

foreach ($mf in $manifestFiles) {
    $kv = Parse-KeyValueFile -Path $mf.FullName
    $workDir = if ($kv.ContainsKey("work_dir")) { $kv["work_dir"] } else { $mf.Directory.FullName }
    $files = Resolve-SearchFiles -WorkDir $workDir -HintPath ($(if ($kv.ContainsKey("report_hint_area")) { $kv["report_hint_area"] } else { "" }))

    $areaHit = Find-FirstMatchLine -Files $files -Patterns @("Total area", "Total Area", "Cell area", "Design area", "Area:")
    $powerHit = Find-FirstMatchLine -Files $files -Patterns @("Total power", "Total Power", "Power:")
    $perfHit = Find-FirstMatchLine -Files $files -Patterns @("Latency", "II", "Initiation Interval", "Throughput", "Interval")

    $areaNum = Extract-FirstNumber -Text $areaHit.line
    $powerNum = Extract-FirstNumber -Text $powerHit.line
    $latNum = Extract-FirstNumber -Text $perfHit.line

    $rows.Add([PSCustomObject]@{
            block_name = $(if ($kv.ContainsKey("block_name")) { $kv["block_name"] } else { "unknown" })
            project_name = $(if ($kv.ContainsKey("project_name")) { $kv["project_name"] } else { "unknown" })
            top_name = $(if ($kv.ContainsKey("top_name")) { $kv["top_name"] } else { "unknown" })
            stage_compile = $(if ($kv.ContainsKey("stage_compile")) { $kv["stage_compile"] } else { "UNKNOWN" })
            stage_extract = $(if ($kv.ContainsKey("stage_extract")) { $kv["stage_extract"] } else { "UNKNOWN" })
            stage_project_save = $(if ($kv.ContainsKey("stage_project_save")) { $kv["stage_project_save"] } else { "UNKNOWN" })
            area_line = $areaHit.line
            area_file = $areaHit.file
            area_numeric = $areaNum
            power_line = $powerHit.line
            power_file = $powerHit.file
            power_numeric = $powerNum
            perf_line = $perfHit.line
            perf_file = $perfHit.file
            perf_numeric = $latNum
            manifest_path = $mf.FullName
            work_dir = $workDir
            messages_path = $(if ($kv.ContainsKey("messages_path")) { $kv["messages_path"] } else { "" })
            report_hint_area = $(if ($kv.ContainsKey("report_hint_area")) { $kv["report_hint_area"] } else { "" })
            report_hint_power = $(if ($kv.ContainsKey("report_hint_power")) { $kv["report_hint_power"] } else { "" })
            report_hint_perf = $(if ($kv.ContainsKey("report_hint_perf")) { $kv["report_hint_perf"] } else { "" })
        })
}

$csvPath = Join-Path $RunRoot "block_metrics_summary.csv"
$mdPath = Join-Path $RunRoot "block_metrics_summary.md"
$estPath = Join-Path $RunRoot "system_level_estimate.md"

if ($rows.Count -eq 0) {
    @(
        "No block manifests found under: $RunRoot",
        "posture: not Catapult closure",
        "posture_scverify: not SCVerify closure"
    ) | Set-Content -Path $mdPath -Encoding Ascii
    "" | Set-Content -Path $csvPath -Encoding Ascii
    @(
        "System estimate unavailable: no block manifests found.",
        "posture: not Catapult closure",
        "posture_scverify: not SCVerify closure"
    ) | Set-Content -Path $estPath -Encoding Ascii
    Write-Host "No manifests found. Summary generated with empty content." -ForegroundColor Yellow
    exit 0
}

$rows | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8

$areaKnown = $rows | Where-Object { $_.area_numeric -ne $null }
$powerKnown = $rows | Where-Object { $_.power_numeric -ne $null }
$perfKnown = $rows | Where-Object { $_.perf_numeric -ne $null }

$areaSum = ($areaKnown | Measure-Object -Property area_numeric -Sum).Sum
$powerSum = ($powerKnown | Measure-Object -Property power_numeric -Sum).Sum
$perfBottleneck = $null
if ($perfKnown.Count -gt 0) {
    $perfBottleneck = $perfKnown | Sort-Object -Property perf_numeric -Descending | Select-Object -First 1
}

$overheadRatio = 0.15
$areaTotalEst = if ($areaKnown.Count -gt 0) { $areaSum * (1.0 + $overheadRatio) } else { $null }
$powerTotalEst = if ($powerKnown.Count -gt 0) { $powerSum * (1.0 + $overheadRatio) } else { $null }

$mdLines = New-Object System.Collections.Generic.List[string]
$mdLines.Add("# REF_V3 Block Metrics Summary")
$mdLines.Add("")
$mdLines.Add("| Block | Project | Top | Compile | Extract | Save | Area | Power | Throughput/Latency/II |")
$mdLines.Add("|---|---|---|---|---|---|---|---|---|")
foreach ($r in $rows) {
    $mdLines.Add(("{0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} |" -f `
            $r.block_name, `
            $r.project_name, `
            $r.top_name, `
            $r.stage_compile, `
            $r.stage_extract, `
            $r.stage_project_save, `
            $(if ($r.area_line -ne "") { $r.area_line } else { "N/A" }), `
            $(if ($r.power_line -ne "") { $r.power_line } else { "N/A" }), `
            $(if ($r.perf_line -ne "") { $r.perf_line } else { "N/A" })))
}
$mdLines.Add("")
$mdLines.Add("posture: not Catapult closure")
$mdLines.Add("posture_scverify: not SCVerify closure")
$mdLines | Set-Content -Path $mdPath -Encoding Ascii

$estLines = New-Object System.Collections.Generic.List[string]
$estLines.Add("# REF_V3 System-Level Estimate (Block-Based)")
$estLines.Add("")
$estLines.Add("- Estimate basis: block metrics under run root")
$estLines.Add("- Formula: Area_total_est ~= sum(block_area) + 15% glue/interconnect/memory overhead")
$estLines.Add("- Formula: Power_total_est ~= sum(block_power) + 15% overhead")
$estLines.Add("- Throughput bottleneck heuristic: largest parsed latency/II-related scalar")
$estLines.Add("")
$estLines.Add(("- Known area blocks: {0}" -f $areaKnown.Count))
$estLines.Add(("- Known power blocks: {0}" -f $powerKnown.Count))
$estLines.Add(("- Known perf blocks: {0}" -f $perfKnown.Count))
$estLines.Add(("- Area_total_est: {0}" -f $(if ($areaTotalEst -ne $null) { "{0}" -f $areaTotalEst } else { "N/A" })))
$estLines.Add(("- Power_total_est: {0}" -f $(if ($powerTotalEst -ne $null) { "{0}" -f $powerTotalEst } else { "N/A" })))
if ($null -ne $perfBottleneck) {
    $estLines.Add(("- Perf bottleneck candidate: {0} ({1})" -f $perfBottleneck.block_name, $perfBottleneck.perf_line))
} else {
    $estLines.Add("- Perf bottleneck candidate: N/A")
}
$estLines.Add("")
$estLines.Add("posture: not Catapult closure")
$estLines.Add("posture_scverify: not SCVerify closure")
$estLines | Set-Content -Path $estPath -Encoding Ascii

Write-Host "Metrics CSV: $csvPath" -ForegroundColor Green
Write-Host "Metrics MD:  $mdPath" -ForegroundColor Green
Write-Host "Estimate MD: $estPath" -ForegroundColor Green
