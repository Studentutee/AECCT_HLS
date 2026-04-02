param(
    [string]$RepoRoot = ".",
    [string]$OutDir = "build\p11anb_deeper",
    [ValidateSet("pre", "post")]
    [string]$Phase = "pre"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Join-RepoPath {
    param(
        [string]$RepoRootPath,
        [string]$Path
    )

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $RepoRootPath $Path))
}

function Get-RepoRelativePath {
    param(
        [string]$BasePath,
        [string]$TargetPath
    )
    $baseUri = New-Object System.Uri(([System.IO.Path]::GetFullPath($BasePath).TrimEnd('\') + '\'))
    $targetUri = New-Object System.Uri([System.IO.Path]::GetFullPath($TargetPath))
    return [System.Uri]::UnescapeDataString($baseUri.MakeRelativeUri($targetUri).ToString()).Replace('\', '/')
}

$repo = [System.IO.Path]::GetFullPath((Resolve-Path $RepoRoot).Path)
$outDirAbs = Join-RepoPath -RepoRootPath $repo -Path $OutDir
New-Item -ItemType Directory -Force -Path $outDirAbs > $null

$logPath = Join-Path $outDirAbs "check_p11anb_attnlayer0_deeper_boundary.log"
$summaryPath = Join-Path $outDirAbs "check_p11anb_attnlayer0_deeper_boundary_summary.txt"
if (-not (Test-Path $logPath)) {
    New-Item -ItemType File -Path $logPath -Force > $null
}

function Write-Log {
    param([string]$Message)
    Write-Host $Message
    Add-Content -Path $logPath -Value $Message -Encoding UTF8
}

function Write-Summary {
    param(
        [string]$Status,
        [string]$Detail
    )
    @(
        ("status: {0}" -f $Status),
        ("phase: {0}" -f $Phase),
        ("repo_root: {0}" -f $repo),
        ("out_dir: {0}" -f (Get-RepoRelativePath -BasePath $repo -TargetPath $outDirAbs)),
        ("log: {0}" -f (Get-RepoRelativePath -BasePath $repo -TargetPath $logPath)),
        ("detail: {0}" -f $Detail)
    ) | Set-Content -Path $summaryPath -Encoding UTF8
}

function Fail-Check {
    param([string]$Reason)
    Write-Log "FAIL: check_p11anb_attnlayer0_deeper_boundary"
    Write-Log $Reason
    Write-Summary -Status "FAIL" -Detail $Reason
    exit 1
}

function Require-Regex {
    param(
        [string]$Text,
        [string]$Pattern,
        [string]$Reason
    )
    if (-not ([System.Text.RegularExpressions.Regex]::IsMatch($Text, $Pattern, [System.Text.RegularExpressions.RegexOptions]::Singleline))) {
        Fail-Check $Reason
    }
}

Add-Content -Path $logPath -Value ("===== check_p11anb_attnlayer0_deeper_boundary phase={0} =====" -f $Phase) -Encoding UTF8
Write-Log ("[p11anb_deeper] phase={0}" -f $Phase)

$attnRel = "src/blocks/AttnLayer0.h"
$tbRel = "tb/tb_p11anb_attnlayer0_boundary_seam_contract.cpp"
$runnerRel = "scripts/local/run_p11anb_attnlayer0_boundary_seam_contract.ps1"

foreach ($rel in @($attnRel, $tbRel, $runnerRel)) {
    if (-not (Test-Path (Join-Path $repo $rel))) {
        Fail-Check ("required file missing: {0}" -f $rel)
    }
}

$attnText = Get-Content -Path (Join-Path $repo $attnRel) -Raw
$tbText = Get-Content -Path (Join-Path $repo $tbRel) -Raw
$runnerText = Get-Content -Path (Join-Path $repo $runnerRel) -Raw

Require-Regex -Text $attnText -Pattern '(?ms)struct\s+AttnLayer0PrebuiltHandoffDesc\s*\{[\s\S]*?out_topfed_payload_enable[\s\S]*?out_topfed_payload_words[\s\S]*?out_topfed_payload_words_valid' -Reason "AttnLayer0 prebuilt descriptor missing topfed OUT payload fields"
Require-Regex -Text $attnText -Pattern '(?ms)make_attn_layer0_prebuilt_handoff_desc\s*\([\s\S]*?bool\s+out_topfed_payload_enable[\s\S]*?const\s+u32_t\*\s+out_topfed_payload_words[\s\S]*?u32_t\s+out_topfed_payload_words_valid' -Reason "AttnLayer0 helper overload missing topfed OUT payload args"
Require-Regex -Text $attnText -Pattern '(?ms)if\s+constexpr\s*\(\s*STAGE_MODE\s*==\s*ATTN_STAGE_OUT[\s\S]*?out_topfed_ready[\s\S]*?ATTN_OUT_TOPFED_PAYLOAD_CONSUME_LOOP[\s\S]*?ATTN_OUT_TOPFED_INVALID_FALLBACK_LOOP' -Reason "AttnLayer0 OUT stage missing deeper consume/fallback loops for topfed payload"
Require-Regex -Text $attnText -Pattern '(?ms)if\s*\(\s*prebuilt_handoff\.out_prebuilt_from_top_managed\s*\)\s*\{[\s\S]*?return;' -Reason "AttnLayer0 legacy out_prebuilt anti-fallback guard missing"

Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_ATTNLAYER0_OUT_TOPFED_PAYLOAD_CONSUME PASS' -Reason "TB missing topfed deeper consume PASS banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_ATTNLAYER0_OUT_TOPFED_PAYLOAD_INVALID_FALLBACK PASS' -Reason "TB missing topfed invalid fallback PASS banner"
Require-Regex -Text $tbText -Pattern '(?ms)P11ANB_ATTNLAYER0_OUT_TOPFED_PAYLOAD_DISABLED_FALLBACK PASS' -Reason "TB missing topfed disabled fallback PASS banner"

Require-Regex -Text $runnerText -Pattern '(?ms)OUT_TOPFED_PAYLOAD_CONSUME PASS' -Reason "runner missing topfed consume gate"
Require-Regex -Text $runnerText -Pattern '(?ms)OUT_TOPFED_PAYLOAD_INVALID_FALLBACK PASS' -Reason "runner missing topfed invalid fallback gate"
Require-Regex -Text $runnerText -Pattern '(?ms)OUT_TOPFED_PAYLOAD_DISABLED_FALLBACK PASS' -Reason "runner missing topfed disabled fallback gate"

Write-Log "PASS: check_p11anb_attnlayer0_deeper_boundary"
Write-Summary -Status "PASS" -Detail "all checks passed"
exit 0
