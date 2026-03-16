param(
    [string]$RepoRoot = "."
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Test-Anchor {
    param(
        [string]$Text,
        [string]$Pattern
    )

    return [System.Text.RegularExpressions.Regex]::IsMatch(
        $Text,
        $Pattern,
        [System.Text.RegularExpressions.RegexOptions]::Singleline
    )
}

$repo = (Resolve-Path $RepoRoot).Path
$findings = New-Object System.Collections.Generic.List[string]

$typesPath = Join-Path $repo "include\AecctTypes.h"
$protocolPath = Join-Path $repo "include\AecctProtocol.h"
$topPath = Join-Path $repo "src\Top.h"

foreach ($p in @($typesPath, $protocolPath, $topPath)) {
    if (-not (Test-Path $p)) {
        $findings.Add("missing file: $p")
    }
}

if ($findings.Count -gt 0) {
    Write-Host "FAIL: check_interface_lock"
    foreach ($item in $findings) {
        Write-Host $item
    }
    exit 1
}

$typesText = Get-Content -Path $typesPath -Raw
$protocolText = Get-Content -Path $protocolPath -Raw
$topText = Get-Content -Path $topPath -Raw

$typeAnchors = @(
    @{ Name = "u16_t typedef"; Pattern = 'typedef\s+ac_int\s*<\s*16\s*,\s*false\s*>\s+u16_t\s*;' },
    @{ Name = "u32_t typedef"; Pattern = 'typedef\s+ac_int\s*<\s*32\s*,\s*false\s*>\s+u32_t\s*;' },
    @{ Name = "ctrl channel typedef"; Pattern = 'typedef\s+ac_channel\s*<\s*u16_t\s*>\s+ctrl_ch_t\s*;' },
    @{ Name = "data channel typedef"; Pattern = 'typedef\s+ac_channel\s*<\s*u32_t\s*>\s+data_ch_t\s*;' }
)

foreach ($anchor in $typeAnchors) {
    if (-not (Test-Anchor -Text $typesText -Pattern $anchor.Pattern)) {
        $findings.Add("type_lock: missing anchor '$($anchor.Name)' in include/AecctTypes.h")
    }
}

$protocolAnchors = @(
    @{ Name = "pack_ctrl_cmd"; Pattern = '\bpack_ctrl_cmd\s*\(' },
    @{ Name = "unpack_ctrl_cmd_opcode"; Pattern = '\bunpack_ctrl_cmd_opcode\s*\(' },
    @{ Name = "pack_ctrl_rsp_ok"; Pattern = '\bpack_ctrl_rsp_ok\s*\(' },
    @{ Name = "pack_ctrl_rsp_done"; Pattern = '\bpack_ctrl_rsp_done\s*\(' },
    @{ Name = "pack_ctrl_rsp_err"; Pattern = '\bpack_ctrl_rsp_err\s*\(' },
    @{ Name = "unpack_ctrl_rsp_kind"; Pattern = '\bunpack_ctrl_rsp_kind\s*\(' },
    @{ Name = "unpack_ctrl_rsp_payload"; Pattern = '\bunpack_ctrl_rsp_payload\s*\(' }
)

foreach ($anchor in $protocolAnchors) {
    if (-not (Test-Anchor -Text $protocolText -Pattern $anchor.Pattern)) {
        $findings.Add("protocol_ssot: missing anchor '$($anchor.Name)' in include/AecctProtocol.h")
    }
}

$topAnchors = @(
    @{ Name = "protocol include"; Pattern = '#include\s+"AecctProtocol\.h"' },
    @{
        Name = "4-channel top contract"
        Pattern = 'static\s+inline\s+void\s+top\s*\(\s*ac_channel\s*<\s*ac_int\s*<\s*16\s*,\s*false\s*>\s*>\s*&\s*ctrl_cmd\s*,\s*ac_channel\s*<\s*ac_int\s*<\s*16\s*,\s*false\s*>\s*>\s*&\s*ctrl_rsp\s*,\s*ac_channel\s*<\s*ac_int\s*<\s*32\s*,\s*false\s*>\s*>\s*&\s*data_in\s*,\s*ac_channel\s*<\s*ac_int\s*<\s*32\s*,\s*false\s*>\s*>\s*&\s*data_out\s*\)'
    }
)

foreach ($anchor in $topAnchors) {
    if (-not (Test-Anchor -Text $topText -Pattern $anchor.Pattern)) {
        $findings.Add("top_signature: missing anchor '$($anchor.Name)' in src/Top.h")
    }
}

if ($findings.Count -gt 0) {
    Write-Host "FAIL: check_interface_lock"
    foreach ($item in $findings) {
        Write-Host $item
    }
    exit 1
}

Write-Host "PASS: check_interface_lock"
exit 0
