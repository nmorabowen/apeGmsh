# Start APE Studio habitat session
param(
    [string]$Slug = "session",
    [switch]$SkipTemplate
)
$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

$py = $env:APEGMSH_PYTHON
if (-not $py) {
    $cand = Join-Path $env:USERPROFILE "venv\opensees_env\Scripts\python.exe"
    if (Test-Path $cand) { $py = $cand } else { $py = "python" }
}

$env:PYTHONIOENCODING = "utf-8"
$args = @("$PSScriptRoot\start_session.py", "--slug", $Slug)
if ($SkipTemplate) { $args += "--skip-template" }
& $py @args
exit $LASTEXITCODE
