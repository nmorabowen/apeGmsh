# Finish APE Studio habitat session
param(
    [string]$Slug = ""
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
$args = @("$PSScriptRoot\finish_session.py")
if ($Slug) { $args += @("--slug", $Slug) }
& $py @args
exit $LASTEXITCODE
