# Finish APE Studio habitat session
param(
    [string]$Slug = ""
)
$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

# Resolve the interpreter that has apeGmsh. First hit wins:
#   1. APEGMSH_PYTHON  - explicit pin, always honoured
#   2. VIRTUAL_ENV     - an activated venv is the honest signal
#   3. office venvs    - APE INGENIERIA convention, both spellings
#   4. PATH python     - last resort, and usually cannot import apeGmsh
# Same shape as apeGmsh's own scripts/studio-mcp.ps1. A habitat travels
# between machines; the interpreter is the one thing that never travels
# with it, so no single path (and no user name) belongs here.
$candidates = @()
if ($env:VIRTUAL_ENV) {
    $candidates += (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe")
}
$candidates += @(
    (Join-Path $env:USERPROFILE "venv\opensees_env\Scripts\python.exe"),
    (Join-Path $env:USERPROFILE "venv\opensees_venv\Scripts\python.exe")
)

$py = $env:APEGMSH_PYTHON
if (-not $py) {
    $py = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $py) {
    $py = "python"
    Write-Host ("WARN: no venv found - falling back to PATH python, which " +
        "probably cannot import apeGmsh. Set APEGMSH_PYTHON to pin one.")
}
Write-Host "python: $py"

$env:PYTHONIOENCODING = "utf-8"
$args = @("$PSScriptRoot\finish_session.py")
if ($Slug) { $args += @("--slug", $Slug) }
& $py @args
exit $LASTEXITCODE
