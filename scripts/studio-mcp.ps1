# ADR 0095 S4a — stdio MCP door for Cursor.
# The office venv's ladruno_opensees.pth prints a banner to STDOUT at
# interpreter startup. That corrupts JSON-RPC unless the quiet env is
# set *before* Python starts.
$ErrorActionPreference = "Stop"
$env:LADRUNO_OPENSEES_QUIET = "1"
$env:APEGMSH_QUIET = "1"

# Habitat files live at <repo>/.apegmsh/. Cursor's MCP cwd is not
# reliable (user-level config often starts in $HOME). Pin cwd to the
# repo that contains this script.
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo

# Prefer an explicit APEGMSH_PYTHON, then the office Ladruno/OpenSees
# venv (folder name is opensees_env on this machine; keep the older
# opensees_venv spelling as a fallback), then PATH python.
$office_env = Join-Path $env:USERPROFILE "venv\opensees_env\Scripts\python.exe"
$office_venv = Join-Path $env:USERPROFILE "venv\opensees_venv\Scripts\python.exe"
if ($env:APEGMSH_PYTHON) {
    $py = $env:APEGMSH_PYTHON
} elseif (Test-Path $office_env) {
    $py = $office_env
} elseif (Test-Path $office_venv) {
    $py = $office_venv
} else {
    $py = "python"
}
# Prefer this checkout's src over any other editable install on the venv.
$env:PYTHONPATH = (Join-Path $repo "src")
& $py -m apeGmsh.studio.mcp
exit $LASTEXITCODE
