# ADR 0095 S4a / S5a — stdio MCP door for Cursor.
# The office venv's ladruno_opensees.pth prints a banner to STDOUT at
# interpreter startup. That corrupts JSON-RPC unless the quiet env is
# set *before* Python starts.
$ErrorActionPreference = "Stop"
$env:LADRUNO_OPENSEES_QUIET = "1"
$env:APEGMSH_QUIET = "1"

# INV-15: do not Set-Location to the apeGmsh repo. Habitat files live
# under the *model* project root. Pass root= on each MCP tool, or set
# APEGMSH_ROOT to that project before starting this script.

$officeCandidates = @(
    (Join-Path $env:USERPROFILE "venv\opensees_env\Scripts\python.exe"),
    (Join-Path $env:USERPROFILE "venv\opensees_venv\Scripts\python.exe")
)
$office = $officeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if ($env:APEGMSH_PYTHON) {
    $py = $env:APEGMSH_PYTHON
} elseif ($office) {
    $py = $office
} else {
    $py = "python"
    [Console]::Error.WriteLine(
        "studio-mcp.ps1: no office venv found; using PATH python. " +
        "Set APEGMSH_PYTHON to pin the interpreter."
    )
}
& $py -m apeGmsh.studio.mcp
exit $LASTEXITCODE
