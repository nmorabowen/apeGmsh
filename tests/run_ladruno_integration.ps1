<#
.SYNOPSIS
  Run the apeGmsh <-> Ladruno-fork end-to-end integration suite.

.DESCRIPTION
  Points apeGmsh's live OpenSees backend at a fork build directory
  containing opensees.pyd (+ its co-located MKL DLLs) via APEGMSH_OPENSEES_BIN,
  then runs `pytest -m ladruno_fork` over tests/opensees/integration_ladruno
  with py 3.12. The backend resolver in apeGmsh/opensees/emitter/live.py picks
  up APEGMSH_OPENSEES_BIN and imports the fork's `opensees` module; the
  conftest auto-skips these tests if the fork backend does not resolve.

.PARAMETER DistBin
  Folder holding the fork's opensees.pyd. Defaults to the sibling OpenSees
  checkout's dist\bin. Pass explicitly when running from a worktree.

.EXAMPLE
  # Uses the project's opensees_env (Py 3.12; apeGmsh editable + gmsh + pytest +
  # the fork 'opensees' on the path). No new venv needed.

  ./tests/run_ladruno_integration.ps1
  ./tests/run_ladruno_integration.ps1 -DistBin C:\Users\nmb\Documents\Github\OpenSees\dist\bin
  ./tests/run_ladruno_integration.ps1 -- -k recorder      # extra args after -- go to pytest

.NOTES
  Runs in the project's ``opensees_env`` (CPython 3.12 — the fork pyd's ABI; has
  apeGmsh editable, gmsh, pytest, and both the fork ``opensees`` and stock
  ``openseespy``). Pass -VenvPython to use a different interpreter (must be 3.12).
  Requires the fork to be BUILT (opensees.pyd present). The fork's splash banner is
  suppressed via LADRUNO_OPENSEES_QUIET (the resolver also sets this).
#>
param(
    [string]$DistBin = (Join-Path $PSScriptRoot '..\..\OpenSees\dist\bin'),
    [string]$VenvPython = 'C:\Users\nmb\venv\opensees_env\Scripts\python.exe',
    [Parameter(ValueFromRemainingArguments = $true)] $Extra
)
$ErrorActionPreference = 'Stop'

if (-not (Test-Path (Join-Path $DistBin 'opensees.pyd'))) {
    throw "No opensees.pyd under '$DistBin'. Build the fork first, or pass -DistBin <path-with-opensees.pyd>."
}
$DistBin = (Resolve-Path $DistBin).Path
$suite   = (Resolve-Path (Join-Path $PSScriptRoot 'opensees\integration_ladruno')).Path
$repo    = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

# Prefer the opensees_env interpreter; fall back to the py launcher's 3.12.
if (Test-Path $VenvPython) {
    $pyExe = $VenvPython; $pyPre = @()
} else {
    $pyExe = 'py';        $pyPre = @('-3.12')
}

$env:APEGMSH_OPENSEES_BIN = $DistBin
$env:PATH                 = "$DistBin;$env:PATH"
$env:LADRUNO_OPENSEES_QUIET = '1'

Write-Host "python       : $pyExe $($pyPre -join ' ')"
Write-Host "fork backend : $DistBin"
Write-Host "suite        : $suite`n"

Push-Location $repo
try {
    & $pyExe @pyPre -m pytest -v -m ladruno_fork $suite @Extra
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
