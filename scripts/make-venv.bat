@echo off
setlocal EnableDelayedExpansion
REM ---------------------------------------------------------------------
REM make-venv.bat - create a fresh apeGmsh virtualenv under C:\venv.
REM
REM Prompts for a name, builds C:\venv\<name>, and installs THIS checkout
REM editable with every extra ([all] + the Studio MCP extra), so the venv
REM covers the whole library: gmsh/h5py/numpy/pandas, openseespy, the Qt +
REM trame viewers, plotting, dxf, and `python -m apeGmsh.studio.mcp`.
REM
REM Why C:\venv and not %USERPROFILE%: the VTK wheels resolve their DLLs
REM through a path baked in at venv-creation time, and a deep parent
REM directory makes `import pyvista` die with "DLL load failed while
REM importing vtkCommonMath: The filename or extension is too long."
REM A short root avoids it outright.
REM
REM `[all]` deliberately does NOT include the `mcp` extra upstream, so it
REM is requested explicitly here - otherwise the Studio habitat installs
REM but cannot start.
REM ---------------------------------------------------------------------

set "VENV_ROOT=C:\venv"
set "REPO=%~dp0.."
pushd "%REPO%" >nul 2>&1 || (echo ERROR: cannot resolve the repo root from %~dp0. & exit /b 1)
set "REPO=%CD%"
popd >nul

echo.
echo   apeGmsh environment bootstrap
echo   repo: %REPO%
echo   root: %VENV_ROOT%
echo.

REM -- 1. Name ----------------------------------------------------------
set "VENV_NAME="
set /p "VENV_NAME=Environment name (created as %VENV_ROOT%\<name>): "
if not defined VENV_NAME (
    echo ERROR: no name given.
    exit /b 1
)
REM Strip surrounding quotes if the user pasted a quoted name.
set "VENV_NAME=%VENV_NAME:"=%"
REM Trim leading/trailing whitespace - `set /p` keeps whatever was typed,
REM and a stray trailing space would otherwise fail the allowlist below
REM with a message that looks wrong to someone who typed a valid name.
for /f "tokens=* delims= " %%A in ("%VENV_NAME%") do set "VENV_NAME=%%A"
:trimtail
if not defined VENV_NAME goto :trimmed
if "%VENV_NAME:~-1%"==" " set "VENV_NAME=%VENV_NAME:~0,-1%" & goto :trimtail
if "%VENV_NAME:~-1%"=="	" set "VENV_NAME=%VENV_NAME:~0,-1%" & goto :trimtail
:trimmed
if not defined VENV_NAME (
    echo ERROR: no name given.
    exit /b 1
)
REM Allowlist rather than a denylist of forbidden characters: cmd does not
REM treat \" as an escape, so a pattern containing < > | breaks the parser
REM before findstr ever sees it. Letters, digits, underscore, dot, dash.
echo(%VENV_NAME%| findstr /r /c:"^[A-Za-z0-9_.-][A-Za-z0-9_.-]*$" >nul || (
    echo ERROR: "%VENV_NAME%" is not a usable folder name.
    echo        Allowed: letters, digits, underscore, dot, dash - no spaces.
    echo        e.g. apegmsh, fem-dev, apegmsh_2026
    exit /b 1
)
REM Dots are legal in a name but "." / ".." / "a..b" walk the tree.
if "%VENV_NAME%"=="." goto :badname
if "%VENV_NAME%"==".." goto :badname
echo(%VENV_NAME%| findstr /c:".." >nul && goto :badname
goto :nameok
:badname
echo ERROR: "%VENV_NAME%" is a path traversal, not a name.
exit /b 1
:nameok

set "TARGET=%VENV_ROOT%\%VENV_NAME%"

REM -- 2. Refuse to clobber silently ------------------------------------
if exist "%TARGET%\Scripts\python.exe" (
    echo.
    echo   %TARGET% already exists.
    set "REPLY="
    set /p "REPLY=Delete and recreate it? [y/N]: "
    if /i not "!REPLY!"=="y" (
        echo Aborted - nothing changed.
        exit /b 1
    )
    echo Removing %TARGET% ...
    rmdir /s /q "%TARGET%" || (echo ERROR: could not remove %TARGET%. & exit /b 1)
) else if exist "%TARGET%" (
    echo ERROR: %TARGET% exists but is not a virtualenv. Refusing to touch it.
    exit /b 1
)

REM -- 3. Pick a base interpreter ---------------------------------------
REM apeGmsh needs >= 3.10; prefer a version with wheels for the whole
REM viewer stack. Newest-first would land on a Python too new for
REM openseespy/vtk wheels, so the order is deliberate.
set "BASEPY="
for %%V in (3.12 3.11 3.13 3.10) do (
    if not defined BASEPY (
        py -%%V -c "import sys" >nul 2>&1 && set "BASEPY=py -%%V"
    )
)
if not defined BASEPY (
    python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1 && set "BASEPY=python"
)
if not defined BASEPY (
    echo ERROR: no Python 3.10+ found. Install one from python.org and retry.
    exit /b 1
)

echo.
echo   base interpreter: %BASEPY%
for /f "delims=" %%P in ('%BASEPY% -c "import sys;print(sys.version.split()[0])"') do echo   version         : %%P
echo.

REM -- 4. Build ---------------------------------------------------------
if not exist "%VENV_ROOT%" mkdir "%VENV_ROOT%" || (echo ERROR: cannot create %VENV_ROOT%. & exit /b 1)

echo [1/3] Creating %TARGET% ...
%BASEPY% -m venv "%TARGET%" || (echo ERROR: venv creation failed. & exit /b 1)

set "VPY=%TARGET%\Scripts\python.exe"

echo [2/3] Upgrading pip ...
"%VPY%" -m pip install --quiet --upgrade pip || (echo ERROR: pip upgrade failed. & exit /b 1)

echo [3/3] Installing apeGmsh with every extra - this pulls PySide6, VTK
echo       and gmsh, so expect a few hundred MB and several minutes ...
"%VPY%" -m pip install -e "%REPO%[all]" "mcp>=1.2" || (
    echo.
    echo ERROR: install failed. The venv at %TARGET% is left in place so you
    echo        can inspect it or rerun pip by hand.
    exit /b 1
)

REM -- 5. Verify --------------------------------------------------------
echo.
echo === apeGmsh doctor ===
set "APEGMSH_QUIET=1"
set "LADRUNO_OPENSEES_QUIET=1"
"%VPY%" -m apeGmsh doctor
set "DOCTOR=%ERRORLEVEL%"

echo.
echo ======================================================================
echo   Environment ready: %TARGET%
echo.
echo   Activate:  %TARGET%\Scripts\activate.bat
echo   Or call:   "%VPY%" your_script.py
echo.
if not "%DOCTOR%"=="0" (
    echo   doctor reported an error-severity finding - read the report above.
    echo   A D5 warning about the OpenSees backend is expected: stock
    echo   openseespy is installed, and the Ladruno fork is a separate
    echo   build. Point APEGMSH_OPENSEES_BIN at its dist\bin to use it.
) else (
    echo   doctor is clean. Fork-only features ^(Ladruno elements, contact,
    echo   equation ties, explicit integrators^) need the Ladruno OpenSees
    echo   build; set APEGMSH_OPENSEES_BIN to its dist\bin folder.
)
echo ======================================================================
echo.

endlocal
exit /b 0
