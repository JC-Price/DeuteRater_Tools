
@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM =========================================================
REM creating_security_key.cmd
REM - Normal mode: no args (double-click) -> GUI flow
REM - Read mode:  --read "<path>"        -> headless key check with debug info
REM =========================================================

REM Directory where this .cmd lives
set "SCRIPT_DIR=%~dp0"

REM Portable Python (relative to misc\)
set "PYTHON_EXE=%SCRIPT_DIR%\..\shared_python\Kinetic_Lipidomics_env\python.exe"

REM Validate Python
if not exist "%PYTHON_EXE%" (
    echo ERROR: Portable Python not found:
    echo   %PYTHON_EXE%
    exit /b 2
)

REM Compute environment root for DLL resolution
for %%I in ("%PYTHON_EXE%\..\..") do set "ENV_DIR=%%~fI"
set "PATH=%ENV_DIR%;%ENV_DIR%\Library\bin;%ENV_DIR%\DLLs;%PATH%"

REM -------------------------------------------------------------------
REM READ MODE: creating_security_key.cmd --read "<target_folder>"
REM -------------------------------------------------------------------
if /i "%~1"=="--read" goto readMode

REM -------------------------------------------------------------------
REM NORMAL MODE: no args -> GUI flow
REM -------------------------------------------------------------------
echo ============================================
echo   creating_security_key – DEBUG MODE
echo ============================================
echo.

echo Using Python:
"%PYTHON_EXE%" -c "import sys; print(sys.executable); print(sys.version)"
echo.

echo Running script: creating_security_key.py
echo --------------------------------------------
"%PYTHON_EXE%" "%SCRIPT_DIR%\creating_security_key.py"
set "EXIT_CODE=%ERRORLEVEL%"
echo --------------------------------------------

echo.
echo Python exited with code %EXIT_CODE%

if not "%EXIT_CODE%"=="0" (
    echo.
    echo !!! ERROR DETECTED !!!
    echo Scroll up to see the traceback.
)

echo.
echo ============================================
echo   Press any key to close
echo ============================================
pause
exit /b %EXIT_CODE%

:readMode
REM Collect all remaining arguments into TARGET_DIR (handles spaces)
set "TARGET_DIR="
shift
:collectArgs
if "%~1"=="" goto argsDone
if defined TARGET_DIR (
    set "TARGET_DIR=!TARGET_DIR! %~1"
) else (
    set "TARGET_DIR=%~1"
)
shift
goto collectArgs

:argsDone
if "!TARGET_DIR!"=="" (
    echo [DEBUG] Missing target directory argument.
    echo False
    exit /b 1
)

echo ============================================
echo   creating_security_key – READ MODE
echo ============================================
echo Target directory: "!TARGET_DIR!"
echo.

"%PYTHON_EXE%" "%SCRIPT_DIR%\creating_security_key.py" --read "!TARGET_DIR!"
set "EXIT_CODE=%ERRORLEVEL%"
echo --------------------------------------------
echo Python exited with code %EXIT_CODE%
exit /b %EXIT_CODE%
