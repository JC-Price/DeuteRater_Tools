@echo off
setlocal EnableExtensions EnableDelayedExpansion

echo ============================================
echo   MS_dial_to_Isobar_Handler – DEBUG MODE
echo ============================================
echo.

REM Directory where this .cmd lives
set "SCRIPT_DIR=%~dp0"

REM Portable Python (relative)
set "PYTHON_EXE=%SCRIPT_DIR%\..\shared_python\Kinetic_Lipidomics_env\python.exe"

if not exist "%PYTHON_EXE%" (
    echo ERROR: Portable Python not found:
    echo %PYTHON_EXE%
    echo.
    goto :end
)

REM Compute environment root
for %%I in ("%PYTHON_EXE%\..\..") do set "ENV_DIR=%%~fI"

REM Ensure local DLL resolution (Tk, numpy, scipy, etc.)
set "PATH=%ENV_DIR%;%ENV_DIR%\Library\bin;%ENV_DIR%\DLLs;%PATH%"

echo Using Python:
"%PYTHON_EXE%" -c "import sys; print(sys.executable); print(sys.version)"
echo.

echo Running script: MS_dial_to_Isobar_Handler.py
echo --------------------------------------------
"%PYTHON_EXE%" "%SCRIPT_DIR%\MS_dial_to_Isobar_Handler.py" %*
set EXIT_CODE=%ERRORLEVEL%
echo --------------------------------------------

echo.
echo Python exited with code %EXIT_CODE%

if not "%EXIT_CODE%"=="0" (
    echo.
    echo !!! ERROR DETECTED !!!
    echo Scroll up to see the traceback.
)

:end
echo.
echo ============================================
echo   Press any key to close
echo ============================================
pause
