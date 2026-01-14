
@echo off
setlocal EnableExtensions

REM Always run from this script's folder so relative paths behave
pushd "%~dp0"

REM Compute workflow root: ...\Lipidomics_Kinetics_Workflow2
for %%I in ("%~dp0..") do set "ROOT=%%~fI"

REM Packed environment root + python
set "ENV=%ROOT%\DeuteRater_python"
set "PY=%ENV%\python.exe"

REM ---- Optional: if user launched from (base), prevent conda variables interfering
set "CONDA_PREFIX="
set "CONDA_DEFAULT_ENV="
set "CONDA_PROMPT_MODIFIER="
set "PYTHONHOME="
set "PYTHONPATH="

REM ---- CRITICAL: ensure conda-style DLL directories are found FIRST
REM Conda envs on Windows commonly store needed DLLs in <env>\Library\bin. [3](https://www.anaconda.com/blog/moving-conda-environments)
set "PATH=%ENV%;%ENV%\Scripts;%ENV%\DLLs;%ENV%\Library\bin;%PATH%"

REM Helps some conda/Python 3.8 DLL search edge cases on some setups
set "CONDA_DLL_SEARCH_MODIFICATION_ENABLE=1"

REM Debug (keep while testing)
echo [INFO] Using python: "%PY%"
echo [INFO] Working dir: "%CD%"
"%PY%" -c "import sys; print('[INFO] sys.executable =', sys.executable)"

REM Run your entrypoint
"%PY%" "%~dp0__main__.py"

REM Keep window open if there was an error (so double-click users can see the traceback)
if errorlevel 1 (
  echo.
  echo [ERROR] Program exited with an error.
  pause
)

popd
endlocal
