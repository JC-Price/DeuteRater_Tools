
@echo off
echo.
echo ================================================
echo   Fixing DeuteRaterEnvironment for conda-pack
echo ================================================
echo.

if "%CONDA_PREFIX%"=="" (
    echo ERROR: No conda environment active.
    echo Please run:  conda activate DeuteRaterEnvironment
    pause
    exit /b 1
)

echo Using environment at: "%CONDA_PREFIX%"
echo.

REM ------------------------------------------------
REM 1. FIX QT PLATFORM PLUGINS
REM ------------------------------------------------
echo Creating Qt platforms directory...
mkdir "%CONDA_PREFIX%\Library\bin\platforms" 2>nul

echo Copying Qt platform plugins (qwindows / qminimal / qoffscreen)...
copy "%CONDA_PREFIX%\Library\plugins\platforms\qwindows.dll" "%CONDA_PREFIX%\Library\bin\platforms\" >nul
copy "%CONDA_PREFIX%\Library\plugins\platforms\qminimal.dll" "%CONDA_PREFIX%\Library\bin\platforms\" >nul
copy "%CONDA_PREFIX%\Library\plugins\platforms\qoffscreen.dll" "%CONDA_PREFIX%\Library\bin\platforms\" >nul
copy "%CONDA_PREFIX%\Library\plugins\platforms\qdirect2d.dll" "%CONDA_PREFIX%\Library\bin\platforms\" >nul

echo Qt platform fix complete.
echo.

REM ------------------------------------------------
REM 2. FIX SSL (libssl + libcrypto)
REM ------------------------------------------------
echo Copying SSL DLLs...
copy "%CONDA_PREFIX%\Library\bin\libcrypto-1_1-x64.dll" "%CONDA_PREFIX%\DLLs\" >nul
copy "%CONDA_PREFIX%\Library\bin\libssl-1_1-x64.dll" "%CONDA_PREFIX%\DLLs\" >nul

echo SSL fix complete.
echo.

REM ------------------------------------------------
REM 3. VERIFY LAYOUT
REM ------------------------------------------------
echo Verifying files...

if exist "%CONDA_PREFIX%\Library\bin\platforms\qwindows.dll" (
    echo ✔ qwindows.dll copied successfully.
) else (
    echo ✘ ERROR: qwindows.dll missing from Library\bin\platforms.
)

if exist "%CONDA_PREFIX%\Library\bin\platforms\qminimal.dll" (
    echo ✔ qminimal.dll present.
) else (
    echo ✘ ERROR: qminimal.dll missing.
)

if exist "%CONDA_PREFIX%\Library\bin\platforms\qoffscreen.dll" (
    echo ✔ qoffscreen.dll present.
) else (
    echo ✘ ERROR: qoffscreen.dll missing.
)

if exist "%CONDA_PREFIX%\DLLs\libssl-1_1-x64.dll" (
    echo ✔ SSL: libssl OK.
) else (
    echo ✘ ERROR: libssl missing.
)

if exist "%CONDA_PREFIX%\DLLs\libcrypto-1_1-x64.dll" (
    echo ✔ SSL: libcrypto OK.
) else (
    echo ✘ ERROR: libcrypto missing.
)

echo.
echo ================================================
echo   Environment fixed — ready to conda-pack!
echo ================================================
echo Now run:
echo     conda-pack -n DeuteRaterEnvironment -o DeuteRaterEnvironment.tar.gz
echo.

pause
