
@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM  Kinetic_Lipdomics_Environment_Wizard.cmd
REM
REM  Resumable installer for two conda-pack environments:
REM   1) DeuteRaterEnvironment.tar.gz  -> .\DeuteRater_python\
REM   2) Kinetic_Lipidomics.tar.gz     -> .\shared_python\Kinetic_Lipidomics_env\
REM
REM  Hardened for Windows:
REM   - Clears Read-only attributes after extraction (attrib -R /S /D)
REM   - Retries conda-unpack on transient file locks/AV scanning
REM   - Auto-flattens double-nesting (e.g. env\env\python.exe)
REM ============================================================

set "HERE=%~dp0"
set "ME=%~f0"

call :Banner

REM ---- Pre-flight: tar.exe required for .tar.gz extraction
call :Step "Pre-flight checks" "Checking required tools..."
where tar.exe >nul 2>nul
if errorlevel 1 (
    call :Fail "tar.exe not found on PATH." ^
               "This wizard needs tar.exe or you must extract manually."
    goto :ENDFAIL
)

REM ---- Cleanup/resume logic
call :Step "Resume & Cleanup check" ^
           "Checking environment folders..."

call :EnsureFolderState "DeuteRater_python"
if errorlevel 1 goto :ENDFAIL

call :EnsureFolderState "shared_python\Kinetic_Lipidomics_env"
if errorlevel 1 goto :ENDFAIL

REM ---- Process environments
call :ProcessOne "DeuteRaterEnvironment.tar.gz" "DeuteRater_python" "DeuteRater"
if errorlevel 1 goto :ENDFAIL

call :ProcessOne "Kinetic_Lipidomics.tar.gz" "shared_python\Kinetic_Lipidomics_env" "Kinetic Lipidomics"
if errorlevel 1 goto :ENDFAIL

call :Step "All done!" "Both environments are installed and ready." "Self-deleting..."
call :SelfDelete
exit /b 0



REM ============================================================
REM  FUNCTIONS
REM ============================================================

:Banner
echo.
echo ============================================================
echo              Welcome to Kinetic Lipidomics
echo ============================================================
echo.
echo Installs two portable Python environments (no conda needed).
echo.
exit /b 0


:Step
setlocal
echo.
echo ------------------------------------------------------------
echo  %~1
echo ------------------------------------------------------------
if not "%~2"=="" echo  - %~2
if not "%~3"=="" echo  - %~3
if not "%~4"=="" echo  - %~4
endlocal
exit /b 0


:Fail
setlocal
echo.
echo ============================================================
echo  [ERROR] %~1
echo ============================================================
if not "%~2"=="" echo  %~2
if not "%~3"=="" echo  %~3
echo.
pause
endlocal
exit /b 1


:EnsureFolderState
setlocal EnableDelayedExpansion
set "TARGETREL=%~1"
set "TARGET=%HERE%%TARGETREL%"

if not exist "%TARGET%" (
    call :Step "Check: %TARGETREL%" "Folder missing (OK — will install)."
    endlocal & exit /b 0
)

if exist "%TARGET%\python.exe" (
    call :Step "Check: %TARGETREL%" "Valid python.exe found — keeping."
    endlocal & exit /b 0
)

call :Step "Check: %TARGETREL%" "Folder incomplete — deleting..."
rmdir /s /q "%TARGET%"
if exist "%TARGET%" (
    call :Fail "Could not delete folder: %TARGETREL%"
    endlocal & exit /b 1
)
call :Step "Check: %TARGETREL%" "Deleted incomplete folder."
endlocal
exit /b 0


:ClearReadOnly
setlocal
set "FOLDER=%~1"
attrib -R "%FOLDER%\*" /S /D >nul 2>nul
endlocal
exit /b 0


:FlattenIfNested
setlocal EnableDelayedExpansion
set "TARGET=%~1"
for %%F in ("%TARGET%") do set "LEAF=%%~nxF"

if exist "%TARGET%\python.exe" endlocal & exit /b 0

if exist "%TARGET%\!LEAF!\python.exe" (
    call :Step "Fix layout" "Flattening nested environment layout..."
    robocopy "%TARGET%\!LEAF!" "%TARGET%" /E /MOVE >nul
    rmdir /s /q "%TARGET%\!LEAF%"
)

endlocal
exit /b 0


:RunCondaUnpackWithRetries
setlocal EnableDelayedExpansion
set "TARGET=%~1"
set "LABEL=%~2"
set "UNPACK=%TARGET%\Scripts\conda-unpack.exe"

set "RETRIES=6"
set "WAITSEC=2"

:TRYAGAIN
call :Step "%LABEL%: conda-unpack" "Running relocation fix (attempt !RETRIES!)..."

call "%UNPACK%"
if not errorlevel 1 (
    endlocal & exit /b 0
)

set /a RETRIES-=1
if !RETRIES! LEQ 0 (
    call :Fail "%LABEL% conda-unpack failed."
    endlocal & exit /b 1
)

ping 127.0.0.1 -n %WAITSEC% >nul
goto :TRYAGAIN


:ProcessOne
setlocal EnableDelayedExpansion

set "ARCHIVE=%~1"
set "TARGETREL=%~2"
set "LABEL=%~3"

set "ARCHIVEPATH=%HERE%%ARCHIVE%"
set "TARGET=%HERE%%TARGETREL%"
set "READYMARK=%TARGET%\.__condapack_ready__"

call :Step "Installing: %LABEL%" "Target: %TARGETREL%"

REM === SKIP EXTRACTION IF ALREADY PRESENT ===
if exist "%TARGET%\python.exe" (
    call :Step "%LABEL%" "Environment already exists."

    if not exist "%READYMARK%" (
        call :ClearReadOnly "%TARGET%"
        call :RunCondaUnpackWithRetries "%TARGET%" "%LABEL%"
        echo ready>"%READYMARK%"
    )

    REM === >>> ADDED FOR DeuteRater ===
    if "%LABEL%"=="DeuteRater" (
        call :FixQtConf "%TARGET%"
    )
    REM === <<< END ADD ===

    if exist "%ARCHIVEPATH%" del /f /q "%ARCHIVEPATH%"
    endlocal & exit /b 0
)

REM === NEED ARCHIVE FOR FRESH INSTALL ===
if not exist "%ARCHIVEPATH%" (
    call :Fail "%LABEL% archive missing." "Expected: %ARCHIVE%"
    endlocal & exit /b 1
)

mkdir "%TARGET%" >nul 2>nul

call :Step "%LABEL%: Extract" "Extracting archive..."
tar -xzf "%ARCHIVEPATH%" -C "%TARGET%"

call :FlattenIfNested "%TARGET%"

if not exist "%TARGET%\python.exe" (
    call :Fail "%LABEL%" "python.exe missing after extraction."
    endlocal & exit /b 1
)

call :ClearReadOnly "%TARGET%"

call "%TARGET%\Scripts\activate.bat"
call :RunCondaUnpackWithRetries "%TARGET%" "%LABEL%"
echo ready>"%READYMARK%"
call "%TARGET%\Scripts\deactivate.bat"

REM === >>> ADDED FOR DeuteRater ===
if "%LABEL%"=="DeuteRater" (
    call :FixQtConf "%TARGET%"
)
REM === <<< END ADD ===

del /f /q "%ARCHIVEPATH%"

call :Step "%LABEL% installed" "Installed to %TARGETREL%"

endlocal
exit /b 0



REM ============================================================
REM  Qt FIX — **ONLY FOR DEUTERATER**
REM ============================================================

:FixQtConf
REM Rewrite qt.conf after relocation
REM %1 = target environment folder

set "QTCONF=%~1\qt.conf"
if not exist "%QTCONF%" exit /b 0

call :Step "Qt Fix" "Updating qt.conf with the new relative paths..."

(
echo [Paths]
echo Prefix=.
echo Binaries=Library/bin
echo Plugins=Library/plugins
echo Imports=Library/qml
echo Qml2Imports=Library/qml
) > "%QTCONF%"

exit /b 0



:SelfDelete
call :Step "Cleanup" "Deleting this wizard..."
start "" /b cmd /c "ping 127.0.0.1 -n 2 >nul & del /f /q ""%ME%"""
exit /b 0


:ENDFAIL
exit /b 1
