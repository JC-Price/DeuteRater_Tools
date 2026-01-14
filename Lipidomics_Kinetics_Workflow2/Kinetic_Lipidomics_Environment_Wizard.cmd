
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
REM
REM  Behavior:
REM   - If target folder exists AND contains python.exe: keep it (resume)
REM   - If target folder exists AND missing python.exe: delete it (cleanup)
REM   - If target folder missing: install it
REM   - Always run conda-unpack when needed, then mark ready
REM   - Delete each .tar.gz after successful completion of that env
REM   - Delete this wizard at the end
REM ============================================================

set "HERE=%~dp0"
set "ME=%~f0"

call :Banner

REM ---- Pre-flight: tar.exe required for .tar.gz extraction
call :Step "Pre-flight checks" "Checking required tools..."
where tar.exe >nul 2>nul
if errorlevel 1 (
    call :Fail "tar.exe not found on PATH." ^
               "This wizard needs tar.exe (common on Windows 10/11) or you must extract manually (e.g., 7-Zip), then rerun."
    goto :ENDFAIL
)

REM ---- Cleanup/resume logic (do NOT destroy good installs)
call :Step "Resume & Cleanup check" ^
           "If an environment folder exists but is incomplete (missing python.exe), it will be deleted and reinstalled." ^
           "If it already has python.exe, it will be kept and the wizard will continue from there."

call :EnsureFolderState "DeuteRater_python"
if errorlevel 1 goto :ENDFAIL

call :EnsureFolderState "shared_python\Kinetic_Lipidomics_env"
if errorlevel 1 goto :ENDFAIL


REM ---- Process environments one at a time (space saving)
call :ProcessOne "DeuteRaterEnvironment.tar.gz" "DeuteRater_python" "DeuteRater"
if errorlevel 1 goto :ENDFAIL

call :ProcessOne "Kinetic_Lipidomics.tar.gz" "shared_python\Kinetic_Lipidomics_env" "Kinetic Lipidomics"
if errorlevel 1 goto :ENDFAIL

call :Step "All done!" ^
           "Both environments are installed and ready." ^
           "This wizard will now delete itself."

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
echo This wizard installs two packaged Python environments (from conda-pack)
echo WITHOUT needing conda installed on this machine.
echo.
echo What will happen (one environment at a time to save space):
echo   1) Extract DeuteRaterEnvironment.tar.gz  ->  .\DeuteRater_python\
echo   2) Clear Read-only attributes (Windows safety)
echo   3) Run conda-unpack to fix paths after extraction (relocation fix)
echo   4) Delete DeuteRaterEnvironment.tar.gz to save space
echo   5) Extract Kinetic_Lipidomics.tar.gz     ->  .\shared_python\Kinetic_Lipidomics_env\
echo   6) Clear Read-only attributes (Windows safety)
echo   7) Run conda-unpack to fix paths after extraction (relocation fix)
echo   8) Delete Kinetic_Lipidomics.tar.gz to save space
echo   9) Delete this wizard when finished
echo.
echo Resume behavior:
echo   - If an env folder already exists AND contains python.exe, we keep it.
echo   - If an env folder exists BUT is missing python.exe, we delete it
echo     (it is treated as incomplete/corrupted) and reinstall.
echo.
echo Note:
echo   - conda-unpack edits files in-place; Windows may block this if files
echo     are read-only or temporarily locked (AV/indexer). This wizard
echo     clears read-only flags and retries conda-unpack automatically.
echo.
echo Please do not close this window until it completes.
echo.
echo ============================================================
echo.
exit /b 0


:Step
REM Usage: call :Step "Title" "Line1" ["Line2"] ["Line3"]
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
REM Usage: call :Fail "ErrorTitle" "HintLine1" ["HintLine2"]
setlocal
echo.
echo ============================================================
echo  [ERROR] %~1
echo ============================================================
if not "%~2"=="" echo  %~2
if not "%~3"=="" echo  %~3
echo.
echo The wizard will stop now. No further environments will be processed.
echo.
pause
endlocal
exit /b 1


:EnsureFolderState
REM Ensures target folder is either:
REM  - kept (if python.exe exists), or
REM  - deleted (if folder exists but python.exe missing), or
REM  - untouched (if folder doesn't exist)
setlocal EnableExtensions EnableDelayedExpansion

set "TARGETREL=%~1"
set "TARGET=%HERE%%TARGETREL%"

if not exist "%TARGET%" (
    call :Step "Check: %TARGETREL%" "Folder does not exist yet. It will be installed if its archive is present."
    endlocal & exit /b 0
)

if exist "%TARGET%\python.exe" (
    call :Step "Check: %TARGETREL%" "Found python.exe. Keeping existing environment (resume mode)."
    endlocal & exit /b 0
) else (
    call :Step "Check: %TARGETREL%" ^
               "Folder exists but python.exe is missing (incomplete install)." ^
               "Deleting folder so it can be reinstalled cleanly..."
    rmdir /s /q "%TARGET%" >nul 2>nul
    if exist "%TARGET%" (
        call :Fail "Could not delete incomplete folder: %TARGETREL%" ^
                   "Try closing any programs using files in that folder and rerun."
        endlocal & exit /b 1
    )
    call :Step "Check: %TARGETREL%" "Incomplete folder removed."
    endlocal & exit /b 0
)


:ClearReadOnly
REM Clears Read-only attributes recursively for a target folder
REM (helps avoid PermissionError during conda-unpack on Windows)
setlocal
set "FOLDER=%~1"
call :Step "Permissions" "Clearing Read-only attributes (if any)..." "%FOLDER%"
attrib -R "%FOLDER%\*" /S /D >nul 2>nul
endlocal
exit /b 0


:FlattenIfNested
REM If extraction produced a double-nested env:
REM   TARGET\LEAF\python.exe exists but TARGET\python.exe does not
REM then move contents up one level and remove the extra folder.
setlocal EnableExtensions EnableDelayedExpansion
set "TARGET=%~1"
for %%F in ("%TARGET%") do set "LEAF=%%~nxF"

if exist "%TARGET%\python.exe" (
    endlocal & exit /b 0
)

if exist "%TARGET%\!LEAF!\python.exe" (
    call :Step "Fix layout" ^
               "Detected nested folder: !LEAF!\!LEAF!\ (flattening to !LEAF!\ )" ^
               "This happens when the archive already contains a top-level folder."

    REM Use robocopy to move contents up reliably
    robocopy "%TARGET%\!LEAF!" "%TARGET%" /E /MOVE >nul
    REM Remove the now-empty nested folder
    rmdir /s /q "%TARGET%\!LEAF!" >nul 2>nul

    if exist "%TARGET%\python.exe" (
        call :Step "Fix layout" "Flatten complete."
        endlocal & exit /b 0
    ) else (
        call :Fail "Flattening attempted but python.exe still not found." ^
                   "Please inspect the extracted folder structure under:" ^
                   "%TARGET%"
        endlocal & exit /b 1
    )
)

endlocal & exit /b 0


:RunCondaUnpackWithRetries
REM Runs conda-unpack with retries (handles transient locks)
REM %~1 = target folder (env root)
REM %~2 = friendly label
setlocal EnableExtensions EnableDelayedExpansion
set "TARGET=%~1"
set "LABEL=%~2"
set "UNPACK=%TARGET%\Scripts\conda-unpack.exe"

if not exist "%UNPACK%" (
    call :Fail "%LABEL% conda-unpack not found." ^
               "Expected: %UNPACK%" ^
               "This archive may not have been packed correctly."
    endlocal & exit /b 1
)

set "RETRIES=6"
set "WAITSEC=2"

:TRY_AGAIN
call :Step "%LABEL%: Relocation fix (conda-unpack)" ^
           "Running conda-unpack (attempt !RETRIES!)..." ^
           "If Windows is scanning/locking files, retries may be needed."

call "%UNPACK%"
if not errorlevel 1 (
    endlocal & exit /b 0
)

set /a RETRIES-=1
if !RETRIES! LEQ 0 (
    call :Fail "%LABEL% conda-unpack failed." ^
               "Windows denied write access (read-only file or locked file)." ^
               "Close VS Code/Python, pause antivirus scanning for this folder, then rerun."
    endlocal & exit /b 1
)

echo [WARN] conda-unpack failed. Waiting %WAITSEC% seconds and retrying...
ping 127.0.0.1 -n %WAITSEC% >nul
goto :TRY_AGAIN


:ProcessOne
REM %~1 = archive filename (in same directory as wizard)
REM %~2 = target folder relative path (in same directory as wizard)
REM %~3 = friendly env label
setlocal EnableExtensions EnableDelayedExpansion

set "ARCHIVE=%~1"
set "TARGETREL=%~2"
set "LABEL=%~3"

set "ARCHIVEPATH=%HERE%%ARCHIVE%"
set "TARGET=%HERE%%TARGETREL%"
set "READYMARK=%TARGET%\.__condapack_ready__"

call :Step "Installing: %LABEL%" ^
           "Archive: %ARCHIVE%" ^
           "Target folder: %TARGETREL%\"

REM If target already installed (python.exe exists), skip extraction.
REM But ensure conda-unpack was finalized (marker).
if exist "%TARGET%\python.exe" (
    call :Step "%LABEL%: Already present" ^
               "Found %TARGETREL%\python.exe. Skipping extraction." ^
               "Ensuring relocation fix (conda-unpack) has been applied..."

    if exist "%READYMARK%" (
        call :Step "%LABEL%: Ready" "Marker found. Environment already finalized."
    ) else (
        call :ClearReadOnly "%TARGET%"
        call :RunCondaUnpackWithRetries "%TARGET%" "%LABEL%"
        if errorlevel 1 ( endlocal & exit /b 1 )
        echo ready>"%READYMARK%"
        call :Step "%LABEL%: Finalized" "conda-unpack completed; marker created."
    )

    REM If archive still exists, delete it to save space
    if exist "%ARCHIVEPATH%" (
        call :Step "%LABEL%: Cleanup" "Environment already installed; deleting leftover archive: %ARCHIVE%"
        del /f /q "%ARCHIVEPATH%" >nul 2>nul
        if exist "%ARCHIVEPATH%" (
            call :Fail "Could not delete archive: %ARCHIVE%" ^
                       "The file may be locked or permissions are restricted."
            endlocal & exit /b 1
        )
        call :Step "%LABEL%: Cleanup" "Archive deleted."
    ) else (
        call :Step "%LABEL%: Cleanup" "No archive found to delete (already removed or never present)."
    )

    endlocal & exit /b 0
)

REM Need archive if installing fresh
if not exist "%ARCHIVEPATH%" (
    call :Fail "%LABEL% archive missing." ^
               "Expected to find: %ARCHIVE%" ^
               "Place it next to this wizard and rerun."
    endlocal & exit /b 1
)

REM Create folder (nested dirs OK)
call :Step "%LABEL%: Prepare folder" "Creating: %TARGETREL%\"
mkdir "%TARGET%" >nul 2>nul
if errorlevel 1 (
    call :Fail "Failed to create target folder for %LABEL%." "%TARGET%"
    endlocal & exit /b 1
)

REM Extract
call :Step "%LABEL%: Extract" ^
           "Extracting (this may take a few minutes)..." ^
           "Please wait."

tar -xzf "%ARCHIVEPATH%" -C "%TARGET%"
if errorlevel 1 (
    call :Fail "%LABEL% extraction failed." ^
               "tar.exe returned an error while extracting:" ^
               "%ARCHIVE%"
    endlocal & exit /b 1
)

call :Step "%LABEL%: Extract" "Extraction complete."

REM Flatten double nesting if needed
call :FlattenIfNested "%TARGET%"
if errorlevel 1 ( endlocal & exit /b 1 )

REM Confirm python.exe exists after extraction (post-flatten)
if not exist "%TARGET%\python.exe" (
    call :Fail "%LABEL% install appears incomplete." ^
               "python.exe was not found after extraction in:" ^
               "%TARGETREL%\"
    endlocal & exit /b 1
)

REM Clear read-only attributes (prevents PermissionError during conda-unpack)
call :ClearReadOnly "%TARGET%"

REM Optional activation (temporary; not required but harmless)
if exist "%TARGET%\Scripts\activate.bat" (
    call :Step "%LABEL%: Activate (temporary)" "Activating environment in this installer window..."
    call "%TARGET%\Scripts\activate.bat"
) else (
    call :Step "%LABEL%: Activate (skipped)" "activate.bat not found; continuing."
)

REM Run conda-unpack with retries (required finalization step)
call :RunCondaUnpackWithRetries "%TARGET%" "%LABEL%"
if errorlevel 1 ( endlocal & exit /b 1 )

echo ready>"%READYMARK%"
call :Step "%LABEL%: Relocation fix" "conda-unpack completed; marker created."

REM Deactivate (cleanup)
if exist "%TARGET%\Scripts\deactivate.bat" (
    call :Step "%LABEL%: Deactivate (cleanup)" "Deactivating to keep the next install clean..."
    call "%TARGET%\Scripts\deactivate.bat"
)

REM Delete archive after success (space saving)
call :Step "%LABEL%: Cleanup" "Deleting archive to save space: %ARCHIVE%"
del /f /q "%ARCHIVEPATH%" >nul 2>nul
if exist "%ARCHIVEPATH%" (
    call :Fail "Could not delete archive: %ARCHIVE%" ^
               "The file may be locked or you may not have permissions."
    endlocal & exit /b 1
)

call :Step "%LABEL% installed" ^
           "Installed to: %TARGETREL%\" ^
           "Archive deleted: %ARCHIVE%"

endlocal & exit /b 0


:SelfDelete
call :Step "Cleanup" "Removing this wizard file..."
start "" /b cmd /c "ping 127.0.0.1 -n 2 >nul & del /f /q ""%ME%"""
exit /b 0


:ENDFAIL
exit /b 1
