
#!/usr/bin/env python3
"""
Auto-generate portable Windows .cmd launchers for Python scripts.

FINAL VERSION – for unzipped conda-packed environments.

Features:
- Generates ONLY .cmd (Windows)
- Uses RELATIVE paths so folders can be moved anywhere
- Locates shared_python\\Kinetic_Lipidomics_env\\python.exe by walking upward
- Uses an UNZIPPED conda-packed environment (fast startup)
- Keeps the command window open for error visibility
- No PYTHONHOME, no PYTHONPATH, no Tcl hacks
"""

from __future__ import annotations

from pathlib import Path
import os
import textwrap

# Relative location of the portable Python executable
SHARED_ENV_SUBPATH = (
    Path("shared_python") / "Kinetic_Lipidomics_env" / "python.exe"
)

# Optional safety boundary for upward traversal
REPO_ROOT_NAME = "Lipidomics_Kinetics_Workflow2"


def _find_portable_python(script_dir: Path) -> Path:
    """
    Walk upward from script_dir to find the portable Python executable.

    Stops at REPO_ROOT_NAME if present.
    """
    script_dir = script_dir.resolve()

    boundary_dir = None
    if REPO_ROOT_NAME in script_dir.parts:
        idx = script_dir.parts.index(REPO_ROOT_NAME)
        boundary_dir = Path(*script_dir.parts[: idx + 1]).resolve()

    for parent in (script_dir, *script_dir.parents):
        candidate = parent / SHARED_ENV_SUBPATH
        if candidate.exists():
            return candidate.resolve()

        if boundary_dir is not None and parent == boundary_dir:
            break

    raise FileNotFoundError(
        f"Could not find '{SHARED_ENV_SUBPATH}' walking upward from {script_dir}"
    )


def generate_launcher(py_script_path: str) -> Path:
    py_path = Path(py_script_path).resolve()

    if not py_path.exists():
        raise FileNotFoundError(f"Script not found: {py_path}")

    script_dir = py_path.parent
    script_name = py_path.name
    base_name = py_path.stem

    # Locate portable Python
    portable_python = _find_portable_python(script_dir)

    # Compute relative path from script directory
    rel_python = os.path.relpath(
        portable_python, start=script_dir
    ).replace("/", "\\")

    if rel_python.startswith(".\\"):
        rel_python = rel_python[2:]

    cmd_text = textwrap.dedent(f"""\
        @echo off
        setlocal EnableExtensions EnableDelayedExpansion

        echo ============================================
        echo   {base_name} – DEBUG MODE
        echo ============================================
        echo.

        REM Directory where this .cmd lives
        set "SCRIPT_DIR=%~dp0"

        REM Portable Python (relative)
        set "PYTHON_EXE=%SCRIPT_DIR%\\{rel_python}"

        if not exist "%PYTHON_EXE%" (
            echo ERROR: Portable Python not found:
            echo %PYTHON_EXE%
            echo.
            goto :end
        )

        REM Compute environment root
        for %%I in ("%PYTHON_EXE%\\..\\..") do set "ENV_DIR=%%~fI"

        REM Ensure local DLL resolution (Tk, numpy, scipy, etc.)
        set "PATH=%ENV_DIR%;%ENV_DIR%\\Library\\bin;%ENV_DIR%\\DLLs;%PATH%"

        echo Using Python:
        "%PYTHON_EXE%" -c "import sys; print(sys.executable); print(sys.version)"
        echo.

        echo Running script: {script_name}
        echo --------------------------------------------
        "%PYTHON_EXE%" "%SCRIPT_DIR%\\{script_name}" %*
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
    """)

    cmd_path = script_dir / f"{base_name}.cmd"
    cmd_path.write_text(cmd_text, encoding="utf-8")

    print(f"Generated: {cmd_path}")
    print(f"Python:    {portable_python}")

    return cmd_path


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":

    generate_launcher(r"Lipidomics_Kinetics_Workflow2.py")

    generate_launcher(r"Concatenate_positive_and_negative\Concatenate_positive_and_negative.py")

    generate_launcher(r"Generating_wide_dataframe\Generating_wide_dataframe.py")

    generate_launcher(r"isobar_handler_gui\isobar_gui.py")

    generate_launcher(r"Standardize_positive_and_negative_lipid_IDs\Standardize_positive_and_negative_lipid_IDs.py")

    generate_launcher(r"msconvert_helper\msconvert_helper.py")

    generate_launcher(r"MS_dial_to_Isobar_Handler\MS_dial_to_Isobar_Handler.py")

    generate_launcher(r"Creating_tidy_regression_matrix_with_CIs\Creating_tidy_regression_matrix_with_CIs.py")

    generate_launcher(r"Lipidomics_post_hoc_analysis1.2\main.py")

    generate_launcher(r"Regression_analysis\Regression_analysis.py")

    generate_launcher(r"Lipidomics_component_analysis\main.py")
    
