#!/usr/bin/env python3
"""
Workflow GUI — embed renderer inside the same Tk window / same process.

Patched version:
 - PyInstaller-safe asset/script resolution
 - Supports run:script.py and run:tool.exe with one HTML syntax
 - .exe files run NON-BLOCKING (Option A)
 - .py files run normally with Python and block
 - HTMLS_DIR resolves correctly under PyInstaller

EXE LAUNCH FIX:
 - Launch executables the same way as run_deuterater.py:
   * Windows: clean environment (strip PYTHON/CONDA/JUPYTER contamination)
   * Linux/macOS: launch via wine
"""
from __future__ import annotations
import sys
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
from pathlib import Path
from typing import List, Optional, Tuple
import re
import tempfile
import os
import atexit
import urllib.parse
import shutil

# ---------------------------------------------------------------------------
# UNIVERSAL PATH RESOLUTION FOR PYINSTALLER + NORMAL PYTHON
# ---------------------------------------------------------------------------

def get_base_dir() -> Path:
    """
    Returns asset base directory:
      - Running from Python: directory of this file
      - PyInstaller onefile/onedir: sys._MEIPASS
    """
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent

def resolve_runtime_path(relative_path: str) -> Path:
    """
    Makes resource paths work in both normal Python and PyInstaller bundles.
    """
    return (get_base_dir() / relative_path).resolve()

# ---------------------------------------------------------------------------
# EXE LAUNCHING (MATCH run_deuterater.py)
# ---------------------------------------------------------------------------


def _launch_exe_like_run_deuterater(path: Path) -> None:
    """
    Windows-only launcher.

    Launches:
      - .exe directly with a minimal clean environment
      - .cmd/.bat via cmd.exe /c (so the batch file can set env vars)
    """
    suffix = path.suffix.lower()

    # Run batch files directly so their environment setup happens
    if suffix in (".cmd", ".bat"):
        env = os.environ.copy()

        # Strip common env vars that can hijack Python behavior
        for k in (
            "PYTHONPATH",
            "PYTHONHOME",
            "CONDA_PREFIX",
            "CONDA_DEFAULT_ENV",
            "VIRTUAL_ENV",
            "JUPYTER_PATH",
            "JUPYTER_CONFIG_DIR",
        ):
            env.pop(k, None)

        subprocess.Popen(
            ["cmd.exe", "/c", str(path)],
            cwd=str(path.parent),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
        )
        return

    # Regular .exe (or anything else executable) with a minimal clean env
    clean_env = {
        k: os.environ.get(k, "")
        for k in ("SYSTEMROOT", "WINDIR", "PATH", "COMSPEC", "PATHEXT", "TEMP", "TMP")
    }

    subprocess.Popen(
        [str(path)],
        cwd=str(path.parent),
        env=clean_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=False,
    )

# ---------------------------------------------------------------------------

# Renderer availability
CEF_AVAILABLE = False
TKINTERWEB_AVAILABLE = False
TKHTMLVIEW_AVAILABLE = False

try:
    from cefpython3 import cefpython as cef
    CEF_AVAILABLE = True
except Exception:
    pass

try:
    from tkinterweb import HtmlFrame
    TKINTERWEB_AVAILABLE = True
except Exception:
    pass

try:
    from tkhtmlview import HTMLLabel
    TKHTMLVIEW_AVAILABLE = True
except Exception:
    pass

# ---------------------------------------------------------------------------
# DIRECTORIES
# ---------------------------------------------------------------------------
RUN_AUTO_COMPLETE = True
BASE_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
HTMLS_DIR = (BASE_DIR / "htmls")
HTMLS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------

STEPS = [
    ("mzML Conversion", "01_mzml_conversion.html"),
    ("Preparing Guidefile for Isobar Handler/DeuteRater", "02_prepare_guidefile.html"),
    ("Isobar Handler", "03_isobar_handler.html"),
    ("Standardizing Polarity Syntax", "04_standardize_naming.html"),
    ("DeuteRater Extraction Step", "05_deuterater_extraction.html"),
    ("DeuteRater Kinetic Analysis", "06_kinetic_analysis.html"),
    ("Concatenating Standardized ID files", "07_concatenate_metadata.html"),
    ("Wide Metabolic Metric Dataframe", "08_wide_metric_dataframe.html"),
    ("Volcano and Conformity Plots", "09_posthoc_analysis.html"),
    ("Tidy Regression Matrix", "10_tidy_regression.html"),
    ("nₗ Component Deconvolution", "11_component_deconvolution.html"),
    ("Biological Variable Coefficient Estimation", "12_biovar_coeff_estimation.html"),
]

MZML_STEP_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial;padding:18px}}
h1{{color:#1f618d}}
.runbox{{padding:12px;border:1px dashed #1f618d;display:inline-block}}
</style></head><body>
<h1>{title}</h1>
<p>Example content.</p>
<a href="run:mzml_convert.py"><div class="runbox">Run mzML conversion</div></a>
</body></html>
"""

GENERIC_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title></head>
<body><h1>{title}</h1><p>Replace this file in htmls/ with your HTML.</p></body></html>
"""

def ensure_htmls():
    HTMLS_DIR.mkdir(exist_ok=True)
    for i, (title, fname) in enumerate(STEPS):
        p = HTMLS_DIR / fname
        if not p.exists():
            p.write_text((MZML_STEP_HTML if i == 0 else GENERIC_HTML).format(title=title),
                         encoding="utf-8")

class Step:
    def __init__(self, index:int, title:str, filename:str):
        self.index = index
        self.title = title
        self.filename = filename
        self.html_path = HTMLS_DIR / filename
        self.done = False
        self.nav_button = None
        self.check_var = None

# ---------------------------------------------------------------------------
# CEF EMBED CLASS
# ---------------------------------------------------------------------------
class CefEmbed:
    def __init__(self, parent_frame: tk.Frame):
        self.parent = parent_frame
        self.browser = None
        self._initialized = False

    def initialize(self):
        if CEF_AVAILABLE:
            cef.Initialize({})
            self._initialized = True

    def create_browser(self, url: str):
        self.parent.update_idletasks()
        window_info = cef.WindowInfo()
        handle = self.parent.winfo_id()
        window_info.SetAsChild(handle)
        self.browser = cef.CreateBrowserSync(window_info, url=url)
        self._message_loop_work()

    def load_url(self, url: str):
        if self.browser:
            self.browser.LoadUrl(url)
        else:
            self.create_browser(url)

    def _message_loop_work(self):
        if not CEF_AVAILABLE:
            return
        try:
            cef.MessageLoopWork()
        except Exception:
            pass
        self.parent.after(10, self._message_loop_work)

    def shutdown(self):
        if CEF_AVAILABLE and self._initialized:
            try:
                cef.Shutdown()
            except Exception:
                pass

# ---------------------------------------------------------------------------
# MAIN GUI CLASS
# ---------------------------------------------------------------------------
class WorkflowGUI(tk.Tk):
    def __init__(self, steps: List[Step], security_ok: bool):
        super().__init__()
        self.security_ok = security_ok  # ← stored for use inside the GUI

        self.title("Kinetic Lipidomics Workflow")
        self.geometry("1150x760")
        self.minsize(900,600)

        try:
            ttk.Style(self).theme_use("clam")
        except Exception:
            pass

        self.steps = steps
        self.current_index = 0
        self._detected_run_script: Optional[str] = None
        self.last_temp_html: Optional[Path] = None

        if CEF_AVAILABLE:
            self.engine = "cef"
        elif TKINTERWEB_AVAILABLE:
            self.engine = "tkinterweb"
        elif TKHTMLVIEW_AVAILABLE:
            self.engine = "tkhtmlview"
        else:
            self.engine = "text"

        self.cef_embed = None
        self._build_ui()
        
        if not self.security_ok:
            # Example: append a warning to the window title
            self.title(self.title() + "  —  Security key mismatch")


        if self.engine == "cef":
            try:
                self.cef_embed = CefEmbed(self.browser_holder)
                self.cef_embed.initialize()
            except Exception:
                self.engine = "tkinterweb" if TKINTERWEB_AVAILABLE else (
                    "tkhtmlview" if TKHTMLVIEW_AVAILABLE else "text"
                )

        self._update_ui()
        atexit.register(self._cleanup)

    # -----------------------------------------------------------------------
    # UI setup
    # -----------------------------------------------------------------------


    def _build_ui(self):
        # Top bar
        top = ttk.Frame(self, padding=(12, 8))
        top.pack(side="top", fill="x")
    
        base_title = "Kinetic Lipidomics analysis — Guided workflow"
    
        # Create/update styles. On macOS 'aqua', foreground colors are often ignored,
        # switching to 'clam' helps ensure colors are applied.
        style = ttk.Style(self)
        if not self.security_ok:
            try:
                style.theme_use('clam')
            except tk.TclError:
                pass
            style.configure("Warning.TLabel", foreground="red", font=("Segoe UI", 14, "bold"))
    
        # Title: black base + red warning suffix (when needed)
        ttk.Label(top, text=base_title, font=("Segoe UI", 14, "bold")).pack(side="left", anchor="w")
        if not self.security_ok:
            ttk.Label(top, text="  —  Security key mismatch; components of this program have been edited.\nConsider going to https://github.com/JC-Price for a new distribution.", style="Warning.TLabel").pack(side="left", anchor="w")
    
    
        # Progress widgets
        self.progress = ttk.Progressbar(top, length=360, mode="determinate")
        self.progress.pack(side="right", padx=(8, 0))
        self.progress_label = ttk.Label(top, text="0%")
        self.progress_label.pack(side="right", padx=(6, 0))
    
        # Main split
        main = ttk.Frame(self, padding=8)
        main.pack(fill="both", expand=True)
    
        # Left: workflow navigation
        left = ttk.Frame(main)
        left.pack(side="left", fill="y", padx=(0, 8))
    
        ttk.Label(left, text="Workflow steps", font=("Segoe UI", 11, "bold")).pack(anchor="nw", pady=(0, 6))
    
        canvas = tk.Canvas(left, borderwidth=0, width=420)
        vsb = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="y", expand=True)
    
        self.steps_container = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=self.steps_container, anchor="nw")
    
        # Fix: use "<Configure>" (not HTML-escaped)
        self.steps_container.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    
        for s in self.steps:
            frame = ttk.Frame(self.steps_container, padding=6)
            frame.pack(fill="x", pady=3)
    
            nf = tk.Frame(frame, width=28, height=28, bg="#1f618d")
            nf.pack_propagate(False)
            nf.pack(side="left", padx=(0, 8))
            tk.Label(
                nf,
                text=f"{s.index+1}",
                bg="#1f618d",
                fg="white",
                font=("Segoe UI", 10, "bold")
            ).pack(fill="both", expand=True)
    
            s.nav_button = ttk.Button(
                frame,
                text=s.title,
                width=36,
                command=lambda idx=s.index: self.go_to_step(idx)
            )
            s.nav_button.pack(side="left", padx=(0, 8))
    
            s.check_var = tk.BooleanVar(master=self, value=False)
            ttk.Checkbutton(frame, variable=s.check_var, state="disabled").pack(side="left")
    
        # Right: content area
        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)
    
        header = ttk.Frame(right)
        header.pack(fill="x")
        self.section_title = ttk.Label(header, font=("Segoe UI", 12, "bold"))
        self.section_title.pack(anchor="w")
    
        self.content_area = ttk.Frame(right, padding=6)
        self.content_area.pack(fill="both", expand=True)
        self.browser_holder = tk.Frame(self.content_area, bg="white")
        self.browser_holder.pack(fill="both", expand=True)
    
        # Footer
        footer = ttk.Frame(right)
        footer.pack(fill="x", pady=6)
    
        self.footer_left = ttk.Frame(footer)
        self.footer_left.pack(side="left", padx=6)
    
        self.run_button_container = ttk.Frame(self.footer_left)
        self.run_button_container.pack(side="left")
    
        ttk.Button(
            self.footer_left,
            text="Open in browser",
            command=self._open_in_system_browser
        ).pack(side="left", padx=6)
    
        self.engine_label = ttk.Label(self.footer_left, text=f"Renderer: {self.engine}")
        self.engine_label.pack(side="left", padx=6)
    
        self.complete_btn = ttk.Button(footer, text="Complete step", command=self.complete_current_step)
        self.complete_btn.pack(side="right")
    
        self.prev_btn = ttk.Button(footer, text="Go to previous step", command=self.go_to_previous_step)
        self.prev_btn.pack(side="right", padx=(0, 6))


    # -----------------------------------------------------------------------
    # UI updates
    # -----------------------------------------------------------------------
    def _update_ui(self):
        total = len(self.steps)
        done_count = sum(1 for s in self.steps if s.done)
        percent = (done_count * 100) // total
        self.progress["value"] = percent
        self.progress_label.config(text=f"{percent}%")

        for s in self.steps:
            s.nav_button.state(["disabled"] if s.done else ["!disabled"])
            s.check_var.set(s.done)

        current = self.steps[self.current_index]
        self.section_title.config(text=f"Step {current.index+1}: {current.title}")
        self._load_step_html(current)
        self.prev_btn.config(state="normal" if self.current_index > 0 else "disabled")

    # -----------------------------------------------------------------------
    # LOAD HTML + detect run:SCRIPT
    # -----------------------------------------------------------------------
    def _load_step_html(self, step: Step):
        import re
        import base64
        import urllib.parse
        from pathlib import Path
    
        # Clear previous run buttons
        for w in self.run_button_container.winfo_children():
            w.destroy()
        self._detected_run_script = None
    
        # Read the HTML
        if step.html_path.exists():
            try:
                html = step.html_path.read_text(encoding="utf-8")
            except Exception as exc:
                html = f"<h2>Error reading HTML</h2><pre>{exc}</pre>"
        else:
            html = f"<h2>MISSING HTML</h2><p>{step.html_path}</p>"
    
        # ----------------------------
        # Convert <img src="..."> to base64
        # ----------------------------
        def img_to_base64(img_path: Path) -> str:
            if not img_path.exists():
                return ""  # fallback: empty src
            with img_path.open("rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            suffix = img_path.suffix.lower()
            if suffix == ".png":
                mime = "image/png"
            elif suffix in (".jpg", ".jpeg"):
                mime = "image/jpeg"
            elif suffix == ".gif":
                mime = "image/gif"
            else:
                mime = "application/octet-stream"
            return f"data:{mime};base64,{encoded}"
    
        def embed_images(html_text: str) -> str:
            # Replace all <img src="..."> with base64 but keep other attributes
            def repl(m):
                full_tag = m.group(0)
                src = m.group(1)
                img_path = (HTMLS_DIR / src).resolve()
                b64 = img_to_base64(img_path)
                if not b64:
                    return full_tag  # leave as-is if missing
        
                # Replace only the src attribute
                return re.sub(r'src=["\'][^"\']+["\']', f'src="{b64}"', full_tag)
        
            return re.sub(r'<img\s+[^>]*src=["\']([^"\']+)["\'][^>]*>', repl, html_text)
    
        html = embed_images(html)
    
        # ----------------------------
        # Detect run: script reference
        # ----------------------------
        m = re.search(r'href=["\']run:([^"\']+)["\']', html)
        if m:
            script_name = m.group(1)
            candidate = resolve_runtime_path(script_name)
            if not candidate.exists():
                candidate = (BASE_DIR / script_name).resolve()
            if candidate.exists():
                self._detected_run_script = str(candidate)
    
        if self._detected_run_script:
            label = f"Run {Path(self._detected_run_script).name}"
            label_match = re.search(r'data-label=["\']([^"\']+)["\']', html)
            if label_match:
                label = label_match.group(1)
            ttk.Button(
                self.run_button_container,
                text=label,
                command=lambda p=self._detected_run_script: self._run_and_maybe_complete(p)
            ).pack(side="left")
        else:
            if m:
                ttk.Button(
                    self.run_button_container,
                    text="Run script (missing)",
                    state="disabled"
                ).pack(side="left")
    
        # ----------------------------
        # Write a temp HTML file for "Open in Browser"
        # ----------------------------
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html",
                                              prefix=f"step_{step.index+1}_",
                                              mode="w", encoding="utf-8")
            tmp.write(html)  # write base64-embedded HTML
            self.last_temp_html = Path(tmp.name)
            tmp.close()
        except Exception:
            self.last_temp_html = None
    
        # ----------------------------
        # Clear previous view
        # ----------------------------
        for widget in self.browser_holder.winfo_children():
            widget.destroy()
    
        # ----------------------------
        # Render HTML using available engine
        # ----------------------------
        data_uri = "data:text/html;charset=utf-8," + urllib.parse.quote(html)
    
        if self.engine == "cef" and self.cef_embed:
            try:
                self.cef_embed.load_url(data_uri)
                return
            except Exception:
                pass
    
        if self.engine == "tkinterweb" and TKINTERWEB_AVAILABLE:
            try:
                frame = HtmlFrame(self.browser_holder)
                frame.pack(fill="both", expand=True)
                frame.load_html(html)
                return
            except Exception:
                pass
    
        if self.engine == "tkhtmlview" and TKHTMLVIEW_AVAILABLE:
            try:
                HTMLLabel(self.browser_holder, html=html).pack(fill="both", expand=True)
                return
            except Exception:
                pass
    
        # fallback - text only
        self._show_text_in_holder(self._strip_html(html))



    # -----------------------------------------------------------------------
    # RUN SCRIPTS (.py = blocking, .exe = non-blocking)
    # -----------------------------------------------------------------------

    def run_script(self, script_path: str) -> Optional[Tuple[int, str, str]]:
        path = Path(script_path)
        if not path.exists():
            messagebox.showerror("Script not found", script_path)
            return None
    
        suffix = path.suffix.lower()
    
        # Windows launchables → non-blocking
        # (.sh support removed)
        if suffix in (".exe", ".cmd", ".bat"):
            try:
                _launch_exe_like_run_deuterater(path)
                return 0, "", ""
            except Exception as exc:
                messagebox.showerror("Execution Error", f"Error running {path}:\n{exc}")
                return None
    
        # Python → blocking
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(path.parent) + os.pathsep + env.get("PYTHONPATH", "")
            proc = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True,
                text=True,
                cwd=str(path.parent),
                env=env
            )
            return proc.returncode, proc.stdout, proc.stderr
        except Exception as exc:
            messagebox.showerror("Execution Error", f"Error running {path}:\n{exc}")
            return None

    


    # -----------------------------------------------------------------------

    def _run_and_maybe_complete(self, script_path: str):
        result = self.run_script(script_path)
        if result is None:
            return
    
        rc, out, err = result
        self._show_script_output(script_path, out, err, rc)
    
        # ✅ Do NOT auto-complete or auto-advance on Run.
        # Progress is manual via the "Complete step" button.
        return


    # -----------------------------------------------------------------------
    def _show_script_output(self, script, stdout_text, stderr_text, returncode):
        # You can add a real output window here if desired.
        pass

    # -----------------------------------------------------------------------
    def _open_in_system_browser(self):
        if not self.last_temp_html or not self.last_temp_html.exists():
            return
        try:
            import webbrowser
            webbrowser.open(self.last_temp_html.as_uri())
        except Exception as exc:
            messagebox.showerror("Browser Error", str(exc))

    # -----------------------------------------------------------------------
    def _cleanup(self):
        if self.last_temp_html and self.last_temp_html.exists():
            try: os.unlink(self.last_temp_html)
            except: pass
        if CEF_AVAILABLE and self.cef_embed:
            try: self.cef_embed.shutdown()
            except: pass

    # Helpers
    def _show_text_in_holder(self, text:str):
        box = ScrolledText(self.browser_holder, wrap="word")
        box.insert("1.0", text)
        box.configure(state="disabled")
        box.pack(fill="both", expand=True)

    def _strip_html(self, html:str) -> str:
        html = re.sub(r"<(script|style).*?>.*?</\1>", "", html, flags=re.S)
        html = re.sub(r"<[^>]+>", "", html)
        return html.replace("&nbsp;"," ").replace("&amp;","&").strip()

    def go_to_step(self, idx:int):
        if 0 <= idx < len(self.steps):
            if idx < self.current_index:
                for i in range(idx, len(self.steps)):
                    self.steps[i].done = False
                    self.steps[i].check_var.set(False)
            self.current_index = idx
            self._update_ui()

    def go_to_previous_step(self):
        if self.current_index > 0:
            new_idx = self.current_index - 1
            for i in range(new_idx, len(self.steps)):
                self.steps[i].done = False
            self.current_index = new_idx
            self._update_ui()

    def complete_current_step(self):
        idx = self.current_index
        for i in range(idx + 1):
            self.steps[i].done = True
        if idx + 1 < len(self.steps):
            self.current_index += 1
        self._update_ui()
        
        

def run_security_gate_read_mode() -> bool:
    """
    Runs ..\\misc\\creating_security_key.cmd in READ MODE for the parent folder.
    Returns True/False based on the key match. Runs BEFORE any GUI is built.
    """
    try:
        this_dir = Path(__file__).resolve().parent
    except Exception:
        # If __file__ is unavailable, allow GUI
        return True

    parent_dir = this_dir.parent                      # parent folder that contains 'misc'
    batch = parent_dir / "misc" / "creating_security_key.cmd"

    # .cmd is Windows-only; if not Windows or batch missing, allow GUI.
    if os.name != "nt" or not batch.exists():
        return True
    

    try:
        # Call the batch quietly in read mode; expect "True" or "False" on stdout
        proc = subprocess.run(
            ["cmd.exe", "/c", str(batch), "--read", str(parent_dir)],
            capture_output=True,
            text=True
        )
        # Find last "True"/"False" token in stdout to be robust to any extra lines
        lines = (proc.stdout or "").strip().splitlines()
        for line in reversed(lines):
            s = line.strip()
            if s in ("True", "False"):
                return s == "True"
        # If not found, default to False (conservative)
        return False
    except Exception:
        # On error, be conservative or permissive. Here: conservative.
        return False



import os
import subprocess
from pathlib import Path
import logging

# Initialize logging once; use env var to toggle verbosity.
# Example: set SECURITY_DEBUG=1 to enable DEBUG logs.
_log_level = logging.DEBUG if os.environ.get("SECURITY_DEBUG") else logging.INFO
logging.basicConfig(
    level=_log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("security_gate")


def run_security_gate_read_mode(verbose: bool = False) -> bool:
    """
    Runs ..\\misc\\creating_security_key.cmd in READ MODE for the parent folder.
    Returns True/False based on the key match. Runs BEFORE any GUI is built.

    Debug logging is emitted at each step. Set verbose=True or SECURITY_DEBUG=1
    to see detailed diagnostics.
    """
    # Upgrade level if verbose explicitly requested
    if verbose and logger.level > logging.DEBUG:
        logger.setLevel(logging.DEBUG)

    # 1) Resolve __file__ and paths
    try:
        this_dir = Path(__file__).resolve().parent
        parent_dir = this_dir
        batch = parent_dir / "misc" / "creating_security_key.cmd"
        logger.debug("Resolved paths | this_dir=%s | parent_dir=%s | batch=%s",
                     this_dir, parent_dir, batch)
    except Exception as e:
        logger.warning(
            "Unable to resolve __file__ or paths (%s). Allowing GUI as fallback.",
            e, exc_info=True
        )
        return True  # Keep permissive as you did for __file__ unavailable

    # 2) Windows + existence check
    if os.name != "nt":
        logger.info("Non-Windows OS detected (os.name=%s). Allowing GUI.", os.name)
        return True

    if not batch.exists():
        logger.warning("Batch file not found at %s. Allowing GUI.", batch)
        return True

    if not os.access(batch, os.R_OK):
        logger.warning("Batch file at %s is not readable. Defaulting to False.", batch)
        return False

    # 3) Build command
    # Note: passing extra args to .cmd requires the script to parse %* or %1 %2 etc.
    cmd = ["cmd.exe", "/c", str(batch), "--read", str(parent_dir)]
    logger.debug("Executing command: %s", cmd)

    # 4) Run subprocess and capture output
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,              # decode to str per locale
            encoding="utf-8",       # prefer UTF-8 to avoid mojibake; adjust if needed
            errors="replace",       # never raise on decoding issues
            timeout=45             # keep it bounded; adjust as needed
        )
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        logger.debug("Subprocess returncode=%s", proc.returncode)
        logger.debug("Subprocess STDOUT:\n%s", stdout or "<empty>")
        if stderr:
            logger.debug("Subprocess STDERR:\n%s", stderr)

        # 5) Parse "True"/"False" tokens from last to first line (robust to extra output)
        lines = stdout.splitlines()
        for idx, line in enumerate(reversed(lines), start=1):
            s = line.strip()
            logger.debug("Parsing line -%d: %r", idx, s)
            if s in ("True", "False"):
                result = (s == "True")
                logger.info("Security gate decision from batch output: %s", result)
                return result

        logger.warning(
            "No explicit 'True'/'False' token found in STDOUT. Defaulting to False."
        )
        return False

    except subprocess.TimeoutExpired as te:
        logger.error("Security key script timed out: %s. Defaulting to False.", te)
        return False
    except Exception as e:
        logger.error("Error running security key script: %s. Defaulting to False.", e, exc_info=True)
        return False



# ---------------------------------------------------------------------------

def main():
    security_ok = run_security_gate_read_mode(verbose=True)
    ensure_htmls()
    steps = [Step(i, title, fname) for i, (title, fname) in enumerate(STEPS)]
    WorkflowGUI(steps, security_ok=security_ok).mainloop()


if __name__ == "__main__":
    main()
