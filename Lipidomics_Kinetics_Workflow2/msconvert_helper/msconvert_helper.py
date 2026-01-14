#!/usr/bin/env python3
"""
msconvert_string_gui.py

Tiny Tkinter GUI that generates a one-line msconvert loop command.

- User enters path to msconvert.exe (or leaves blank if it's on PATH).
- User chooses OS (Windows batch vs bash).
- User specifies vendor file extension (*.d, *.raw, etc.).
- User toggles vendor peak picking.

If peak picking is ON:
    mkdir "Centroided mzMLs" ...
Else:
    mkdir "mzMLs" ...

Output string is copied to clipboard and also shown in the GUI.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import platform, shlex

# --- quoting helpers ---
def _quote_windows(path: str) -> str:
    if not path:
        return ""
    p = path.strip()
    if p.startswith('"') and p.endswith('"'):
        return p
    if " " in p or "\t" in p:
        return f'"{p}"'
    return p

def _quote_bash(path: str) -> str:
    return shlex.quote(path.strip()) if path.strip() else ""

def build_command(ms_path: str, os_choice: str, ext: str, peak: bool) -> str:
    """Builds a one-liner msconvert command (no loop needed)."""
    out_dir = "Centroided mzMLs" if peak else "mzMLs"

    # --- normalize msconvert path ---
    ms_path = ms_path.strip()
    if ms_path:
        if os_choice == "windows":
            if not ms_path.lower().endswith("msconvert.exe"):
                ms_path = ms_path.rstrip("\\/") + "\\msconvert.exe"
        else:
            if not ms_path.endswith("msconvert"):
                ms_path = ms_path.rstrip("/") + "/msconvert"

    # --- quoting helpers ---
    if os_choice == "windows":
        ms = _quote_windows(ms_path) if ms_path else "msconvert.exe"
        peak_part = ' --filter "peakPicking true 1-"' if peak else ""
        return f'{ms} "*.{ext}" --mzML{peak_part} --zlib --64 -o ".\\{out_dir}"'
    else:
        ms = _quote_bash(ms_path) if ms_path else "msconvert"
        peak_part = "--filter 'peakPicking true 1-'" if peak else ""
        return f'{ms} ./*.{ext} --mzML {peak_part} --zlib --64 -o ./{shlex.quote(out_dir)}'


# --- GUI ---
class MSConvertStringGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("msconvert loop command builder")
        self.resizable(False, False)

        # variables
        self.var_ms = tk.StringVar()
        self.var_os = tk.StringVar(value="windows" if platform.system().lower().startswith("win") else "bash")
        self.var_ext = tk.StringVar(value="d")  # default vendor extension
        self.var_peak = tk.BooleanVar(value=True)

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        ttk.Label(self, text="Path to msconvert executable:").grid(row=0, column=0, sticky="w", **pad)
        ent = ttk.Entry(self, textvariable=self.var_ms, width=60)
        ent.grid(row=1, column=0, sticky="ew", padx=(10,0))
        ttk.Button(self, text="Browse…", command=self._browse).grid(row=1, column=1, sticky="e", padx=(6,10))

        os_frame = ttk.LabelFrame(self, text="Operating System")
        os_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(4,0))
        ttk.Radiobutton(os_frame, text="Windows (batch .cmd)", variable=self.var_os, value="windows").pack(anchor="w", padx=8, pady=2)
        ttk.Radiobutton(os_frame, text="macOS/Linux (bash .sh)", variable=self.var_os, value="bash").pack(anchor="w", padx=8, pady=2)

        ext_frame = ttk.Frame(self)
        ext_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=(4,0))
        ttk.Label(ext_frame, text="File extension to convert (without *.):").pack(side="left")
        ttk.Entry(ext_frame, textvariable=self.var_ext, width=10).pack(side="left", padx=(6,0))

        ttk.Checkbutton(self, text="Include vendor peak picking filter", variable=self.var_peak).grid(row=4, column=0, columnspan=2, sticky="w", padx=10, pady=(6,0))

        ttk.Button(self, text="Generate & Copy to Clipboard", command=self._copy).grid(row=5, column=0, columnspan=2, pady=(12,10))

        self.txt = tk.Text(self, width=95, height=6, wrap="word", padx=6, pady=6)
        self.txt.grid(row=6, column=0, columnspan=2, padx=10, pady=(0,10))
        self.txt.configure(state="disabled", background="#0b1020", foreground="#d6ffd6", font=("Courier New", 10))

    def _browse(self):
        path = filedialog.askopenfilename(title="Select msconvert executable", filetypes=[("Executables", "*.exe"), ("All files", "*.*")])
        if path:
            self.var_ms.set(path)

    def _copy(self):
        ms = self.var_ms.get().strip()
        ext = self.var_ext.get().strip() or "d"
        cmd = build_command(ms, self.var_os.get(), ext, self.var_peak.get())
        self.clipboard_clear()
        self.clipboard_append(cmd)
        self._set_preview(cmd)
        messagebox.showinfo("Copied", "Command has been copied to clipboard!")

    def _set_preview(self, text: str):
        self.txt.configure(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", text)
        self.txt.configure(state="disabled")

if __name__ == "__main__":
    MSConvertStringGUI().mainloop()
