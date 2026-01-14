# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 Bradley Naylor, Christian Andersen, Michael Porter, Kyle Cutler, Chad Quilling, Benjamin Driggs,
    Coleman Nielsen, Martin Sorensen, J.C. Price, and Brigham Young University
Credit: ChatGPT - helped with building GUI from the ground up, debugging, general flow
All rights reserved.
Redistribution and use in source and binary forms,
with or without modification, are permitted provided
that the following conditions are met:
    * Redistributions of source code must retain the
      above copyright notice, this list of conditions
      and the following disclaimer.
    * Redistributions in binary form must reproduce
      the above copyright notice, this list of conditions
      and the following disclaimer in the documentation
      and/or other materials provided with the distribution.
    * Neither the author nor the names of any contributors
      may be used to endorse or promote products derived
      from this software without specific prior written
      permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DaIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import ttkbootstrap as tb
from ttkbootstrap.constants import *

from concurrent.futures import ThreadPoolExecutor
import queue, time
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import isobar_core
import isobar_handler
import random
from pathlib import Path

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = ImageTk = None  # fallback if Pillow isn't installed

# Point these to your files
LOGO_PNG     = Path("assets/logo.png")   # your single logo PNG
APP_ICON_ICO = Path("assets/app.ico")    # optional Windows .ico


class TaskRunner:
    def __init__(self, root):
        self.root = root
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.fut = None
        self.q = queue.Queue()

    def run(self, func, *args, on_done=None, on_error=None):
        if self.fut and not self.fut.done():
            messagebox.showinfo("Busy", "A task is already running.")
            return

        def job():
            try:
                res = func(self.progress, *args)
                self.q.put(("done", res))
            except Exception as e:
                self.q.put(("error", e))

        self.fut = self.pool.submit(job)
        self._pump(on_done, on_error)

    def progress(self, msg, pct=None):
        self.q.put(("progress", (msg, pct)))

    def _pump(self, on_done, on_error):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "progress":
                    self.root._progress_payload = payload
                    self.root.event_generate("<<Progress>>", when="tail")
                elif kind == "console":
                    self.root._progress_payload = ("console", payload)
                    self.root.event_generate("<<Progress>>", when="tail")
                elif kind == "done":
                    if on_done: on_done(payload)
                elif kind == "error":
                    if on_error: on_error(payload)
        except queue.Empty:
            pass
        if self.fut and not self.fut.done():
            self.root.after(50, self._pump, on_done, on_error)


class StreamRedirector:
    """
    Redirects writes from worker thread into GUI log.
    queue_put expects: self.queue_put(("console", line))
    """
    def __init__(self, queue_put, wake_event=None):
        self.queue_put = queue_put
        self.wake_event = wake_event  # kept for compatibility
        self._buf = ""

    def write(self, txt):
        if not txt:
            return
        # tqdm uses '\r' a lot; treat it like a newline for the GUI log
        txt = txt.replace("\r", "\n")
        self._buf += txt
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line:
                self.queue_put(("console", line))

    def flush(self):
        if self._buf:
            self.queue_put(("console", self._buf))
            self._buf = ""


class App(tb.Window):
    def __init__(self):
        super().__init__(themename="flatly")
        self.title("Isobar Handler")
        self.geometry("1100x720")
        self.df = None
        self.view_df = None
        self.colmap = None

        self._live_line_mark = None

        # --- Brand strip (big logo + title) ---
        self._init_brand_strip()   # row 0

        # --- Controls bar (your buttons) ---
        bar = ttk.Frame(self, padding=(8,8,8,4))
        bar.grid(row=1, column=0, sticky="ew")
        self.columnconfigure(0, weight=1)

        self.btn_load = ttk.Button(bar, text="Load CSV", bootstyle="secondary",
                                   command=self.on_load_csv)
        self.btn_load.pack(side="left", padx=(0,8))
        self.btn_align = ttk.Button(bar, text="Automated Alignment", bootstyle="primary",
                                    command=self.on_align)
        self.btn_align.pack(side="left")
        self.btn_export_project = ttk.Button(bar, text="Export Project", bootstyle="success",
                                             command=self.on_export_project)
        self.btn_export_project.pack(side="left", padx=8)
        self.btn_load_project = ttk.Button(
            bar, text="Load Project", bootstyle="info", command=self.on_load_project
        )
        self.btn_load_project.pack(side="left", padx=8)
        self.btn_settings = ttk.Button(bar, text="Settings", bootstyle="warning",
                                       command=self.on_settings)
        self.btn_settings.pack(side="left")

        ttk.Label(bar, text="Theme:", padding=(12,0)).pack(side="left")
        themes = sorted(tb.Style().theme_names())
        self.theme_box = ttk.Combobox(bar, values=themes, width=12, state="readonly")
        self.theme_box.set("flatly")
        self.theme_box.pack(side="left", padx=4)
        self.theme_box.bind("<<ComboboxSelected>>", lambda e: self._switch_theme())

        # --- Tabs ---
        self.nb = ttk.Notebook(self)
        self.nb.grid(row=2, column=0, sticky="nsew", padx=8, pady=4)
        self.rowconfigure(2, weight=1)
        self._build_tabs()

        # --- Status bar ---
        status = ttk.Frame(self, padding=(8,4))
        status.grid(row=3, column=0, sticky="ew")
        self.status = ttk.Label(status, text="Ready")
        self.status.pack(side="left")
        self.pbar = ttk.Progressbar(status, length=220, mode="determinate",
                                    bootstyle="info-striped")
        self.pbar.pack(side="right")

        self.runner = TaskRunner(self)
        self.bind("<<Progress>>", self._on_progress)

        # shake detector state
        self._last_geo = None            # (x, y, w, h)
        self._shake_hist = []            # list of (timestamp, direction)
        self._shake_cooldown_until = 0.0 # prevent rapid re-triggers

        # listen for window move/resize
        self.bind("<Configure>", self._on_configure)

    # ---------- Branding ----------
    def _init_brand_strip(self):
        """Row 0: a colored strip with big logo + title, derived from logo palette."""
        bg, fg = self._brand_palette_from_logo(LOGO_PNG)

        s = tb.Style()
        s.configure("Brand.TFrame", background=bg)
        s.configure("Brand.TLabel", background=bg, foreground=fg)
        s.configure("Brand.Title.TLabel", background=bg, foreground=fg,
                    font=("Segoe UI", 20, "bold"))

        brand = ttk.Frame(self, style="Brand.TFrame")
        brand.grid(row=0, column=0, sticky="ew")
        brand.columnconfigure(2, weight=1)  # spacer flex

        # logo (big)
        self.logo_img = None
        self.logo_label = ttk.Label(brand, cursor="hand2", style="Brand.TLabel")
        self.logo_label.grid(row=0, column=0, sticky="w", padx=(12, 10), pady=8)
        self.logo_label.bind("<Button-1>", lambda e: self._show_about())
        self._load_logo(LOGO_PNG, target_h=56)  # bigger!

        # title
        self.title_label = ttk.Label(brand, text="Isobar Handler", style="Brand.Title.TLabel")
        self.title_label.grid(row=0, column=1, sticky="w", pady=8)

        # spacer (keeps height and balance)
        ttk.Label(brand, text="", style="Brand.TLabel").grid(row=0, column=2, sticky="ew")

        # set window icon (ico/PNG)
        self._set_window_icon()

    def _brand_palette_from_logo(self, path):
        """
        Return (bg_hex, fg_hex) derived from logo average color.
        Fallbacks to nice defaults if PIL or file unavailable.
        """
        default_bg = "#2c3e50"  # flatly-ish navbar color
        default_fg = "#ffffff"

        if Image is None or not path.exists():
            return default_bg, default_fg

        try:
            img = Image.open(path).convert("RGBA")
            img = img.resize((64, 64), Image.LANCZOS)
            pixels = [p for p in img.getdata() if p[3] > 10]
            if not pixels:
                return default_bg, default_fg

            r = sum(p[0] for p in pixels) // len(pixels)
            g = sum(p[1] for p in pixels) // len(pixels)
            b = sum(p[2] for p in pixels) // len(pixels)
            bg = f"#{r:02x}{g:02x}{b:02x}"

            # perceived brightness → choose readable text
            luma = int(0.2126*r + 0.7152*g + 0.0722*b)
            fg = "#111111" if luma > 185 else "#ffffff"
            return bg, fg
        except Exception:
            return default_bg, default_fg

    def _set_window_icon(self):
        try:
            if APP_ICON_ICO.exists():
                self.iconbitmap(str(APP_ICON_ICO))
        except Exception:
            pass
        try:
            if LOGO_PNG.exists():
                self.iconphoto(True, tk.PhotoImage(file=str(LOGO_PNG)))
        except Exception:
            pass

    def _load_logo(self, path: Path, target_h: int = 56):
        """Load and scale the toolbar logo. Uses Pillow if available; otherwise no scaling."""
        if not path.exists():
            return
        try:
            if Image is not None:
                img = Image.open(path)
                w, h = img.size
                if h > 0:
                    new_w = max(1, int(w * target_h / h))
                    img = img.resize((new_w, target_h), Image.LANCZOS)
                self.logo_img = ImageTk.PhotoImage(img)
                self.logo_label.configure(image=self.logo_img)
            else:
                # Fallback: no Pillow → no scaling, just load
                self.logo_img = tk.PhotoImage(file=str(path))
                self.logo_label.configure(image=self.logo_img)
        except Exception:
            pass

    def _show_about(self):
        messagebox.showinfo(
            "About Isobar Handler",
            "Isobar Handler\n\nFast alignment + EIC review\n(logo click = about)"
        )

    # ---------- Shake-to-theme ----------
    def _on_configure(self, ev):
        """
        Detect a quick left-right 'shake' of the window by watching position
        changes (Configure fires on move & resize — we ignore pure resizes).
        """
        now = time.time()
        x, y = self.winfo_x(), self.winfo_y()
        w, h = ev.width, ev.height

        # first measurement
        if self._last_geo is None:
            self._last_geo = (x, y, w, h)
            return

        lx, ly, lw, lh = self._last_geo
        self._last_geo = (x, y, w, h)

        # ignore if this event was just a resize
        if (w, h) != (lw, lh):
            return

        dx, dy = x - lx, y - ly

        # only count 'horizontal-ish' move chunks (tune thresholds here)
        HORIZ_MIN = 25     # px
        VERT_MAX  = 50     # px
        if abs(dx) < HORIZ_MIN or abs(dy) > VERT_MAX:
            return

        direction = 1 if dx > 0 else -1
        self._shake_hist.append((now, direction))
        WINDOW = 0.8
        self._shake_hist = [(t, d) for (t, d) in self._shake_hist if t >= now - WINDOW]

        MIN_CHANGES = 3
        if len(self._shake_hist) >= MIN_CHANGES + 1:
            changes = sum(
                1 for i in range(1, len(self._shake_hist))
                if self._shake_hist[i][1] != self._shake_hist[i - 1][1]
            )
            if changes >= MIN_CHANGES and now >= self._shake_cooldown_until:
                self._apply_random_theme()
                self._shake_hist.clear()
                self._shake_cooldown_until = now + 1.5  # cooldown seconds

    def _apply_random_theme(self):
        style = tb.Style()
        current = style.theme_use()
        themes = [t for t in style.theme_names() if t != current]
        if not themes:
            return
        new_theme = random.choice(themes)
        style.theme_use(new_theme)
        if hasattr(self, "theme_box"):
            try:
                self.theme_box.set(new_theme)
            except Exception:
                pass
        if hasattr(self, "status"):
            try:
                self.status.config(text=f"✨ Shook into theme: {new_theme}")
            except Exception:
                pass

    # ---------- Plot helpers ----------
    def _plot_message(self, text: str):
        self.ax.clear()
        self.ax.text(0.5, 0.5, text, ha="center", va="center",
                     transform=self.ax.transAxes, fontsize=12)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.canvas.draw_idle()

    def _set_live_line(self, text: str):
        """Update a single 'live' line at the end of the Align tab log."""
        log = self.log
        if self._live_line_mark is None:
            if log.index("end-1c") != "1.0":
                last_char = log.get("end-2c", "end-1c")
                if last_char != "\n":
                    log.insert("end", "\n")
            start = log.index("end-1c")
            log.insert("end", text)
            self._live_line_mark = start
        else:
            start = self._live_line_mark
            self.log.delete(start, "end-1c")
            self.log.insert("end", text)
        self.log.see("end")

    def _finalize_live_line(self):
        if self._live_line_mark is not None:
            last_char = self.log.get("end-2c", "end-1c")
            if last_char != "\n":
                self.log.insert("end", "\n")
            self._live_line_mark = None

    # ---------- Project load ----------
    def on_load_project(self):
        import isobar_handler
    
        # Clear preview cache in core
        isobar_core.clear_preview_cache()
    
        # Temporarily disable isobar_handler.display_buttons to avoid double UI updates
        saved_display_buttons = getattr(isobar_handler, "display_buttons", None)
        if saved_display_buttons:
            isobar_handler.display_buttons = lambda *a, **k: None
    
        try:
            # Ask the handler to load the project (handler will open its own dialog)
            # This prevents the GUI from showing its own askopenfilename dialog.
            isobar_handler.load_variables()
    
            # restore display_buttons if we changed it
            if saved_display_buttons:
                isobar_handler.display_buttons = saved_display_buttons
    
            # Grab the loaded dataframe from the handler
            df = getattr(isobar_handler, "original_df", None)
            if df is not None:
                try:
                    id_col, mz_col, name_col = self._resolve_columns(df)
                    if mz_col:
                        df[mz_col] = pd.to_numeric(df[mz_col], errors="coerce")
                    self.df = df
                    self.view_df = df
                    self.colmap = (id_col, mz_col, name_col)
                    # You can show the handler's internal path if it stores it, else generic label:
                    self._set_dataset_labels("(loaded project)")
                    self._populate_tree(self.view_df)
                    self.nb.select(self.explore)
                    self.status.config(text="Loaded project")
                except Exception as e:
                    messagebox.showinfo("Project loaded",
                                        f"Project loaded, but explorer table may not be updated: {e}")
            else:
                messagebox.showinfo("Project loaded", "Project loaded, but no original_df found.")
    
        except Exception as e:
            messagebox.showerror("Load failed", f"Could not load project:\n{e}")
        finally:
            # always restore display_buttons if not restored above (defensive)
            if saved_display_buttons and getattr(isobar_handler, "display_buttons", None) != saved_display_buttons:
                isobar_handler.display_buttons = saved_display_buttons

    # ---------- Tabs/UI ----------
    def _build_tabs(self):
        # Project tab
        self.project = ttk.Frame(self.nb, padding=12)
        self.nb.add(self.project, text="Project")
        proj_top = ttk.Frame(self.project)
        proj_top.pack(fill="x")
        ttk.Label(proj_top, text="Dataset:", bootstyle="secondary").pack(side="left")
        self.ds_label = ttk.Label(proj_top, text="(none)")
        self.ds_label.pack(side="left", padx=8)
        self.ds_stats = ttk.Label(self.project, text="Rows: 0 | Columns: 0",
                                  bootstyle="secondary")
        self.ds_stats.pack(anchor="w", pady=(6, 0))

        # Align tab
        self.align = ttk.Frame(self.nb, padding=12)
        self.nb.add(self.align, text="Align")
        self.log = tk.Text(self.align, height=12)
        self.log.pack(fill="both", expand=True)

        # Explorer tab
        self.explore = ttk.Frame(self.nb, padding=12)
        self.nb.add(self.explore, text="Explorer")
        self._build_explorer()

    def _build_explorer(self):
        bar = ttk.Frame(self.explore)
        bar.pack(fill="x", pady=(0,8))
        # General search
        ttk.Label(bar, text="Search:").pack(side="left")
        self.search_var = tk.StringVar()
        ttk.Entry(bar, textvariable=self.search_var, width=18).pack(side="left", padx=(2,8))
        # m/z filter
        ttk.Label(bar, text="m/z:").pack(side="left")
        self.mz_from = tk.StringVar()
        ttk.Entry(bar, textvariable=self.mz_from, width=8).pack(side="left")
        ttk.Label(bar, text="-").pack(side="left")
        self.mz_to = tk.StringVar()
        ttk.Entry(bar, textvariable=self.mz_to, width=8).pack(side="left", padx=(0,8))
        # Name filter
        ttk.Label(bar, text="Name:").pack(side="left")
        self.name_filter = tk.StringVar()
        ttk.Entry(bar, textvariable=self.name_filter, width=18).pack(side="left", padx=(2,8))
        # Export Selected button
        ttk.Button(bar, text="Export Selected", bootstyle="success",
                   command=self.export_selected_rows).pack(side="left", padx=(12,0))
        # Filter buttons
        ttk.Button(bar, text="Apply Filters", bootstyle="primary",
                   command=self._apply_advanced_filters).pack(side="left", padx=(6,0))
        ttk.Button(bar, text="Clear", bootstyle="secondary",
                   command=self._clear_advanced_filters).pack(side="left", padx=(6,0))

        table_frame = ttk.Frame(self.explore)
        table_frame.pack(fill="both", expand=True)
        cols = ("id", "mz", "name")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", selectmode="extended")
        self.tree.heading("id", text="Alignment ID")
        self.tree.heading("mz", text="m/z")
        self.tree.heading("name", text="Name")
        self.tree.column("id", width=430, anchor="w")
        self.tree.column("mz", width=120, anchor="center")
        self.tree.column("name", width=420, anchor="w")
        self.tree.pack(side="left", fill="both", expand=True)
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=vsb.set)
        vsb.pack(side="right", fill="y")

#        plot_frame = ttk.Frame(self.explore)
#        plot_frame.pack(fill="both", expand=True, pady=(8,0))
#        self.fig = Figure(figsize=(5,3))
#        self.ax = self.fig.add_subplot(111)
#        self.ax.set_title("EIC preview")
#        self.ax.set_xlabel("Retention time (min)")
#        self.ax.set_ylabel("Intensity (%)")
#        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
#        self.canvas.get_tk_widget().pack(fill="both", expand=True)
#        self.tree.bind("<<TreeviewSelect>>", self._on_select_id)

    def _switch_theme(self):
        tb.Style().theme_use(self.theme_box.get())

    def _set_dataset_labels(self, path):
        if self.df is None:
            self.ds_label.config(text="(none)")
            self.ds_stats.config(text="Rows: 0 | Columns: 0")
        else:
            self.ds_label.config(text=path if path else "(loaded)")
            self.ds_stats.config(text=f"Rows: {len(self.df):,} | Columns: {self.df.shape[1]}")

    def _resolve_columns(self, df: pd.DataFrame):
        cols = {c.lower(): c for c in df.columns}
        id_col = cols.get("alignment id") or cols.get("alignment_id")
        mz_col = (cols.get("precursor m/z") or cols.get("average mz") or
                  cols.get("reference m/z") or cols.get("average m/z"))
        name_col = (cols.get("metabolite name") or cols.get("adducted_name") or id_col)
        if not id_col:
            id_col = "__temp_id__"
            tmp_mz = pd.to_numeric(df[mz_col], errors="coerce").round(4).astype("string") if mz_col else "NaN"
            tmp_nm = df[name_col].astype("string") if name_col else "NA"
            df[id_col] = tmp_nm + "_" + tmp_mz
        return id_col, mz_col, name_col

    def _clear_tree(self):
        for iid in self.tree.get_children():
            self.tree.delete(iid)

    def _populate_tree(self, data: pd.DataFrame, limit: int = 5000):
        self._clear_tree()
        if data is None or data.empty or self.colmap is None:
            return
        id_col, mz_col, name_col = self.colmap
        ids = data[id_col].astype("string")
        names = data[name_col].astype("string") if name_col else pd.Series([""]*len(data))
        mz_vals = pd.to_numeric(data[mz_col], errors="coerce").round(4).astype("string") if mz_col else pd.Series([""]*len(data))
        n = min(limit, len(data))
        for i in range(n):
            self.tree.insert("", "end", values=(ids.iat[i], mz_vals.iat[i], names.iat[i]))

    def _apply_advanced_filters(self):
        if self.df is None or self.colmap is None:
            return
        df = self.df
        id_col, mz_col, name_col = self.colmap

        query = (self.search_var.get() or "").strip()
        mask = pd.Series([True]*len(df))
        if query:
            mask &= (
                df[id_col].astype(str).str.contains(query, case=False, na=False) |
                df[name_col].astype(str).str.contains(query, case=False, na=False) |
                (df[mz_col].astype(str).str.contains(query, na=False) if mz_col else False)
            )

        mzf, mzt = self.mz_from.get().strip(), self.mz_to.get().strip()
        if mz_col and (mzf or mzt):
            try:
                mz_vals = pd.to_numeric(df[mz_col], errors="coerce")
                if mzf: mask &= (mz_vals >= float(mzf))
                if mzt: mask &= (mz_vals <= float(mzt))
            except Exception:
                pass

        nfilter = (self.name_filter.get() or "").strip()
        if nfilter and name_col:
            mask &= df[name_col].astype(str).str.contains(nfilter, case=False, na=False)

        self.view_df = df[mask].copy()
        self._populate_tree(self.view_df)

    def _clear_advanced_filters(self):
        self.search_var.set("")
        self.mz_from.set("")
        self.mz_to.set("")
        self.name_filter.set("")
        self.view_df = self.df
        self._populate_tree(self.view_df)

    def _redraw_current_selection(self):
        if self.tree.selection():
            self._on_select_id(None)

    # --- Plot logic (EIC preview) --- EDITED OUT FOR NOW DUE TO LACK OF FUNCTIONABILITY
#    def _on_select_id(self, _evt):
#        sel = self.tree.selection()
#        if not sel:
#            self._plot_message("No selection")
#            return
#
#        values = self.tree.item(sel[0], "values")
#        if not values:
#            self._plot_message("No data")
#            return
#
#        align_id = str(values[0])
#
#        try:
#            # summed preview from the core (corrected if available, otherwise smoothed cache)
#            preview = isobar_core.get_eic_preview(align_id)
#            if not preview:
#                self._plot_message("No EIC yet.\nRun Automated Alignment to generate previews.")
#                return
#
#            x_min, y = preview
#            self.ax.clear()
#            self.ax.plot(x_min, y)                  # summed preview
#            self.ax.set_title(align_id)
#            self.ax.set_xlabel("Retention time (min)")
#            self.ax.set_ylabel("Intensity (%)")
#           self.ax.grid(True, alpha=0.3)
#            self.canvas.draw_idle()
#
#        except Exception as e:
#            self._plot_message(f"Preview error:\n{e}")

    # --- Export selected rows to CSV or Excel ---
    def export_selected_rows(self):
        if self.df is None or self.colmap is None:
            messagebox.showinfo("Export", "No data loaded.")
            return
        items = self.tree.selection()
        if not items:
            messagebox.showinfo("Export", "No rows selected in Explorer.")
            return
        ids = [self.tree.item(iid, "values")[0] for iid in items]
        id_col = self.colmap[0]
        to_export = self.df[self.df[id_col].astype(str).isin(ids)]
        if to_export.empty:
            messagebox.showinfo("Export", "No matching rows to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")]
        )
        if not path:
            return
        try:
            if path.endswith(".xlsx"):
                to_export.to_excel(path, index=False)
            else:
                to_export.to_csv(path, index=False)
            messagebox.showinfo("Exported", f"Exported {len(to_export)} rows to {path}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    # ---------- CSV load / Align ----------
    def on_load_csv(self):
        path = filedialog.askopenfilename(title="Select ID-file", filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        isobar_core.clear_preview_cache()
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as e:
            messagebox.showerror("Load failed", f"Could not read CSV:\n{e}")
            return
        try:
            id_col, mz_col, name_col = self._resolve_columns(df)
            if mz_col:
                df[mz_col] = pd.to_numeric(df[mz_col], errors="coerce")
            self.df = df
            self.view_df = df
            self.colmap = (id_col, mz_col, name_col)
        except Exception as e:
            messagebox.showerror("Column mapping failed", str(e))
            return
        self._set_dataset_labels(path)
        self.nb.select(self.explore)
        self._populate_tree(self.view_df)
        self.status.config(text=f"Loaded: {path}")

    def _set_busy(self, busy: bool):
        widgets = [self.btn_load, self.btn_align, self.btn_export_project, self.btn_load_project, self.btn_settings, self.theme_box]
        for w in widgets:
            try:
                w.configure(state=("disabled" if busy else "normal"))
            except Exception:
                pass
        self.configure(cursor=("watch" if busy else ""))

    def on_align(self):
        if self.df is None or self.df.empty:
            messagebox.showinfo("No data", "Load an ID CSV first.")
            return

        self._finalize_live_line()
        self.log.delete("1.0", "end")
        self.pbar["value"] = 0
        self.status.config(text="Starting alignment…")
        self.nb.select(self.align)

        def run(progress):
            return isobar_core.automated_alignment(progress, self.df)

        def done(summary):
            self._finalize_live_line()
            self.log.insert("end", "Alignment complete.\n")
            self.log.see("end")
            self.status.config(text="Alignment complete")
            self.pbar["value"] = 100

        def error(e):
            self._finalize_live_line()
            messagebox.showerror("Error", str(e))
            self.status.config(text="Error")

        self.runner.run(run, on_done=done, on_error=error)

    # ---------- Toolbar actions ----------
    def on_export_project(self):
        try:
            isobar_handler.save_final_df()
            self.status.config(text="Project exported")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))
            self.status.config(text="Export failed")

    def on_settings(self):
        try:
            isobar_handler.settings_gui()
        except Exception as e:
            messagebox.showerror("Settings error", str(e))

    # ---------- Progress pump handler ----------
    def _on_progress(self, _ev):
        payload = getattr(self, "_progress_payload", ("", None))

        # Console line from StreamRedirector: always a real line (not live)
        if isinstance(payload, tuple) and payload and payload[0] == "console":
            _, line = payload
            self._finalize_live_line()
            if line:
                self.log.insert("end", line + "\n")
                self.log.see("end")
            return

        # Regular progress tuple (msg, pct) → live line
        try:
            msg, pct = payload
        except Exception:
            return

        if msg:
            self._set_live_line(msg)
            self.status.config(text=msg)

        if pct is not None:
            self.pbar["value"] = max(0, min(100, pct))
            if pct >= 100:
                self._finalize_live_line()


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    App().mainloop()
