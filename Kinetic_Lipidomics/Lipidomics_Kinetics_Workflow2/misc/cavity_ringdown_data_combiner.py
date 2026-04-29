# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:37:50 2026

@author: Brigham Young Univ
"""

"""
 Data Combiner
Combines any number of CSV files into a single concatenated output using tkinter + pandas.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import os


class CavityRingdownCombiner(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Data Combiner")
        self.geometry("750x550")
        self.resizable(True, True)
        self.configure(bg="#1e2330")

        self.selected_files = []
        self.combined_df = None

        self._build_ui()

    # ------------------------------------------------------------------ #
    #  UI construction                                                     #
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        # ── Title bar ──
        title_frame = tk.Frame(self, bg="#131825", pady=10)
        title_frame.pack(fill="x")
        tk.Label(
            title_frame,
            text="⬡  Data Combiner",
            font=("Courier New", 15, "bold"),
            fg="#5ef6d0",
            bg="#131825",
        ).pack()

        # ── File list area ──
        list_frame = tk.LabelFrame(
            self,
            text="  Selected CSV Files  ",
            font=("Courier New", 9),
            fg="#8899bb",
            bg="#1e2330",
            bd=1,
            relief="solid",
            padx=8,
            pady=6,
        )
        list_frame.pack(fill="both", expand=False, padx=16, pady=(12, 4))

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        self.file_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            height=7,
            bg="#131825",
            fg="#c8d8f0",
            selectbackground="#2e4470",
            activestyle="none",
            font=("Courier New", 9),
            bd=0,
            highlightthickness=0,
        )
        self.file_listbox.pack(fill="both", expand=True)
        scrollbar.config(command=self.file_listbox.yview)

        # ── File control buttons ──
        btn_frame = tk.Frame(self, bg="#1e2330")
        btn_frame.pack(fill="x", padx=16, pady=(4, 8))

        self._btn(btn_frame, "＋ Add CSV Files", self._add_files, "#2a7a5e").pack(side="left", padx=(0, 6))
        self._btn(btn_frame, "✕ Remove Selected", self._remove_selected, "#7a2a2a").pack(side="left", padx=(0, 6))
        self._btn(btn_frame, "⟳ Clear All", self._clear_all, "#3a3a5a").pack(side="left")

        # ── Options ──
        opts_frame = tk.LabelFrame(
            self,
            text="  Options  ",
            font=("Courier New", 9),
            fg="#8899bb",
            bg="#1e2330",
            bd=1,
            relief="solid",
            padx=10,
            pady=6,
        )
        opts_frame.pack(fill="x", padx=16, pady=(0, 8))

        self.reset_index_var = tk.BooleanVar(value=True)
        self.drop_duplicates_var = tk.BooleanVar(value=False)
        self.add_source_col_var = tk.BooleanVar(value=True)
        self.dedup_alignment_id_var = tk.BooleanVar(value=False)

        self._check(opts_frame, "Reset index after concat", self.reset_index_var).grid(row=0, column=0, sticky="w", padx=(0, 20))
        self._check(opts_frame, "Drop duplicate rows", self.drop_duplicates_var).grid(row=0, column=1, sticky="w", padx=(0, 20))
        self._check(opts_frame, "Add 'source_file' column", self.add_source_col_var).grid(row=0, column=2, sticky="w")
        self._check(opts_frame, "Drop duplicates by 'Alignment ID' (keep first)", self.dedup_alignment_id_var).grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))

        # ── Combine button ──
        combine_frame = tk.Frame(self, bg="#1e2330")
        combine_frame.pack(pady=(0, 8))
        self._btn(combine_frame, "⚡  Combine Files", self._combine, "#1a6e8a", width=22, font_size=11).pack()

        # ── Status bar ──
        self.status_var = tk.StringVar(value="Ready — add CSV files to begin.")
        status_bar = tk.Label(
            self,
            textvariable=self.status_var,
            font=("Courier New", 8),
            fg="#556688",
            bg="#131825",
            anchor="w",
            padx=10,
        )
        status_bar.pack(fill="x", side="bottom")

        # ── Preview table ──
        preview_frame = tk.LabelFrame(
            self,
            text="  Preview (first 50 rows)  ",
            font=("Courier New", 9),
            fg="#8899bb",
            bg="#1e2330",
            bd=1,
            relief="solid",
            padx=6,
            pady=6,
        )
        preview_frame.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        self.tree_frame = tk.Frame(preview_frame, bg="#1e2330")
        self.tree_frame.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(self.tree_frame, show="headings")
        self._style_treeview()

        tree_xscroll = ttk.Scrollbar(self.tree_frame, orient="horizontal", command=self.tree.xview)
        tree_yscroll = ttk.Scrollbar(self.tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(xscrollcommand=tree_xscroll.set, yscrollcommand=tree_yscroll.set)

        tree_yscroll.pack(side="right", fill="y")
        tree_xscroll.pack(side="bottom", fill="x")
        self.tree.pack(fill="both", expand=True)

        # ── Save button ──
        save_frame = tk.Frame(self, bg="#1e2330")
        save_frame.pack(pady=(0, 12))
        self._btn(save_frame, "💾  Save Combined CSV", self._save, "#4a6e1a", width=22, font_size=11).pack()

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _btn(self, parent, text, command, bg, width=18, font_size=9):
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg="#e8f0ff",
            activebackground=bg,
            activeforeground="#ffffff",
            font=("Courier New", font_size, "bold"),
            relief="flat",
            cursor="hand2",
            width=width,
            padx=4,
            pady=4,
        )

    def _check(self, parent, text, var):
        return tk.Checkbutton(
            parent,
            text=text,
            variable=var,
            bg="#1e2330",
            fg="#c8d8f0",
            selectcolor="#131825",
            activebackground="#1e2330",
            activeforeground="#5ef6d0",
            font=("Courier New", 9),
        )

    def _style_treeview(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "Treeview",
            background="#131825",
            foreground="#c8d8f0",
            fieldbackground="#131825",
            rowheight=22,
            font=("Courier New", 8),
        )
        style.configure(
            "Treeview.Heading",
            background="#1e2330",
            foreground="#5ef6d0",
            font=("Courier New", 8, "bold"),
        )
        style.map("Treeview", background=[("selected", "#2e4470")])

    def _set_status(self, msg, colour="#556688"):
        self.status_var.set(msg)
        self.update_idletasks()

    # ------------------------------------------------------------------ #
    #  Actions                                                             #
    # ------------------------------------------------------------------ #
    def _add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select CSV Files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        added = 0
        for p in paths:
            if p not in self.selected_files:
                self.selected_files.append(p)
                self.file_listbox.insert("end", os.path.basename(p))
                added += 1
        if added:
            self._set_status(f"{len(self.selected_files)} file(s) queued.")

    def _remove_selected(self):
        indices = list(self.file_listbox.curselection())
        for i in reversed(indices):
            self.file_listbox.delete(i)
            del self.selected_files[i]
        self._set_status(f"{len(self.selected_files)} file(s) remaining.")

    def _clear_all(self):
        self.selected_files.clear()
        self.file_listbox.delete(0, "end")
        self.combined_df = None
        self._clear_tree()
        self._set_status("Cleared.")

    def _combine(self):
        if not self.selected_files:
            messagebox.showwarning("No Files", "Please add at least one CSV file first.")
            return

        frames = []
        errors = []
        for path in self.selected_files:
            try:
                df = pd.read_csv(path)
                if self.add_source_col_var.get():
                    df.insert(0, "source_file", os.path.basename(path))
                frames.append(df)
            except Exception as e:
                errors.append(f"{os.path.basename(path)}: {e}")

        if errors:
            messagebox.showerror("Read Errors", "\n".join(errors))
            return

        combined = pd.concat(frames, ignore_index=self.reset_index_var.get())

        status_parts = [f"Combined {len(self.selected_files)} files → {len(combined)} rows"]

        if self.drop_duplicates_var.get():
            before = len(combined)
            combined = combined.drop_duplicates()
            dropped = before - len(combined)
            status_parts.append(f"{dropped} full-row duplicates removed")

        if self.dedup_alignment_id_var.get():
            col = "Alignment ID"
            if col not in combined.columns:
                messagebox.showwarning(
                    "Column Not Found",
                    f"'{col}' column was not found in the combined data.\n"
                    "Skipping deduplication by Alignment ID.",
                )
            else:
                before = len(combined)
                combined = combined.drop_duplicates(subset=[col], keep="first")
                dropped = before - len(combined)
                status_parts.append(f"{dropped} Alignment ID duplicates removed")

        self._set_status(f"{' · '.join(status_parts)} · {len(combined.columns)} columns.")
        self.combined_df = combined
        self._populate_tree(combined.head(50))

    def _save(self):
        if self.combined_df is None:
            messagebox.showwarning("Nothing to Save", "Combine files first before saving.")
            return

        out_path = filedialog.asksaveasfilename(
            title="Save Combined CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="combined_data.csv",
        )
        if not out_path:
            return
        try:
            self.combined_df.to_csv(out_path, index=False)
            self._set_status(f"Saved → {os.path.basename(out_path)}")
            messagebox.showinfo("Saved", f"Combined data saved to:\n{out_path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    # ------------------------------------------------------------------ #
    #  Treeview helpers                                                    #
    # ------------------------------------------------------------------ #
    def _clear_tree(self):
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = []

    def _populate_tree(self, df: pd.DataFrame):
        self._clear_tree()
        cols = list(df.columns)
        self.tree["columns"] = cols
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=max(90, len(col) * 9), anchor="w", stretch=False)
        for _, row in df.iterrows():
            self.tree.insert("", "end", values=list(row))


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    app = CavityRingdownCombiner()
    app.mainloop()