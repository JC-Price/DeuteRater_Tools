# gui_full_tabs_with_yaml_config.py
# GUI scaffold for Homeostasis, Volcano, Conformity, and Settings tabs.
# - Each plot tab:
#   (1) Enable checkbox controlling the whole tab.
#   (2) "Analyze Which Groups?" (All, Low-order, High-order, Other) with an
#       "Other" JSON-like text area that enables only when the tab and "Other" are enabled.
#   (3) Filters panel with five subsections: Rate, Asymptote, Abundance, nL (nₗ), Other.
#       • Homeostasis & Volcano: each subsection has a single "Filters" editor.
#       • Conformity: each subsection has "Primary" and "Secondary" editors.
#       • In all tabs, "Other" shows a "Custom analysis column" entry BEFORE the filters.
# - Settings tab:
#       • Auto-loads post_hoc_default.yaml if present next to this script
#       • Import another YAML
#       • Save the current configuration to YAML
#
# YAML tree (example):
# Homeostasis:
#   enabled: true
#   groups:
#     All: false
#     Low-order: true
#     High-order: false
#     Other:
#       enabled: true
#       text: "<json-like text>"
#   filters:
#     Rate: { enabled: true, text: "..." }
#     Other: { enabled: true, custom_column: "custom_metric_column", text: "..." }
# Conformity:
#   enabled: true
#   groups: ...
#   filters:
#     Rate: { enabled: true, primary: "...", secondary: "..." }
#     Other: { enabled: true, custom_column: "x", primary: "...", secondary: "..." }

from __future__ import annotations
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

# Optional theming
try:
    import ttkbootstrap as tb
    from ttkbootstrap.constants import BOTH, YES
    USE_TTKBOOTSTRAP = True
except Exception:
    USE_TTKBOOTSTRAP = False
    BOTH, YES = tk.BOTH, True

# Optional YAML (required for load/save). We degrade gracefully if missing.
try:
    import yaml
    YAML_AVAILABLE = True
except Exception:
    yaml = None
    YAML_AVAILABLE = False
    
    
# --- small public API so main can register a callback and close the window ---
_ON_ANALYZE = None   # type: Optional[callable]
_ROOT = None         # will hold the Tk root so main can close it if needed

def set_on_analyze(cb):
    """Register a callable cb(config: dict) that will be called when Analyze is pressed."""
    global _ON_ANALYZE
    _ON_ANALYZE = cb

def close_window():
    """Close the GUI programmatically (safe to call from main after analyze)."""
    global _ROOT
    try:
        if _ROOT is not None:
            _ROOT.destroy()
    except Exception:
        pass


# ----------------------- Utility: enable/disable with greying -----------------------
def _set_enabled(widget: tk.Widget, enabled: bool, disabled_label_style: str, enabled_label_style: str):
    """Recursively toggle child widgets' interactive state and label color."""
    for child in widget.winfo_children():
        try:
            if isinstance(child, (ttk.Entry, ttk.Button, ttk.Combobox,
                                  ttk.Checkbutton, ttk.Radiobutton, ttk.Spinbox, ttk.Scale)):
                child.state(["!disabled"] if enabled else ["disabled"])
            elif isinstance(child, ttk.Treeview):
                child.state(["!disabled"] if enabled else ["disabled"])
            elif isinstance(child, ttk.Label):
                child.configure(style=enabled_label_style if enabled else disabled_label_style)
        except tk.TclError:
            pass
        if isinstance(child, (ttk.Frame, ttk.LabelFrame, ttk.Panedwindow)):
            _set_enabled(child, enabled, disabled_label_style, enabled_label_style)


# ----------------------- "Analyze Which Groups?" box & instructions -----------------------
def _build_other_instructions(parent: ttk.Frame) -> ttk.Frame:
    wrap = ttk.Frame(parent)
    ttk.Label(wrap, text="Other: Filter Instructions", font=("Segoe UI", 11, "bold"))\
        .grid(row=0, column=0, sticky="w", pady=(8, 2))
    txt = (
        "Build a filter dictionary here (check the sample DataFrame for exact column names).\n\n"
        "• Bio-condition wildcard:\n"
        "  Use '*' where the condition goes, e.g. 'Abundance rate_*'. Expand to both …_Experiment and …_Control before plotting.\n\n"
        "• Group-specific (rare):\n"
        "  Replace '*' with 'Control' or 'Experiment' (case-sensitive).\n\n"
        "• Strings:\n"
        "  Use 'in:' and 'not in:' with comma-separated values, e.g. {'MS Dial Flag': 'not in:low score,no MS2'}\n\n"
        "• Scope of 'Other' filters:\n"
        "  The dictionary typed under 'Other' is applied to ALL rows indiscriminately."
    )
    ttk.Label(wrap, text=txt, justify="left", anchor="w", wraplength=820).grid(row=1, column=0, sticky="w")
    wrap.grid_columnconfigure(0, weight=1)
    return wrap


def _build_group_box(parent: ttk.Frame):
    """Box 1: Analyze Which Groups? Returns (frame, var_all, var_low, var_high, var_other, other_text)."""
    box = ttk.LabelFrame(parent, text="Analyze Which Groups?", padding=10)

    var_all   = tk.BooleanVar(value=False)
    var_low   = tk.BooleanVar(value=False)
    var_high  = tk.BooleanVar(value=False)
    var_other = tk.BooleanVar(value=False)

    r = 0
    ttk.Checkbutton(box, text="All (everything together)", variable=var_all).grid(row=r, column=0, sticky="w"); r += 1
    ttk.Checkbutton(box, text="Low-order Ontologies (TAG, PC, PE, CL, etc.)",  variable=var_low)\
        .grid(row=r, column=0, sticky="w", pady=(2,0)); r += 1
    ttk.Checkbutton(box, text="High-order Ontologies (Lysos, Ethers, Ceramides, Glycerolipids, etc.)", variable=var_high)\
        .grid(row=r, column=0, sticky="w", pady=(2,0)); r += 1
    ttk.Checkbutton(box, text="Other", variable=var_other)\
        .grid(row=r, column=0, sticky="w", pady=(2,2)); r += 1

    other_panel = _build_other_instructions(box)
    other_panel.grid(row=r, column=0, sticky="nsew", pady=(2, 4)); r += 1

    ttk.Label(box, text="Custom filter dictionary (JSON-like):").grid(row=r, column=0, sticky="w"); r += 1

    tf = ttk.Frame(box); tf.grid(row=r, column=0, sticky="nsew"); r += 1
    other_text = tk.Text(tf, height=8, width=90, wrap="word", font=("Consolas", 10))
    other_text.insert(
        "1.0",
        "# Example (applies to ALL rows):\n"
        "{\n"
        "  'Abundance rate_*': {'min': 0.001, 'max': 5},\n"
        "  'MS Dial Flag': 'not in:low score,no MS2'\n"
        "}\n"
    )
    yscroll = ttk.Scrollbar(tf, orient="vertical", command=other_text.yview)
    other_text.configure(yscrollcommand=yscroll.set)
    other_text.grid(row=0, column=0, sticky="nsew", padx=(0,4), pady=(4,0))
    yscroll.grid(row=0, column=1, sticky="ns", pady=(4,0))

    box.grid_columnconfigure(0, weight=1)
    tf.grid_columnconfigure(0, weight=1); tf.grid_rowconfigure(0, weight=1)
    return box, var_all, var_low, var_high, var_other, other_text


# ----------------------- Helpers for Filters panel -----------------------
def _help_text_for_tab(plot_type: str) -> str:
    pt = plot_type.lower()
    if pt == "volcano":
        return (
            "This tab uses the dataset after each subsection's Filters. Typical axes: log2_* vs -log10* "
            "with thresholds (FC_cut_off, stats_cut_off). Only rows passing the active subsection filters appear."
        )
    if pt == "homeostasis":
        return (
            "This tab uses the dataset after each subsection's Filters. The background frame (0 lines and y = −x) "
            "indicates balance; plotted points are those surviving the active subsection filters."
        )
    # conformity
    return (
        "Conformity: For each enabled subsection below, Primary filters define the grey baseline set, "
        "and Secondary filters refine to the colored set. Paired t-tests are computed for both sets."
    )


def _make_editor(parent: ttk.Widget, preset: str = None) -> tk.Text:
    text = tk.Text(parent, height=8, width=90, wrap="word", font=("Consolas", 10))
    text.insert("1.0", preset or "{\n}\n")
    yscroll = ttk.Scrollbar(parent, orient="vertical", command=text.yview)
    text.configure(yscrollcommand=yscroll.set)
    text.grid(row=0, column=0, sticky="nsew", padx=(0,4))
    yscroll.grid(row=0, column=1, sticky="ns")
    parent.grid_columnconfigure(0, weight=1)
    parent.grid_rowconfigure(0, weight=1)
    return text


def _add_custom_column_input(parent: ttk.Frame, hint: str = "e.g., custom_metric_column"):
    """Adds a labeled Entry for the 'Other' subsection."""
    row = ttk.Frame(parent)
    row.grid_columnconfigure(1, weight=1)
    ttk.Label(row, text="Custom analysis column:", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", padx=(0,8))
    entry = ttk.Entry(row, width=40)
    entry.insert(0, hint)
    entry.grid(row=0, column=1, sticky="ew")
    return row, entry


# ----------------------- Subsections -----------------------
def _build_subsection_simple(parent: ttk.Frame, title: str, help_text: str,
                             default_preset: str = None, add_custom_field: bool = False):
    """
    Used by Homeostasis/Volcano: Enable toggle + single Filters editor.
    If add_custom_field=True and title == 'Other', the custom column entry
    is placed BEFORE the filters editor.
    """
    outer = ttk.LabelFrame(parent, text=title, padding=8)
    outer.grid_columnconfigure(0, weight=1)

    header = ttk.Frame(outer); header.grid(row=0, column=0, sticky="ew")
    ttk.Label(header, text=f"{title} Filters", font=("Segoe UI", 10, "bold")).pack(side="left")
    var_enabled = tk.BooleanVar(value=False)
    ttk.Checkbutton(header, text="Enable section", variable=var_enabled).pack(side="left", padx=(10,0))

    body = ttk.Frame(outer); body.grid(row=1, column=0, sticky="nsew", pady=(6,0))
    ttk.Label(body, text=help_text, justify="left", wraplength=1100)\
        .grid(row=0, column=0, columnspan=2, sticky="w", pady=(0,6))

    row_idx = 1
    custom_entry = None
    if add_custom_field and title.lower().startswith("other"):
        custom_row, custom_entry = _add_custom_column_input(body)
        custom_row.grid(row=row_idx, column=0, columnspan=2, sticky="ew", pady=(0,8))
        row_idx += 1

    editor_frame = ttk.Frame(body); editor_frame.grid(row=row_idx, column=0, columnspan=2, sticky="nsew")
    text_widget = _make_editor(editor_frame, preset=default_preset)
    text_widget.configure(state="disabled")

    body.grid_columnconfigure(0, weight=1); body.grid_rowconfigure(row_idx, weight=1)
    return outer, var_enabled, text_widget, custom_entry


def _build_subsection_conformity(parent: ttk.Frame, title: str,
                                 primary_preset: str = None, secondary_preset: str = None,
                                 add_custom_field: bool = False):
    """
    Used by Conformity: Enable toggle + Primary & Secondary editors.
    If add_custom_field=True and title == 'Other', the custom column entry
    is placed BEFORE the filters editors.
    """
    outer = ttk.LabelFrame(parent, text=title, padding=8)
    outer.grid_columnconfigure(0, weight=1)
    outer.grid_columnconfigure(1, weight=1)

    header = ttk.Frame(outer); header.grid(row=0, column=0, columnspan=2, sticky="ew")
    ttk.Label(header, text=f"{title} Filters", font=("Segoe UI", 10, "bold")).pack(side="left")
    var_enabled = tk.BooleanVar(value=False)
    ttk.Checkbutton(header, text="Enable section", variable=var_enabled).pack(side="left", padx=(10,0))

    row_idx = 1
    custom_entry = None
    if add_custom_field and title.lower().startswith("other"):
        # Place custom column BEFORE editors (attach directly to outer)
        custom_row, custom_entry = _add_custom_column_input(outer)
        custom_row.grid(row=row_idx, column=0, columnspan=2, sticky="ew", pady=(4,6))
        row_idx += 1

    # Editors row
    left = ttk.LabelFrame(outer, text="Primary filters", padding=6)
    right = ttk.LabelFrame(outer, text="Secondary filters", padding=6)
    left.grid(row=row_idx, column=0, sticky="nsew", padx=(0,6))
    right.grid(row=row_idx, column=1, sticky="nsew", padx=(6,0))
    txt_primary = _make_editor(left, preset=primary_preset or "{\n}\n")
    txt_secondary = _make_editor(right, preset=secondary_preset or "{\n}\n")
    txt_primary.configure(state="disabled"); txt_secondary.configure(state="disabled")

    outer.grid_rowconfigure(row_idx, weight=1)
    return outer, var_enabled, txt_primary, txt_secondary, custom_entry


# ----------------------- Filters panel with 5 subsections -----------------------
def _build_box2(parent: ttk.Frame, plot_type: str):
    """
    Second panel with five subsections.
    - Homeostasis/Volcano: each subsection has a single Filters editor; 'Other' gets custom column box.
    - Conformity: each subsection has Primary & Secondary editors; 'Other' gets custom column box.
    """
    outer = ttk.LabelFrame(parent, text="Metabolic Metrics", padding=10)
    outer.grid_columnconfigure(0, weight=1)
    outer.grid_columnconfigure(1, weight=1)

    ttk.Label(outer, text=_help_text_for_tab(plot_type), justify="left", wraplength=1100)\
        .grid(row=0, column=0, columnspan=2, sticky="w", pady=(0,8))

    sections = []
    pt = plot_type.lower()

    if pt == "conformity":
        rate, v_rate, t_rate_p, t_rate_s, e_rate = _build_subsection_conformity(
            outer, "Rate",
            primary_preset="# Example:\n{\n  'Abundance rate_Control': {'min': 0.001, 'max': 5},\n  'Abundance rate_Experiment': {'min': 0.001, 'max': 5}\n}\n",
            secondary_preset="# Example:\n{\n  'Abundance 95pct_confidence_K_Control': {'min': 0, 'max': .1},\n  'Abundance 95pct_confidence_K_Experiment': {'min': 0, 'max': .1}\n}\n",
            add_custom_field=False
        )
        asym, v_asym, t_asym_p, t_asym_s, e_asym = _build_subsection_conformity(
            outer, "Asymptote",
            primary_preset="# Example:\n{\n  'Abundance rate_Control': {'min': 0.001, 'max': 5},\n  'Abundance rate_Experiment': {'min': 0.001, 'max': 5},\n  'Abundance R2_Control': {'min': 0.6},\n  'Abundance R2_Experiment': {'min': 0.6}\n}\n",
            secondary_preset="# Example:\n{\n  'Abundance 95pct_confidence_A_Control': {'min': 0, 'max': .25},\n  'Abundance 95pct_confidence_A_Experiment': {'min': 0, 'max': .25}\n}\n",
            add_custom_field=False
        )
        abn, v_abn, t_abn_p, t_abn_s, e_abn = _build_subsection_conformity(
            outer, "Abundance",
            primary_preset="# Example:\n{\n  'Control_abundance_RSD': {'min': 0, 'max': 0.5},\n  'Experiment_abundance_RSD': {'min': 0, 'max': 0.5}\n}\n",
            secondary_preset="# Example:\n{\n  'MS Dial Flag': 'not in:low score,no MS2'\n}\n",
            add_custom_field=False
        )
        nl, v_nl, t_nl_p, t_nl_s, e_nl = _build_subsection_conformity(
            outer, "nL (nₗ)",
            primary_preset="# Example:\n{\n  'num_nv_time_points_Control': {'min': 2, 'max': 200},\n  'num_nv_time_points_Experiment': {'min': 2, 'max': 200}\n}\n",
            secondary_preset="# Example:\n{\n  'relative_n_val_average_margin_Control': {'min': 0, 'max': .25},\n  'relative_n_val_average_margin_Experiment': {'min': 0, 'max': .25}\n}\n",
            add_custom_field=False
        )
        
        flux, v_flux, t_flux_p, t_flux_s, e_flux = _build_subsection_conformity(
            outer, "Flux",
            primary_preset="# Example:\n{\n  'FC_flux_Control': {'min': 0.5, 'max': 2.0},\n  'FC_flux_Experiment': {'min': 0.5, 'max': 2.0}\n}\n",
            secondary_preset="# Example:\n{\n  '-log10flux_p': {'min': 1.3}\n}\n",
            add_custom_field=False
        )
        synth, v_synth, t_synth_p, t_synth_s, e_synth = _build_subsection_conformity(
            outer, "Synthesis Flux",
            primary_preset="# Example:\n{\n  'FC_synth_flux_Control': {'min': 0.5, 'max': 2.0},\n  'FC_synth_flux_Experiment': {'min': 0.5, 'max': 2.0}\n}\n",
            secondary_preset="# Example:\n{\n  '-log10synth_flux_p': {'min': 1.3}\n}\n",
            add_custom_field=False
        )
        diet, v_diet, t_diet_p, t_diet_s, e_diet = _build_subsection_conformity(
            outer, "Dietary Flux",
            primary_preset="# Example:\n{\n  'FC_diet_flux_Control': {'min': 0.5, 'max': 2.0},\n  'FC_diet_flux_Experiment': {'min': 0.5, 'max': 2.0}\n}\n",
            secondary_preset="# Example:\n{\n  '-log10diet_flux_p': {'min': 1.3}\n}\n",
            add_custom_field=False
        )

        
        other, v_other, t_other_p, t_other_s, e_other = _build_subsection_conformity(
            outer, "Other",
            primary_preset="# Example:\n{\n  'Ontology': 'in:PA,PC,PE'\n}\n",
            secondary_preset="# Example:\n{\n  'MS Dial Flag': 'not in:low score,no MS2'\n}\n",
            add_custom_field=True
        )

         # ---- GRID ORDER ----
        rate.grid( row=1, column=0, sticky="nsew", padx=(0,6),  pady=(0,6))
        asym.grid( row=1, column=1, sticky="nsew", padx=(6,0),  pady=(0,6))
        abn.grid(  row=2, column=0, sticky="nsew", padx=(0,6),  pady=(0,6))
        nl.grid(   row=2, column=1, sticky="nsew", padx=(6,0),  pady=(0,6))
        flux.grid( row=3, column=0, sticky="nsew", padx=(0,6),  pady=(0,6))
        synth.grid(row=3, column=1, sticky="nsew", padx=(6,0),  pady=(0,6))
        diet.grid( row=4, column=0, sticky="nsew", padx=(0,6),  pady=(0,6))
        other.grid(row=4, column=1, sticky="nsew", padx=(6,0),  pady=(0,6))
        outer.grid_rowconfigure(1, weight=1)
        outer.grid_rowconfigure(2, weight=1)
        outer.grid_rowconfigure(3, weight=1)

        sections = [
            ("Rate", v_rate,  t_rate_p,  t_rate_s,  e_rate),
            ("Asymptote", v_asym, t_asym_p, t_asym_s, e_asym),
            ("Abundance", v_abn, t_abn_p, t_abn_s, e_abn),
            ("nL", v_nl, t_nl_p, t_nl_s, e_nl),
            ("Flux", v_flux, t_flux_p, t_flux_s, e_flux),
            ("Synthesis Flux", v_synth, t_synth_p, t_synth_s, e_synth),
            ("Dietary Flux", v_diet, t_diet_p, t_diet_s, e_diet),
            ("Other", v_other, t_other_p, t_other_s, e_other),
        ]
        return outer, sections, True

    # Homeostasis / Volcano: simple editor per subsection; add custom for "Other"
    rate,  v_rate,  t_rate,  e_rate  = _build_subsection_simple(
        outer, "Rate", "Filters specific to rate-based metrics (e.g., Abundance rate_*).",
        default_preset="# Example:\n{\n  'Abundance rate_*': {'min': 0.001, 'max': 5}\n}\n",
        add_custom_field=False
    )
    asym,  v_asym,  t_asym,  e_asym  = _build_subsection_simple(
        outer, "Asymptote", "Filters for asymptote metrics (e.g., Abundance asymptote_*).",
        default_preset="# Example:\n{\n  'Abundance asymptote_*': {'min': 0, 'max': 1.5}\n}\n",
        add_custom_field=False
    )
    abn,   v_abn,   t_abn,   e_abn   = _build_subsection_simple(
        outer, "Abundance", "Filters for DR-corrected abundance or related columns.",
        default_preset="# Example:\n{\n  'Experiment_abundance_RSD': {'min': 0, 'max': 0.5}\n}\n",
        add_custom_field=False
    )
    nl,    v_nl,    t_nl,    e_nl    = _build_subsection_simple(
        outer, "nL (nₗ)", "Filters for n-value metrics (e.g., n_value_* and margins).",
        default_preset="# Example:\n{\n  'num_nv_time_points_*': {'min': 2, 'max': 200}\n}\n",
        add_custom_field=False
    )
    
        # --- Flux metrics (Volcano) ---
    flux, v_flux, t_flux, e_flux = _build_subsection_simple(
        outer, "Flux",
        help_text="Total lipid flux (FC_rate × FC_abn). Use to visualize combined metabolic activity changes.",
        default_preset="# Example:\n{\n  'FC_flux': {'min': 0.5, 'max': 2.0},\n  '-log10flux_p': {'min': 1.3}\n}\n",
        add_custom_field=False
    )
    
    synth, v_synth, t_synth, e_synth = _build_subsection_simple(
        outer, "Synthesis Flux",
        help_text="Endogenous synthesis flux (FC_flux × (A_exp / A_ctl)). Reflects biosynthetic lipid turnover.",
        default_preset="# Example:\n{\n  'FC_synth_flux': {'min': 0.5, 'max': 2.0},\n  '-log10synth_flux_p': {'min': 1.3}\n}\n",
        add_custom_field=False
    )
    
    diet, v_diet, t_diet, e_diet = _build_subsection_simple(
        outer, "Dietary Flux",
        help_text="Dietary/precursor-driven flux ((1−A_exp)/(1−A_ctl)). Indicates serum-derived lipid influx.",
        default_preset="# Example:\n{\n  'FC_diet_flux': {'min': 0.5, 'max': 2.0},\n  '-log10diet_flux_p': {'min': 1.3}\n}\n",
        add_custom_field=False
    )

    
    other, v_other, t_other, e_other = _build_subsection_simple(
        outer, "Other", "Miscellaneous or composite filters not covered above.",
        default_preset="# Example:\n{\n  'MS Dial Flag': 'not in:low score,no MS2'\n}\n",
        add_custom_field=True   # custom column BEFORE filters
    )

     # ---- GRID ORDER ----
    rate.grid(  row=1, column=0, sticky="nsew", padx=(0,6),  pady=(0,6))
    asym.grid(  row=1, column=1, sticky="nsew", padx=(6,0),  pady=(0,6))
    abn.grid(   row=2, column=0, sticky="nsew", padx=(0,6),  pady=(0,6))
    nl.grid(    row=2, column=1, sticky="nsew", padx=(6,0),  pady=(0,6))
    flux.grid(  row=3, column=0, sticky="nsew", padx=(0,6),  pady=(0,6))
    synth.grid( row=3, column=1, sticky="nsew", padx=(6,0),  pady=(0,6))
    diet.grid(  row=4, column=0, sticky="nsew", padx=(0,6),  pady=(0,6))
    other.grid( row=4, column=1, sticky="nsew", padx=(6,0),  pady=(0,6))
    outer.grid_rowconfigure(1, weight=1)
    outer.grid_rowconfigure(2, weight=1)
    outer.grid_rowconfigure(3, weight=1)

    sections = [
        ("Rate", v_rate,  t_rate,  None),
        ("Asymptote", v_asym, t_asym, None),
        ("Abundance", v_abn, t_abn, None),
        ("nL", v_nl, t_nl, None),
        ("Flux", v_flux, t_flux, None),
        ("Synthesis Flux", v_synth, t_synth, None),
        ("Dietary Flux", v_diet, t_diet, None),
        ("Other", v_other, t_other, e_other),  # only Other has custom entry
    ]
    return outer, sections, False


# ----------------------- YAML I/O helpers -----------------------
def load_yaml_config(filepath: Path) -> dict:
    if not YAML_AVAILABLE:
        messagebox.showerror("YAML not available", "PyYAML is not installed. Run: pip install pyyaml")
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f)
        return obj or {}
    except Exception as e:
        messagebox.showerror("Load YAML failed", f"Could not load:\n{filepath}\n\n{e}")
        return {}


def save_yaml_config(filepath: Path, config: dict) -> bool:
    if not YAML_AVAILABLE:
        messagebox.showerror("YAML not available", "PyYAML is not installed. Run: pip install pyyaml")
        return False
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
        return True
    except Exception as e:
        messagebox.showerror("Save YAML failed", f"Could not save:\n{filepath}\n\n{e}")
        return False


# ----------------------- Build a tab and register state -----------------------
def _build_tab_with_enable(parent, title: str, plot_type: str,
                           disabled_label_style: str, enabled_label_style: str,
                           state_registry: dict):
    frame = ttk.Frame(parent, padding=10)
    frame.grid_columnconfigure(0, weight=1)

    # Heading + Enable
    heading = ttk.Frame(frame)
    heading.grid(row=0, column=0, sticky="ew", pady=(0, 8))
    ttk.Label(heading, text=title, font=("Segoe UI", 14, "bold")).pack(side="left")
    tab_enabled = tk.BooleanVar(value=False)
    ttk.Checkbutton(heading, text="Enable", variable=tab_enabled).pack(side="left", padx=(10,0))

    # Box 1: Analyze Which Groups?
    box1, var_all, var_low, var_high, var_other_group, other_text = _build_group_box(frame)
    box1.grid(row=1, column=0, sticky="nsew", padx=2, pady=(0,8))

    # ── NEW: Only build Filters panel if NOT homeostasis ───────────────
    box2 = None
    sections = []
    is_conformity = False
    if plot_type.lower() != "homeostasis":
        box2, sections, is_conformity = _build_box2(frame, plot_type)
        box2.grid(row=2, column=0, sticky="nsew", padx=2, pady=(0,0))
        frame.grid_rowconfigure(2, weight=1)

    frame.grid_rowconfigure(1, weight=1)

    # --- State management ---
    def apply_other_text_state(*_):
        want = tab_enabled.get() and var_other_group.get()
        other_text.configure(state="normal" if want else "disabled")

    def apply_section_states(*_):
        on = tab_enabled.get()
        _set_enabled(box1, on, disabled_label_style, enabled_label_style)
        if box2 is not None:
            _set_enabled(box2, on, disabled_label_style, enabled_label_style)

        # Enable editors only when both tab and subsection are enabled
        for sec in sections:
            if is_conformity:
                _, var_sec, txt_p, txt_s, entry_custom = sec
                want = on and var_sec.get()
                txt_p.configure(state=("normal" if want else "disabled"))
                txt_s.configure(state=("normal" if want else "disabled"))
                if entry_custom is not None:
                    entry_custom.state(["!disabled"] if want else ["disabled"])
            else:
                _, var_sec, txt, entry_custom = sec
                want = on and var_sec.get()
                txt.configure(state=("normal" if want else "disabled"))
                if entry_custom is not None:
                    entry_custom.state(["!disabled"] if want else ["disabled"])

        apply_other_text_state()

    tab_enabled.trace_add("write", apply_section_states)
    var_other_group.trace_add("write", apply_other_text_state)
    for sec in sections:
        if is_conformity:
            _, var_sec, *_ = sec
        else:
            _, var_sec, *_ = sec
        var_sec.trace_add("write", apply_section_states)

    # Initial states
    other_text.configure(state="disabled")
    for sec in sections:
        if is_conformity:
            _, var_sec, txt_p, txt_s, entry_custom = sec
            var_sec.set(False)
            txt_p.configure(state="disabled"); txt_s.configure(state="disabled")
            if entry_custom is not None:
                entry_custom.state(["disabled"])
        else:
            _, var_sec, txt, entry_custom = sec
            var_sec.set(False)
            txt.configure(state="disabled")
            if entry_custom is not None:
                entry_custom.state(["disabled"])
    apply_section_states()

    # --- Register in state_registry ---
    state_registry[title] = {
        "enabled_var": tab_enabled,
        "apply_state_fn": apply_section_states,
        "groups": {
            "All": var_all,
            "Low-order": var_low,
            "High-order": var_high,
            "Other": {
                "var": var_other_group,
                "text": other_text
            }
        },
        "filters": sections,          # will be [] for Homeostasis
        "is_conformity": is_conformity
    }
    return frame

# ----------------------- YAML <-> GUI conversion -----------------------
def dump_gui_state(state_registry: dict) -> dict:
    config = {}
    for tab_name, tab in state_registry.items():
        if tab_name.startswith("_"):
            continue
        tab_cfg = {}
        tab_cfg["enabled"] = bool(tab["enabled_var"].get())
        

        # Groups
        g = {}
        for name, val in tab["groups"].items():
            if isinstance(val, dict):  # "Other"
                enabled = bool(val["var"].get())
                textval = val["text"].get("1.0", "end").strip() if enabled else ""
                g[name] = {"enabled": enabled, "text": textval}
            else:
                g[name] = bool(val.get())
        tab_cfg["groups"] = g

        # Filters
        f = {}
        is_conf = tab["is_conformity"]
        for sec in tab["filters"]:
            if is_conf:
                sec_name, var_enabled, txt_p, txt_s, custom_entry = sec
                enabled = bool(var_enabled.get())
                item = {
                    "enabled": enabled,
                    "primary": txt_p.get("1.0", "end").strip() if enabled else "",
                    "secondary": txt_s.get("1.0", "end").strip() if enabled else ""
                }
                if custom_entry is not None:
                    item["custom_column"] = custom_entry.get().strip()
            else:
                sec_name, var_enabled, txt, custom_entry = sec
                enabled = bool(var_enabled.get())
                item = {
                    "enabled": enabled,
                    "text": txt.get("1.0", "end").strip() if enabled else ""
                }
                if custom_entry is not None:
                    item["custom_column"] = custom_entry.get().strip()
            f[sec_name] = item

        tab_cfg["filters"] = f
        config[tab_name] = tab_cfg
        

    # -----------------------
    # Settings (non-plot)
    # -----------------------
        settings_tab = state_registry.get("_settings_tab")
        if settings_tab is not None and hasattr(settings_tab, "_norm_var"):
            config["Settings"] = {
                "perform_abundance_normalization": bool(
                    settings_tab._norm_var.get()
                ),
                "use_standards_normalization": bool(
                    settings_tab._standards_norm_var.get()
                ),
                "standards_baseline_group": (
                    settings_tab._standards_baseline_entry.get().strip()
                    if hasattr(settings_tab, "_standards_baseline_entry")
                    else None
                ),
            }

    
                
        
    return config


def apply_gui_state(config: dict, state_registry: dict):
    # Set tab-level enables first (so traces enable/disable children)
    # -----------------------
    # Settings (non-plot)
    # -----------------------
    settings_cfg = config.get("Settings", {})
    settings_tab = state_registry.get("_settings_tab")
    
    if settings_tab is not None and hasattr(settings_tab, "_norm_var"):
        settings_tab._norm_var.set(
            bool(settings_cfg.get("perform_abundance_normalization", False))
        )


    
    
    for tab_name, tab_cfg in config.items():
        if tab_name not in state_registry:
            continue
        tab = state_registry[tab_name]
        tab["enabled_var"].set(bool(tab_cfg.get("enabled", False)))

    # Populate groups and filters
    for tab_name, tab_cfg in config.items():
        if tab_name not in state_registry:
            continue
        tab = state_registry[tab_name]
        is_conf = tab["is_conformity"]

        # Groups
        groups_cfg = tab_cfg.get("groups", {})
        for gname, gval in groups_cfg.items():
            if gname not in tab["groups"]:
                continue
            target = tab["groups"][gname]
            if isinstance(target, dict):  # Other
                target["var"].set(bool(gval.get("enabled", False)))
                # Make sure text is editable while inserting
                target["text"].configure(state="normal")
                target["text"].delete("1.0", "end")
                target["text"].insert("1.0", gval.get("text", ""))
            else:
                target.set(bool(gval))

        # Filters
        filters_cfg = tab_cfg.get("filters", {})
        for sec in tab["filters"]:
            if is_conf:
                name, var_enabled, txt_p, txt_s, custom_entry = sec
                if name not in filters_cfg:
                    continue
                fval = filters_cfg[name]
                var_enabled.set(bool(fval.get("enabled", False)))

                # Enable texts before inserting (traces should toggle, but be safe)
                txt_p.configure(state="normal")
                txt_s.configure(state="normal")
                txt_p.delete("1.0", "end"); txt_p.insert("1.0", fval.get("primary", ""))
                txt_s.delete("1.0", "end"); txt_s.insert("1.0", fval.get("secondary", ""))
                if custom_entry is not None and "custom_column" in fval:
                    custom_entry.delete(0, "end"); custom_entry.insert(0, fval["custom_column"])
            else:
                name, var_enabled, txt, custom_entry = sec
                if name not in filters_cfg:
                    continue
                fval = filters_cfg[name]
                var_enabled.set(bool(fval.get("enabled", False)))

                txt.configure(state="normal")
                txt.delete("1.0", "end"); txt.insert("1.0", fval.get("text", ""))
                if custom_entry is not None and "custom_column" in fval:
                    custom_entry.delete(0, "end"); custom_entry.insert(0, fval["custom_column"])

        # Final refresh to apply disabled/enabled correctly
        tab["apply_state_fn"]()


# ----------------------- Settings tab -----------------------
def _build_settings_tab(parent, state_registry) -> ttk.Frame:
    frame = ttk.Frame(parent, padding=10)

    ttk.Label(
        frame,
        text="Settings",
        font=("Segoe UI", 14, "bold")
    ).pack(anchor="w", pady=(0, 8))

    # -----------------------
    # Abundance normalization
    # -----------------------
    norm_var = tk.BooleanVar(value=False)

    norm_frame = ttk.LabelFrame(
        frame,
        text="Abundance normalization",
        padding=10
    )
    norm_frame.pack(fill="x", anchor="w", pady=(6, 10))

    ttk.Checkbutton(
        norm_frame,
        text="Apply per-mzML abundance normalization",
        variable=norm_var
    ).pack(anchor="w")

    # -----------------------
    # Standards normalization
    # -----------------------
    standards_var = tk.BooleanVar(value=True)

    standards_frame = ttk.LabelFrame(
        frame,
        text="Standards normalization",
        padding=10
    )
    standards_frame.pack(fill="x", anchor="w", pady=(0, 10))

    ttk.Checkbutton(
        standards_frame,
        text="Normalize experiment vs control using internal standards",
        variable=standards_var
    ).pack(anchor="w")

    # ---- Baseline group entry (ONLY enabled if standards_var is checked) ----
    baseline_row = ttk.Frame(standards_frame)
    baseline_row.pack(fill="x", anchor="w", pady=(6, 0))

    ttk.Label(
        baseline_row,
        text="Baseline group for standards normalization (e.g. A3, YM-4v3-E3):"
    ).pack(side="left", padx=(0, 8))

    baseline_entry = ttk.Entry(baseline_row, width=30)
    baseline_entry.insert(0, "A3")
    baseline_entry.state(["disabled"])
    baseline_entry.pack(side="left", fill="x", expand=True)

    def _apply_standards_state(*_):
        if standards_var.get():
            baseline_entry.state(["!disabled"])
        else:
            baseline_entry.state(["disabled"])

    standards_var.trace_add("write", _apply_standards_state)
    _apply_standards_state()  # initialize state

    # -----------------------
    # Attach state to Settings tab
    # -----------------------
    frame._norm_var = norm_var
    frame._standards_norm_var = standards_var
    frame._standards_baseline_entry = baseline_entry

    # ----------------------------------------------------
    # YAML status + buttons
    # ----------------------------------------------------
    status_var = tk.StringVar(value="No config loaded")

    btn_row = ttk.Frame(frame)
    btn_row.pack(anchor="w", pady=(0, 8))

    btn_import = ttk.Button(btn_row, text="Import YAML", width=20)
    btn_save = ttk.Button(btn_row, text="Save current config as YAML", width=28)
    btn_reload_default = ttk.Button(
        btn_row,
        text="Reload default (post_hoc_default.yaml)",
        width=34
    )

    btn_import.grid(row=0, column=0, padx=(0, 6))
    btn_save.grid(row=0, column=1, padx=(0, 6))
    btn_reload_default.grid(row=0, column=2, padx=(0, 6))

    ttk.Label(frame, textvariable=status_var).pack(anchor="w", pady=(4, 0))

    # Disable YAML buttons if PyYAML missing
    if not YAML_AVAILABLE:
        for b in (btn_import, btn_save, btn_reload_default):
            b.state(["disabled"])
        ttk.Label(
            frame,
            foreground="#a33",
            text="PyYAML not installed — YAML features disabled.  pip install pyyaml"
        ).pack(anchor="w", pady=(10, 0))

    # ----------------------------------------------------
    # Button handlers
    # ----------------------------------------------------
    def import_yaml():
        path = filedialog.askopenfilename(
            title="Import YAML Configuration",
            filetypes=[("YAML files", "*.yaml *.yml")]
        )
        if not path:
            return
        cfg = load_yaml_config(Path(path))
        if cfg:
            apply_gui_state(cfg, state_registry)
            status_var.set(f"Loaded: {path}")

    def save_yaml():
        path = filedialog.asksaveasfilename(
            title="Save YAML Configuration",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml *.yml")]
        )
        if not path:
            return
        cfg = dump_gui_state(state_registry)
        if save_yaml_config(Path(path), cfg):
            status_var.set(f"Saved to: {path}")

    def reload_default():
        default_path = Path(__file__).parent / "post_hoc_default.yaml"
        if not default_path.exists():
            messagebox.showinfo(
                "Default not found",
                f"No default YAML at:\n{default_path}"
            )
            return
        cfg = load_yaml_config(default_path)
        if cfg:
            apply_gui_state(cfg, state_registry)
            status_var.set(f"Loaded default: {default_path}")

    btn_import.configure(command=import_yaml)
    btn_save.configure(command=save_yaml)
    btn_reload_default.configure(command=reload_default)

    # ----------------------------------------------------
    # Auto-load default YAML (if present)
    # ----------------------------------------------------
    default_path = Path(__file__).parent / "post_hoc_default.yaml"
    if YAML_AVAILABLE and default_path.exists():
        try:
            cfg = load_yaml_config(default_path)
            if cfg:
                apply_gui_state(cfg, state_registry)
                status_var.set(f"Loaded default: {default_path}")
        except Exception:
            pass  # non-fatal

    return frame



# ----------------------- App entry -----------------------
def launch_gui():
    global _ROOT, _ON_ANALYZE

    if USE_TTKBOOTSTRAP:
        root = tb.Window(themename="flatly")
        root.title("Lipidomics Plots")
    else:
        root = tk.Tk()
        root.title("Lipidomics Plots")
    root.geometry("1280x860")
    _ROOT = root  # allow external close_window() to access it

    style = ttk.Style(root)
    style.configure("Enabled.TLabel", foreground="")
    style.configure("Disabled.TLabel", foreground="#8a8a8a")

    # Registry for all state handles used by YAML round-trip
    state_registry = {}

    # Top toolbar (Analyze button lives here)
    toolbar = ttk.Frame(root)
    toolbar.pack(fill="x", side="top", padx=6, pady=6)

    def _on_analyze_clicked():
        # Build the full config tree from the UI state
        try:
            cfg = dump_gui_state(state_registry)
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Error", f"Failed to read GUI state: {e}")
            return
    
        if not _ON_ANALYZE:
            from tkinter import messagebox
            messagebox.showinfo(
                "No analyze hook",
                "No analyze callback has been registered."
            )
            return
    
        import threading
        from tkinter import messagebox
    
        def worker():
            try:
                _ON_ANALYZE(cfg)
            except Exception as e:
                err_msg = f"Analyze callback raised:\n{e}"
                if _ROOT is not None:
                    _ROOT.after(
                        0,
                        lambda msg=err_msg: messagebox.showerror(
                            "Analyze failed",
                            msg
                        )
                    )

        # Run analysis in background thread
        threading.Thread(
            target=worker,
            daemon=True
        ).start()


    analyze_btn = ttk.Button(toolbar, text="Analyze", command=_on_analyze_clicked)
    analyze_btn.pack(side="right", padx=4)

    # Notebook (tabs)
    nb = ttk.Notebook(root)
    nb.pack(fill=BOTH, expand=YES)

    # Tabs - pass state_registry into builders so they can register themselves
    homeostasis_tab = _build_tab_with_enable(
        nb, "Homeostasis", "homeostasis", "Disabled.TLabel", "Enabled.TLabel", state_registry
    )
    volcano_tab     = _build_tab_with_enable(
        nb, "Volcano",     "volcano",     "Disabled.TLabel", "Enabled.TLabel", state_registry
    )
    conformity_tab  = _build_tab_with_enable(
        nb, "Conformity",  "conformity",  "Disabled.TLabel", "Enabled.TLabel", state_registry
    )
    settings_tab    = _build_settings_tab(nb, state_registry)
    state_registry["_settings_tab"] = settings_tab

    nb.add(homeostasis_tab, text="Homeostasis")
    nb.add(volcano_tab,     text="Volcano")
    nb.add(conformity_tab,  text="Conformity")
    nb.add(settings_tab,    text="Settings")
    nb.select(0)

    root.mainloop()



if __name__ == "__main__":
    launch_gui()
