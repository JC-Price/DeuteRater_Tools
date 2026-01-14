"""
Copyright (c) 2025 Bradley Naylor, Christian Andersen, Michael Porter, Kyle Cutler, Chad Quilling, Benjamin Driggs,
    Coleman Nielsen, Martin Sorensen, J.C. Price, and Brigham Young University
Credit: ChatGPT - helped with building core from the ground up, debugging, and general flow
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
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
# isobar_core.py
# Bridge between the GUI and heavy pipeline in isobar_handler.py
# Exposes:
#   - clear_preview_cache()
#   - get_eic_overlays(align_id, limit_files=None)
#   - automated_alignment(progress_cb, df)

from __future__ import annotations
import time
from typing import Optional, Tuple, List
import numpy as np
import types

# ------------------------------------------------------------------
# Import your pipeline module explicitly
# ------------------------------------------------------------------
try:
    import isobar_handler as PIPE
except Exception as e:
    # Minimal stub so GUI can start; previews will just say "No EIC"
    class _Dummy:
        isobar_frames = {}
    PIPE = _Dummy()  # type: ignore[attr-defined]
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

# ------------------------------------------------------------------
# Caches
# ------------------------------------------------------------------
_preview_cache: dict[str, Tuple[np.ndarray, np.ndarray]] = {}
_overlay_cache: dict[str, Tuple[np.ndarray, np.ndarray, List[Tuple[str, np.ndarray]]]] = {}

def clear_preview_cache():
    """Clear cached EIC previews/overlays. Call on new CSV or before alignment runs."""
    _preview_cache.clear()
    _overlay_cache.clear()

# --- tqdm → GUI relay wrapper ---
def _wrap_tqdm_for_gui(PIPE, progress_cb):
    """
    Replace PIPE.tqdm (the function imported in isobar_handler) with a wrapper
    that yields the same items but sends clean progress lines to the GUI.
    Returns the original so we can restore it.
    """
    original = getattr(PIPE, "tqdm", None)

    def tqdm_gui(iterable=None, *args, **kwargs):
        desc        = kwargs.get("desc", "")
        total       = kwargs.get("total", None)
        unit        = kwargs.get("unit", "")
        mininterval = float(kwargs.get("mininterval", 0.2))

        it = iterable
        if it is None:
            it = range(total or 0)

        if total is None and hasattr(it, "__len__"):
            try:
                total = len(it)
            except Exception:
                total = None

        count = 0
        last  = 0.0
        for item in it:
            yield item
            count += 1
            now = time.time()
            if now - last >= mininterval:
                pct = int(count/total*100) if total else None
                msg = f"{desc}: {count}/{total or '?'} {unit}".strip()
                try:
                    progress_cb(msg, pct)
                except Exception:
                    pass
                last = now
        # final tick
        pct = 100 if total else None
        msg = f"{desc}: {count}/{total or count} {unit} - done".strip()
        try:
            progress_cb(msg, pct)
        except Exception:
            pass

    # install wrapper
    setattr(PIPE, "tqdm", tqdm_gui)
    return original

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _decode_key_safe(k: str) -> str:
    """Reverse npz key mangling if the pipeline provides a decoder."""
    dec = getattr(PIPE, "_decode_key", None)
    if callable(dec):
        try:
            return dec(k)
        except Exception:
            return k
    return k

def _frame_for_alignment_id(align_id: str):
    """
    Find the (key, frame) for an Alignment ID.
    Tries dict key, frame.ID, or membership in frame.sub_df['Alignment ID'].
    """
    isobar_frames = getattr(PIPE, "isobar_frames", {})
    if not isobar_frames:
        return None, None

    # direct key
    if align_id in isobar_frames:
        return align_id, isobar_frames[align_id]

    # scan frames
    for key, fr in isobar_frames.items():
        try:
            if getattr(fr, "ID", None) == align_id:
                return key, fr
            sdf = getattr(fr, "sub_df", None)
            if sdf is not None and "Alignment ID" in sdf.columns:
                if any(sdf["Alignment ID"].astype("string") == align_id):
                    return key, fr
        except Exception:
            continue
    return None, None

def _robust_sum_datasets(traces, num_points: int = 2000):
    """
    Sum multiple (x,y) datasets on a shared grid with linear interpolation.
    Returns (common_x, y_sum, interpolated_ys) or (None, None, []).
    Prefer pipeline's sum_datasets if present.
    """
    sum_ds = getattr(PIPE, "sum_datasets", None)
    if callable(sum_ds):
        try:
            return sum_ds(traces, num_points=num_points)
        except Exception:
            pass

    # Local fallback
    if not traces:
        return None, None, []
    cleaned = []
    for x, y in traces:
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if x.size < 2:
            continue
        order = np.argsort(x, kind="mergesort")
        x, y = x[order], y[order]
        xu, idx = np.unique(x, return_index=True)
        yu = y[idx]
        if xu.size < 2:
            continue
        cleaned.append((xu, yu))
    if not cleaned:
        return None, None, []
    try:
        min_x = max(x.min() for x, _ in cleaned)
        max_x = min(x.max() for x, _ in cleaned)
    except Exception:
        return None, None, []
    if not np.isfinite(min_x) or not np.isfinite(max_x) or max_x <= min_x:
        return None, None, []
    npts = int(max(2, num_points))
    common_x = np.linspace(min_x, max_x, npts, dtype=float)
    interpolated_ys = []
    for x, y in cleaned:
        if x.size < 2:
            continue
        ys = np.interp(common_x, x, y, left=np.nan, right=np.nan)
        interpolated_ys.append(ys)
    if not interpolated_ys:
        return None, None, []
    y_stack = np.vstack(interpolated_ys)
    y_sum = np.nan_to_num(y_stack).sum(axis=0)
    return common_x, y_sum, interpolated_ys

def _sum_norm_traces(traces) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Sum normalized (0–100) per-file traces and renormalize the sum to 0–100.
    Returns (x_sec, y_pct) in seconds for x.
    """
    common_x, y_sum, _ = _robust_sum_datasets(traces, num_points=2000)
    if common_x is None or y_sum is None:
        return None
    ymax = float(np.max(y_sum)) if y_sum.size else 0.0
    if not np.isfinite(ymax) or ymax <= 0:
        return None
    y_pct = (y_sum / ymax) * 100.0
    return common_x, y_pct

def _sum_norm_traces_with_components(traces, labels, num_points=2000):
    """
    Sum multiple (x,y) traces on a shared grid, returning:
      (common_x, y_sum_pct, [(label, y_comp_pct), ...])
    All components are normalized to the summed max for apples-to-apples comparison.
    """
    common_x, y_sum, interpolated_ys = _robust_sum_datasets(traces, num_points=num_points)
    if common_x is None or y_sum is None or not interpolated_ys:
        return None
    ymax = float(np.nanmax(y_sum)) if np.size(y_sum) else 0.0
    if not np.isfinite(ymax) or ymax <= 0:
        return None
    y_sum_pct = (y_sum / ymax) * 100.0
    comps = []
    for i, ys in enumerate(interpolated_ys):
        lab = labels[i] if i < len(labels) else f"File{i+1}"
        comps.append((lab, (np.nan_to_num(ys) / ymax) * 100.0))
    return common_x, y_sum_pct, comps

# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def get_eic_preview(align_id: str):
    """
    Return (x_minutes, y_percent) for a quick EIC preview of the selected Alignment ID.
    Tries, in order:
      1) corrected, normalized smoothed traces (if polynomials exist),
      2) normalized smoothed traces (pre-polynomial),
      3) None (no EICs yet).
    Cached for snappy UX.
    """
    key_str = str(align_id)
    if key_str in _preview_cache:
        return _preview_cache[key_str]

    key, fr = _frame_for_alignment_id(key_str)
    if fr is None:
        return None

    # Case 1: corrected normalized traces after polynomials
    try:
        collect = getattr(PIPE, "collect_datasets", None)
        if getattr(fr, "Pol1dict", None) and callable(collect):
            datasets, _ = collect(key)  # normalized & corrected
            if datasets:
                summed = _sum_norm_traces(datasets)
                if summed:
                    x_sec, y_pct = summed
                    result = (np.asarray(x_sec) / 60.0, np.asarray(y_pct))
                    _preview_cache[key_str] = result
                    return result
    except Exception:
        pass

    # Case 2: pre-polynomial normalized smoothed traces from cache npz
    try:
        spath = getattr(fr, "smooth_path", None)
        if spath:
            traces = []
            with np.load(spath, allow_pickle=False) as npz:
                for k in npz.files:
                    arr = npz[k]
                    if arr.shape[0] >= 2:
                        x = arr[0].astype(float, copy=False)
                        y = arr[1].astype(float, copy=False)  # already 0–100
                        traces.append((x, y))
            if traces:
                summed = _sum_norm_traces(traces)
                if summed:
                    x_sec, y_pct = summed
                    result = (np.asarray(x_sec) / 60.0, np.asarray(y_pct))
                    _preview_cache[key_str] = result
                    return result
    except Exception:
        pass

    return None

def get_eic_overlays(align_id: str, limit_files: int | None = None):
    """
    Return per-file overlays aligned on a common grid:
        (x_minutes, y_sum_pct, [(label, y_pct_on_same_grid), ...])
    Prefers corrected (post-polynomial) traces; falls back to pre-polynomial smoothed traces.
    Cached per Alignment ID.
    """
    key_str = str(align_id)
    if key_str in _overlay_cache:
        x, y, comps = _overlay_cache[key_str]
        if isinstance(limit_files, int) and limit_files > 0 and len(comps) > limit_files:
            return x, y, comps[:limit_files]
        return _overlay_cache[key_str]

    key, fr = _frame_for_alignment_id(key_str)
    if fr is None:
        return None

    # Case 1: corrected normalized smoothed traces (post-polynomial)
    try:
        if getattr(fr, "Pol1dict", None):
            # lazy-load normalized smoothed traces
            get_smooth = getattr(PIPE, "_get_smooth", None)
            if callable(get_smooth):
                sdict = get_smooth(fr)  # {file.mzML: (x, y% 0–100)}
            else:
                sdict = None

            if sdict:
                traces, labels = [], []
                # RT warp to the reference using per-file Pol1dict
                for fname, (x, y) in sdict.items():
                    if fname not in fr.Pol1dict:
                        continue
                    coeffs, rmsd, lo, hi = fr.Pol1dict[fname]
                    rescale = getattr(PIPE, "rescale_to_legendre_space", None)
                    apply_poly = getattr(PIPE, "apply_legendre_polynomial", None)
                    correct_rt = getattr(PIPE, "correct_retention_time", None)
                    if not (callable(rescale) and callable(apply_poly) and callable(correct_rt)):
                        continue
                    xs = rescale(x, lo, hi)
                    corr = apply_poly(xs, coeffs)
                    x_corr = correct_rt(np.clip(x, lo, hi), corr, lo, hi)
                    traces.append((x_corr, y))
                    labels.append(fname.replace(".mzML", ""))

                if traces:
                    summed = _sum_norm_traces_with_components(traces, labels, num_points=2000)
                    if summed:
                        common_x, y_sum_pct, comps = summed
                        result = (np.asarray(common_x) / 60.0, np.asarray(y_sum_pct), comps)
                        _overlay_cache[key_str] = result
                        if isinstance(limit_files, int) and limit_files > 0 and len(comps) > limit_files:
                            return result[0], result[1], result[2][:limit_files]
                        return result
    except Exception:
        pass

    # Case 2: pre-polynomial normalized smoothed traces (npz)
    try:
        spath = getattr(fr, "smooth_path", None)
        if spath:
            traces, labels = [], []
            with np.load(spath, allow_pickle=False) as npz:
                for k in npz.files:
                    arr = npz[k]
                    if arr.shape[0] >= 2:
                        x = arr[0].astype(float, copy=False)
                        y = arr[1].astype(float, copy=False)  # already 0–100
                        fname = _decode_key_safe(k)
                        traces.append((x, y))
                        labels.append(fname.replace(".mzML", ""))

            if traces:
                summed = _sum_norm_traces_with_components(traces, labels, num_points=2000)
                if summed:
                    common_x, y_sum_pct, comps = summed
                    result = (np.asarray(common_x) / 60.0, np.asarray(y_sum_pct), comps)
                    _overlay_cache[key_str] = result
                    if isinstance(limit_files, int) and limit_files > 0 and len(comps) > limit_files:
                        return result[0], result[1], result[2][:limit_files]
                    return result
    except Exception:
        pass

    return None

def automated_alignment(progress_cb, df) -> dict:
    """
    Seed the heavy pipeline with the GUI's DataFrame and run alignment.
    Also hooks tqdm so progress shows nicely in the Align tab.
    """
    settings     = getattr(PIPE, "settings", None)
    make_frames  = getattr(PIPE, "create_isobar_frames", None)
    real_align   = getattr(PIPE, "automated_alignment", None)

    if not (settings and callable(make_frames) and callable(real_align)):
        # keep the GUI responsive even on failure
        for i in range(0, 101, 5):
            time.sleep(0.01)
            progress_cb("Preparing…", i)
        return {"status": "error", "detail": "pipeline entry points not found"}

    # seed globals the pipeline expects
    setattr(PIPE, "original_df", df.copy())
    setattr(PIPE, "isobar_frames", {})

    ppm = 20
    try:
        ppm = settings.extraction.get("ppm_value", 20)
    except Exception:
        pass

    progress_cb(f"Building isobar groups (±{ppm} ppm)…", 5)
    make_frames(df, ppm)

    # hook tqdm → GUI
    original_tqdm = _wrap_tqdm_for_gui(PIPE, progress_cb)

    try:
        progress_cb("Running automated alignment…", 20)
        res = real_align()
        progress_cb("Alignment finished", 100)
        return {"status": "ok", "source": "pipeline", "result": str(res)}
    except Exception as e:
        progress_cb(f"Alignment error: {e}", None)
        raise
    finally:
        # restore tqdm
        if original_tqdm is not None:
            setattr(PIPE, "tqdm", original_tqdm)
