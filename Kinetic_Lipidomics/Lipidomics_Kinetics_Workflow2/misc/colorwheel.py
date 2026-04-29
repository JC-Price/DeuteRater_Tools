#!/usr/bin/env python3
"""
A color wheel chart for making nice figures. 
Pick a position on the wheel and use the 4 colors as your choices for good contrast. 
"""

import math
import random
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Tuple

import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import colorsys



# Color utilities
def clamp(x: int, lo: int = 0, hi: int = 255) -> int:
    return max(lo, min(hi, x))


def parse_color(code: str) -> Tuple[int, int, int]:

    s = (code or "").strip().lower()
    if not s:
        raise ValueError("Empty color string.")

    if s.startswith("rgb"):
        # Format: rgb(r,g,b)
        if "(" not in s or ")" not in s:
            raise ValueError(f"Invalid rgb() format: {code}")
        inside = s[s.find("(") + 1 : s.find(")")]
        parts = inside.replace("%", "").split(",")
        if len(parts) != 3:
            raise ValueError(f"Invalid rgb() format: {code}")
        r, g, b = [int(float(p.strip())) for p in parts]
        return (clamp(r), clamp(g), clamp(b))

    if s.startswith("#"):
        s = s[1:]
    if len(s) == 3:
        r = int(s[0] * 2, 16)
        g = int(s[1] * 2, 16)
        b = int(s[2] * 2, 16)
        return (r, g, b)
    if len(s) == 6:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return (r, g, b)

    raise ValueError(f"Unsupported color code: {code}")


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)



# Model-based hue rotations
def _rotate_hls(rgb: Tuple[int, int, int], delta_h: float) -> Tuple[int, int, int]:
    """Rotate hue by delta_h (0..1 = 0..360°) in HSL (colorsys HLS), keep L,S."""
    r, g, b = [v / 255.0 for v in rgb]
    h, l, s = colorsys.rgb_to_hls(r, g, b)  # HLS order
    h2 = (h + delta_h) % 1.0
    r2, g2, b2 = colorsys.hls_to_rgb(h2, l, s)
    return (clamp(int(round(r2 * 255))), clamp(int(round(g2 * 255))), clamp(int(round(b2 * 255))))


def _rotate_hsv(rgb: Tuple[int, int, int], delta_h: float) -> Tuple[int, int, int]:
    """Rotate hue by delta_h (0..1 = 0..360°) in HSV, keep S,V."""
    r, g, b = [v / 255.0 for v in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h2 = (h + delta_h) % 1.0
    r2, g2, b2 = colorsys.hsv_to_rgb(h2, s, v)
    return (clamp(int(round(r2 * 255))), clamp(int(round(g2 * 255))), clamp(int(round(b2 * 255))))


def complementary_hsl(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return _rotate_hls(rgb, 0.5)  # 180°


def complementary_hsv(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return _rotate_hsv(rgb, 0.5)  # 180°


def rotate_90_hsl(rgb: Tuple[int, int, int]) -> Tuple[Tuple[int,int,int], Tuple[int,int,int]]:
    """Return (+90°, −90°) using HSL rotation."""
    return _rotate_hls(rgb, 0.25), _rotate_hls(rgb, -0.25 % 1.0)


def rotate_90_hsv(rgb: Tuple[int, int, int]) -> Tuple[Tuple[int,int,int], Tuple[int,int,int]]:
    """Return (+90°, −90°) using HSV rotation."""
    return _rotate_hsv(rgb, 0.25), _rotate_hsv(rgb, -0.25 % 1.0)


def invert_rgb(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    r, g, b = rgb
    return (255 - r, 255 - g, 255 - b)


# Wheel generation
def hsv_to_rgb_np(h, s, v):
    h = (h % 1.0) * 6.0
    i = np.floor(h).astype(int)
    f = h - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    r = np.empty_like(h)
    g = np.empty_like(h)
    b = np.empty_like(h)

    i_mod = np.mod(i, 6)
    mask0 = i_mod == 0
    mask1 = i_mod == 1
    mask2 = i_mod == 2
    mask3 = i_mod == 3
    mask4 = i_mod == 4
    mask5 = i_mod == 5

    r[mask0], g[mask0], b[mask0] = v[mask0], t[mask0], p[mask0]
    r[mask1], g[mask1], b[mask1] = q[mask1], v[mask1], p[mask1]
    r[mask2], g[mask2], b[mask2] = p[mask2], v[mask2], t[mask2]
    r[mask3], g[mask3], b[mask3] = p[mask3], q[mask3], v[mask3]
    r[mask4], g[mask4], b[mask4] = t[mask4], p[mask4], v[mask4]
    r[mask5], g[mask5], b[mask5] = v[mask5], p[mask5], q[mask5]

    return r, g, b


def generate_color_wheel_image(size: int = 560) -> Image.Image:
    radius = size // 2
    # coordinate grid centered at (0,0)
    y, x = np.ogrid[-radius:radius, -radius:radius]
    dist = np.sqrt(x * x + y * y)
    angle = np.arctan2(y, x)  # -pi..pi

    h = ((angle + np.pi) / (2 * np.pi))  # 0..1
    s = np.clip(dist / radius, 0, 1)
    v = np.ones_like(h)

    mask = dist <= radius

    r, g, b = hsv_to_rgb_np(h, s, v)
    rgb = np.stack([r, g, b], axis=-1)

    # Outside the circle -> white
    rgb[~mask] = 1.0

    img = (rgb * 255).astype(np.uint8)
    return Image.fromarray(img, mode="RGB")


def _marker_xy(size: int, hue_01: float, sat_01: float) -> Tuple[float, float]:
    radius = size / 2.0
    cx = cy = radius
    angle = hue_01 * 2 * math.pi - math.pi
    # keep ~4% margin from outer edge
    r_max = radius * 0.96
    r_marker = max(0.0, min(sat_01, 1.0)) * r_max
    x = cx + r_marker * math.cos(angle)
    y = cy + r_marker * math.sin(angle)
    return x, y


def draw_markers_on_wheel(
    img,
    base_h: float,         
    base_s: float,          
    orig_rgb,
    comp_rgb,
    plus90_rgb,
    minus90_rgb,
) -> Image.Image:
    size = img.size[0]

    # Hue positions derived from base hue
    h0 = base_h % 1.0
    h180 = (h0 + 0.5) % 1.0
    hplus = (h0 + 0.25) % 1.0
    hminus = (h0 - 0.25) % 1.0

    x0, y0 = _marker_xy(size, h0, base_s)
    x180, y180 = _marker_xy(size, h180, base_s)
    xp, yp = _marker_xy(size, hplus, base_s)
    xm, ym = _marker_xy(size, hminus, base_s)

    draw = ImageDraw.Draw(img)

    # Lines: original<->complement and +90°<->−90°
    draw.line([(x0, y0), (x180, y180)], fill=(0, 0, 0), width=2)
    draw.line([(xp, yp), (xm, ym)], fill=(0, 0, 0), width=2)

    def draw_dot(x, y, color, r=9):
        bbox = [x - r, y - r, x + r, y + r]
        draw.ellipse(bbox, fill=color, outline=(0, 0, 0), width=2)

    draw_dot(x0, y0, orig_rgb)
    draw_dot(x180, y180, comp_rgb)
    draw_dot(xp, yp, plus90_rgb)
    draw_dot(xm, ym, minus90_rgb)

    # Labels
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    draw.text((x0 + 10, y0), f"0°\n{rgb_to_hex(orig_rgb)}", fill=(0, 0, 0), font=font)
    draw.text((xp + 10, yp), f"+90°\n{rgb_to_hex(plus90_rgb)}", fill=(0, 0, 0), font=font)
    draw.text((x180 + 10, y180), f"180°\n{rgb_to_hex(comp_rgb)}", fill=(0, 0, 0), font=font)
    draw.text((xm + 10, ym), f"−90°\n{rgb_to_hex(minus90_rgb)}", fill=(0, 0, 0), font=font)

    # Outer ring
    draw.ellipse([2, 2, size - 2, size - 2], outline=(0, 0, 0), width=1)
    return img


def make_swatches_image_four(
    orig: Tuple[int, int, int],
    plus90: Tuple[int, int, int],
    comp180: Tuple[int, int, int],
    minus90: Tuple[int, int, int],
    width=760,
    height=160,
) -> Image.Image:
    img = Image.new("RGB", (width, height), (255, 255, 255))
    sw = width // 4
    blocks = [orig, plus90, comp180, minus90]
    labels = ["0°", "+90°", "180°", "−90°"]

    for i, (rgb, lab) in enumerate(zip(blocks, labels)):
        x0 = i * sw
        block = Image.new("RGB", (sw, height), rgb)
        img.paste(block, (x0, 0))
        # Add label
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        txt = f"{lab}\n{rgb_to_hex(rgb)}"
        draw.text((x0 + 6, 6), txt, fill=(0, 0, 0), font=font)

    return img


# ---------------------------
# Tkinter GUI
# ---------------------------
class ComplementGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Complementary & 90° Colors (Clickable & Draggable Wheel + Randomize)")
        self.geometry("1000x900")
        self.resizable(True, True)

        # Wheel size (fixed so click coords match pixel coords)
        self.wheel_size = 560

        # Track current HSV pick (hue & saturation used for marker positions)
        self.pick_h = 0.0  # 0..1
        self.pick_s = 1.0  # 0..1

        # State
        self.current_wheel_img = None  # PIL Image
        self.current_wheel_tk = None   # ImageTk (to keep reference)
        self.current_swatches_img = None
        self.current_swatches_tk = None

        # UI
        self._build_ui()

    def _build_ui(self):
        pad = 8

        # Top controls
        ctrl = ttk.Frame(self)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=pad, pady=pad)

        ttk.Label(ctrl, text="Color code:").grid(row=0, column=0, sticky="w")
        self.entry_color = ttk.Entry(ctrl, width=24)
        self.entry_color.grid(row=0, column=1, sticky="w", padx=(5, 15))
        self.entry_color.insert(0, "#3498db")

        ttk.Label(ctrl, text="Model:").grid(row=0, column=2, sticky="w")
        self.model_var = tk.StringVar(value="hsl")
        self.combo_model = ttk.Combobox(
            ctrl, textvariable=self.model_var,
            values=["hsl", "hsv", "invert"],
            width=8, state="readonly"
        )
        self.combo_model.grid(row=0, column=3, sticky="w", padx=(5, 15))

        ttk.Button(ctrl, text="Compute", command=self.on_compute).grid(row=0, column=4, padx=(0, 10))
        ttk.Button(ctrl, text="Save Wheel...", command=self.on_save_wheel).grid(row=0, column=5, padx=(0, 10))
        ttk.Button(ctrl, text="Save Swatches...", command=self.on_save_swatches).grid(row=0, column=6, padx=(0, 10))
        ttk.Button(ctrl, text="Randomize", command=self.on_randomize).grid(row=0, column=7)

        # Hex results
        res = ttk.Frame(self)
        res.pack(side=tk.TOP, fill=tk.X, padx=pad, pady=pad)

        self.hex_vars = {
            "0": tk.StringVar(value=""),
            "+90": tk.StringVar(value=""),
            "180": tk.StringVar(value=""),
            "-90": tk.StringVar(value=""),
        }

        def make_hex_row(row, label, key):
            ttk.Label(res, text=label).grid(row=row, column=0, sticky="e")
            ent = ttk.Entry(res, textvariable=self.hex_vars[key], width=16)
            ent.grid(row=row, column=1, sticky="w", padx=(5, 10))
            ttk.Button(res, text="Copy", command=lambda: self._copy(self.hex_vars[key].get())).grid(row=row, column=2, padx=(0, 15))

        make_hex_row(0, "0° (Original):", "0")
        make_hex_row(1, "+90°:", "+90")
        make_hex_row(2, "180° (Complement):", "180")
        make_hex_row(3, "−90°:", "-90")

        # Swatch previews (four)
        swatch_frame = ttk.Frame(self)
        swatch_frame.pack(side=tk.TOP, fill=tk.X, padx=pad, pady=(0, pad))

        ttk.Label(swatch_frame, text="0°").grid(row=0, column=0, sticky="w")
        ttk.Label(swatch_frame, text="+90°").grid(row=0, column=1, sticky="w", padx=(12,0))
        ttk.Label(swatch_frame, text="180°").grid(row=0, column=2, sticky="w", padx=(12,0))
        ttk.Label(swatch_frame, text="−90°").grid(row=0, column=3, sticky="w", padx=(12,0))

        self.canvas_0 = tk.Canvas(swatch_frame, width=200, height=60, bd=1, relief="solid")
        self.canvas_p = tk.Canvas(swatch_frame, width=200, height=60, bd=1, relief="solid")
        self.canvas_180 = tk.Canvas(swatch_frame, width=200, height=60, bd=1, relief="solid")
        self.canvas_m = tk.Canvas(swatch_frame, width=200, height=60, bd=1, relief="solid")

        self.canvas_0.grid(row=1, column=0, padx=(0, 12), pady=(3, 3))
        self.canvas_p.grid(row=1, column=1, padx=(0, 12), pady=(3, 3))
        self.canvas_180.grid(row=1, column=2, padx=(0, 12), pady=(3, 3))
        self.canvas_m.grid(row=1, column=3, pady=(3, 3))

        # Wheel preview
        wheel_frame = ttk.Frame(self)
        wheel_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=pad, pady=pad)

        ttk.Label(
            wheel_frame,
            text="Color Wheel (click/drag: Hue = angle • Saturation = radius • Value = 1)"
        ).pack(anchor="w")

        self.wheel_label = ttk.Label(wheel_frame)
        self.wheel_label.pack(pady=(5, 0))
        # Bind click & drag to pick color (both hue & saturation)
        self.wheel_label.bind("<Button-1>", self.on_wheel_click)
        self.wheel_label.bind("<B1-Motion>", self.on_wheel_click)

        # Run once with default
        self.after(200, self.on_compute)

    def _copy(self, text: str):
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update()
        messagebox.showinfo("Copied", f"Copied: {text}")

    # ---------- RANDOMIZE ----------
    def on_randomize(self):
        h = random.random()                # 0..1
        s = max(0.15, random.random())     # avoid near-gray; tweak lower bound if desired
        v = 1.0

        rr, gg, bb = colorsys.hsv_to_rgb(h, s, v)
        rgb = (int(round(rr * 255)), int(round(gg * 255)), int(round(bb * 255)))

        # Save pick hue & saturation for marker placement
        self.pick_h = h
        self.pick_s = s

        # Update the color entry and recompute
        self.entry_color.delete(0, tk.END)
        self.entry_color.insert(0, rgb_to_hex(rgb))
        self.on_compute()

    # ---------- CLICK HANDLER ----------
    def on_wheel_click(self, event: tk.Event):
        if self.current_wheel_img is None:
            return

        size = self.wheel_size
        x, y = event.x, event.y
        if not (0 <= x < size and 0 <= y < size):
            return

        cx = cy = size / 2.0
        dx = x - cx
        dy = y - cy
        r = math.hypot(dx, dy)
        radius = size / 2.0

        if r > radius:
            # Clamp to edge if slightly outside
            r = radius

        # Hue in [0,1] from angle; Saturation from radius; Value fixed at 1
        hue = (math.atan2(dy, dx) + math.pi) / (2 * math.pi)  # 0..1
        sat = min(1.0, r / radius)
        val = 1.0

        rr, gg, bb = colorsys.hsv_to_rgb(hue, sat, val)
        rgb = (int(round(rr * 255)), int(round(gg * 255)), int(round(bb * 255)))

        # Save pick hue & saturation for marker placement
        self.pick_h = hue
        self.pick_s = sat

        # Update the color entry and recompute the rest
        self.entry_color.delete(0, tk.END)
        self.entry_color.insert(0, rgb_to_hex(rgb))
        self.on_compute()

    def on_compute(self):
        code = self.entry_color.get()
        model = (self.model_var.get() or "hsl").lower()
        try:
            orig_rgb = parse_color(code)
        except Exception as e:
            messagebox.showerror("Invalid color", str(e))
            return

        # Sync the wheel markers to the current text color (HSV)
        r, g, b = [v / 255.0 for v in orig_rgb]
        h0, s0, v0 = colorsys.rgb_to_hsv(r, g, b)
        # Markers reflect the current color in the entry box:
        self.pick_h, self.pick_s = h0, s0

        # Compute 180° (complement) and ±90°
        if model in ("hsl", "hls"):
            comp_rgb = complementary_hsl(orig_rgb)      # 180° in HSL
            plus90_rgb, minus90_rgb = rotate_90_hsl(orig_rgb)
        elif model == "hsv":
            comp_rgb = complementary_hsv(orig_rgb)      # 180° in HSV
            plus90_rgb, minus90_rgb = rotate_90_hsv(orig_rgb)
        elif model in ("invert", "rgb"):
            # Complement via inversion; ±90° via HSV hue rotation so wheel positions make sense.
            comp_rgb = invert_rgb(orig_rgb)
            plus90_rgb, minus90_rgb = rotate_90_hsv(orig_rgb)
        else:
            messagebox.showerror("Error", "Model must be one of: hsl, hsv, invert")
            return

        # Update hex outputs
        self.hex_vars["0"].set(rgb_to_hex(orig_rgb))
        self.hex_vars["+90"].set(rgb_to_hex(plus90_rgb))
        self.hex_vars["180"].set(rgb_to_hex(comp_rgb))
        self.hex_vars["-90"].set(rgb_to_hex(minus90_rgb))

        # Update swatch canvases
        def fill(canvas: tk.Canvas, hexcol: str):
            canvas.delete("all")
            canvas.create_rectangle(0, 0, int(canvas["width"]), int(canvas["height"]), fill=hexcol, outline="")
        fill(self.canvas_0, rgb_to_hex(orig_rgb))
        fill(self.canvas_p, rgb_to_hex(plus90_rgb))
        fill(self.canvas_180, rgb_to_hex(comp_rgb))
        fill(self.canvas_m, rgb_to_hex(minus90_rgb))

        # Wheel with markers (fixed size for consistent picking coordinates)
        wheel = generate_color_wheel_image(size=self.wheel_size)
        wheel = draw_markers_on_wheel(
            wheel,
            base_h=self.pick_h,
            base_s=self.pick_s,
            orig_rgb=orig_rgb,
            comp_rgb=comp_rgb,
            plus90_rgb=plus90_rgb,
            minus90_rgb=minus90_rgb,
        )
        self.current_wheel_img = wheel
        self.current_wheel_tk = ImageTk.PhotoImage(wheel)
        self.wheel_label.configure(image=self.current_wheel_tk)
        self.wheel_label.configure(width=self.wheel_size, height=self.wheel_size)

        # 4-color swatch strip for saving
        sw_img = make_swatches_image_four(orig_rgb, plus90_rgb, comp_rgb, minus90_rgb)
        self.current_swatches_img = sw_img
        self.current_swatches_tk = ImageTk.PhotoImage(sw_img)

    def on_save_wheel(self):
        if self.current_wheel_img is None:
            messagebox.showwarning("No image", "Please compute a color first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")],
            title="Save Color Wheel As..."
        )
        if path:
            self.current_wheel_img.save(path)
            messagebox.showinfo("Saved", f"Saved wheel to:\n{path}")

    def on_save_swatches(self):
        if self.current_swatches_img is None:
            messagebox.showwarning("No image", "Please compute a color first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")],
            title="Save Swatches As..."
        )
        if path:
            self.current_swatches_img.save(path)
            messagebox.showinfo("Saved", f"Saved swatches to:\n{path}")


if __name__ == "__main__":
    app = ComplementGUI()
    app.mainloop()