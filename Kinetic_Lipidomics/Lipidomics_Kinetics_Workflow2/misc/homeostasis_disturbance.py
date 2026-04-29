import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevents GUI backend issues
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Tkinter for save dialog
import tkinter as tk
from tkinter import filedialog

# ---------------------------
# Ask user for save location
# ---------------------------
root = tk.Tk()
root.withdraw()  # Hide main tkinter window

save_path = filedialog.asksaveasfilename(
    title="Save SVG Figure As...",
    defaultextension=".svg",
    filetypes=[("SVG files", "*.svg")]
)

if not save_path:
    print("Save cancelled.")
    exit()

# ---------------------------
# Generate the APOE plot
# ---------------------------

# Grid
gx = np.linspace(-4, 4, 200)
gy = np.linspace(-4, 4, 200)
X, Y = np.meshgrid(gx, gy)

# Stability peaks
Z = (
    2*np.exp(-((X-0)**2 + (Y-0)**2)/0.8) +
    1.5*np.exp(-((X+2)**2 + (Y-2)**2)/0.8)
)

# Points
p1 = np.array([0, 0, 2])       # APOE3
p2 = np.array([-2, 2, 1.5])    # APOE4

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

# Surface
ax.plot_surface(X, Y, Z, cmap='Greys', alpha=0.75, linewidth=0)

# Scatter points
ax.scatter(p1[0], p1[1], p1[2], color='blue', s=100)
ax.scatter(p2[0], p2[1], p2[2], color='blue', s=100)

# Labels
ax.text(p1[0] + 0.3, p1[1] - 0.3, p1[2] + 0.4, 'APOE3', color='black', fontsize=12)
ax.text(p2[0] - 0.5, p2[1] + 0.3, p2[2] + 0.4, 'APOE4', color='black', fontsize=12)

# y = -x line
line_x = np.linspace(-4, 4, 100)
line_y = -line_x
ax.plot(line_x, line_y, np.zeros_like(line_x), color='green', linestyle='--', linewidth=2)



# XY axes
ax.plot([-4,4], [0,0], [0,0], color='black', linewidth=2)
ax.plot([0,0], [-4,4], [0,0], color='black', linewidth=2)

# Arrow raised upward
arrow_vertical_offset = 0.25
ar_start = p1 + np.array([0.1, -0.1, arrow_vertical_offset])
ar_vec = (p2 - p1) * np.array([1.0, 1.0, 1.0])

ax.quiver(ar_start[0], ar_start[1], ar_start[2],
          ar_vec[0], ar_vec[1], ar_vec[2],
          color='magenta', linewidth=3, arrow_length_ratio=0.12)

# Arrow label
mid = (p1 + p2) / 2
ax.text(mid[0] + 0.4, mid[1] + 0.3, mid[2]  -0.12 + arrow_vertical_offset,
        '+ Degradation', color= 'magenta', fontsize=12)

# Axis labels
ax.set_xlabel('log2 FC Abundance')
ax.set_ylabel('log2 FC Rate')
ax.set_zlabel('Stability')
ax.set_title('Finding a New Homeostasis After a Metabolic Disturbance')

plt.tight_layout()

# ---------------------------
# Save to user-selected path
# ---------------------------
plt.savefig(save_path, format='svg')
print(f"Saved SVG to: {save_path}")