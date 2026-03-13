#!/usr/bin/env python3

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D   # enables 3D plotting

FOLDER = "../data/errorfiles/thresholdsweep"
LAST_N = 10_000

# Pattern: thr_hid=N_thr_out=N_trial=N.txt
pattern = re.compile(r"thr_hid=(\d+)_thr_out=(\d+)_trial=(\d+)\.txt")

# Store hit rates per (thr_hid, thr_out)
data = defaultdict(list)

for fname in os.listdir(FOLDER):
    match = pattern.match(fname)
    if not match:
        continue

    thr_hid, thr_out, trial = map(int, match.groups())
    path = os.path.join(FOLDER, fname)

    values = np.loadtxt(path, delimiter=",")[:, 1]
    stable_values = values[-LAST_N:]
    hit_rate = stable_values.mean()

    data[(thr_hid, thr_out)].append(hit_rate)

# --------------------------------------------------
# Build grid
# --------------------------------------------------
hid_vals = sorted({k[0] for k in data.keys()})
out_vals = sorted({k[1] for k in data.keys()})

heatmap = np.full((len(hid_vals), len(out_vals)), np.nan)

for (hid, out), rates in data.items():
    i = hid_vals.index(hid)
    j = out_vals.index(out)
    heatmap[i, j] = np.mean(rates)

# Create meshgrid for surface
X, Y = np.meshgrid(out_vals, hid_vals)

# --------------------------------------------------
# Plot 3D surface
# --------------------------------------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(
    X,
    Y,
    heatmap,
    cmap="viridis",
    edgecolor="k",
    linewidth=0.3,
    antialiased=True
)

fig.colorbar(surf, ax=ax, shrink=0.6, label="Average Hit Rate")

ax.set_xlabel("thr_out")
ax.set_ylabel("thr_hid")
ax.set_zlabel("Average Hit Rate")
ax.set_title("3D Surface of Average Hit Rate (Last 10k Samples)")

plt.tight_layout()
plt.show()
