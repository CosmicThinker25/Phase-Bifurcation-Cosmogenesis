"""
phase_sector_map.py — FINAL FIXED VERSION
-----------------------------------------

Generates a 2D map (m_phi × k_rot) showing the asymptotic
sector classification (A/B/C) using the results produced by
phase_sector_scan.py (Phase 2.1).

This version:
  • Fixes deprecated colormap call
  • Fixes KeyError for "{\\rm rot}" in title
  • Uses modern pyplot.colormaps
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt


# ===============================================================
# 1. Load summary CSV
# ===============================================================

SUMMARY_FILE = "results_phase_sectors/phase_sectors_summary.csv"

if not os.path.isfile(SUMMARY_FILE):
    raise RuntimeError("phase_sectors_summary.csv not found. "
                       "Run phase_sector_scan.py first.")

rows = []
with open(SUMMARY_FILE, "r") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

# Convert types
for r in rows:
    r["m_phi"] = float(r["m_phi"])
    r["k_rot"] = float(r["k_rot"])
    r["delta_phi_ini"] = float(r["delta_phi_ini"])
    r["sector"] = r["sector"]


# ===============================================================
# 2. Choose subset for a fixed q (all rows have q=1.0)
# ===============================================================

q_plot = 1.0
rows_q = [r for r in rows if abs(float(r["q"]) - q_plot) < 1e-9]

if len(rows_q) == 0:
    raise RuntimeError("No rows found for q={}".format(q_plot))

# Extract unique parameter values
m_list = sorted(set(r["m_phi"] for r in rows_q))
k_list = sorted(set(r["k_rot"] for r in rows_q))
dphi_list = sorted(set(r["delta_phi_ini"] for r in rows_q))

# We choose the Δφ_ini = 0.01 slice for the 2D map
target_ini = 0.01
rows_map = [r for r in rows_q if abs(r["delta_phi_ini"] - target_ini) < 1e-6]

if len(rows_map) == 0:
    raise RuntimeError("No rows found with delta_phi_ini = {}".format(target_ini))


# ===============================================================
# 3. Build sector map (integer codes)
# ===============================================================

# Sector coding
code = {"A": 0, "B": 1, "C": 2}

# Initialize grid
grid = np.zeros((len(m_list), len(k_list)), dtype=int)

for r in rows_map:
    i = m_list.index(r["m_phi"])
    j = k_list.index(r["k_rot"])
    sector = r["sector"]
    grid[i, j] = code.get(sector, 2)

# ===============================================================
# 4. Plot
# ===============================================================

plt.figure(figsize=(8, 6))

# Modern colormap usage
cmap = plt.colormaps.get_cmap("viridis").resampled(3)

im = plt.imshow(
    grid,
    origin="lower",
    cmap=cmap,
    extent=[min(k_list), max(k_list), min(m_list), max(m_list)],
    aspect="auto"
)

plt.colorbar(im, ticks=[0, 1, 2], label="Sector")
plt.clim(-0.5, 2.5)

plt.xlabel(r"$k_{\rm rot}$")
plt.ylabel(r"$m_\phi$")

# FIXED: f-string avoids KeyError
plt.title(
    fr"Asymptotic Phase Sectors in $(m_\phi, k_{{\rm rot}})$ for $q={q_plot}$"
)

plt.grid(alpha=0.25)
plt.tight_layout()

OUT_PATH = "results_phase_sectors/phase_sector_map.png"
plt.savefig(OUT_PATH, dpi=300)
plt.show()

print("\nMAP GENERATED:", OUT_PATH)
