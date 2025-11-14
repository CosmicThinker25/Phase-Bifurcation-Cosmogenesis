"""
phase_sector_boundary_fine_scan.py
----------------------------------

PHASE 2.3b — Fine scan of the A/C boundary ("Martin curve") in the
(m_phi, k_rot) plane for the physical branch Δφ_ini = 0.01.

This version scans ONLY in the regions where sector transitions
actually happen (m_phi ~ 0.5 and m_phi ~ 2.0), based on the coarse
Phase 2.1 sector map.

Outputs:
  - results_phase_sectors/fine_boundary_points.csv
  - results_phase_sectors/phase_sector_boundary_fine.png
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# Correct import from your module:
from phase_evolution_ode import run_phase_evolution


# ================================================================
# PARAMETERS OF THE FINE SCAN
# ================================================================

q = 1.0
delta_phi_ini = 0.01

# m_phi values around the two actual transitions observed in Phase 2.1
m_values = [
    0.40, 0.45, 0.50, 0.55, 0.60,   # lower frontier region
    1.80, 1.90, 2.00, 2.10, 2.20    # upper frontier region
]

# Fine grid in k_rot
k_min, k_max = 0.00, 0.60
N_k = 80   # finer than before
k_values = np.linspace(k_min, k_max, N_k)

# Output folder
OUT_DIR = "results_phase_sectors"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(OUT_DIR, "fine_boundary_points.csv")
OUT_FIG = os.path.join(OUT_DIR, "phase_sector_boundary_fine.png")


# ================================================================
# CLASSIFIER
# ================================================================
def classify_sector(phi_array):
    """
    Return 'A' (synchrony) or 'C' (escape/drift).

    Uses the tail behaviour of Δφ(a) at large a.
    """
    tail = phi_array[-200:]
    mean_tail = np.mean(tail)
    std_tail = np.std(tail)

    # Synchrony A: small mean and small oscillation
    if mean_tail < 0.5 and std_tail < 0.1:
        return "A"

    return "C"


# ================================================================
# SCAN
# ================================================================
boundary_points = []

print("\n==============================")
print(" PHASE 2.3b — FINE BOUNDARY SCAN")
print("==============================\n")

for m in m_values:
    print(f"Scanning m_phi = {m} ...")

    sector_list = []

    for k in k_values:
        # Integrate Δφ(a) using your actual solver
        a_arr, dphi_arr, dphidot_arr = run_phase_evolution(
            m_phi=m,
            k_rot=k,
            q=q,
            delta_phi_ini=delta_phi_ini,
            delta_phidot_ini=0.0,
            a_ini=1e-3,
            a_max=10.0,
            n_steps=2000,
        )

        s = classify_sector(dphi_arr)
        sector_list.append(s)

    # Detect A → C transition
    idx_A = [i for i, s in enumerate(sector_list) if s == "A"]
    idx_C = [i for i, s in enumerate(sector_list) if s == "C"]

    if not idx_A or not idx_C:
        print(f"  No boundary for m_phi={m} (all A or all C)")
        continue

    last_A = max(idx_A)
    first_C = min(idx_C)

    if first_C <= last_A:
        print(f"  m_phi={m}: degenerate overlap, skipping")
        continue

    k_last_A = k_values[last_A]
    k_first_C = k_values[first_C]
    k_crit = 0.5 * (k_last_A + k_first_C)

    print(f"  Boundary detected at k_crit = {k_crit:.4f}")
    boundary_points.append((m, k_crit))


# ================================================================
# SAVE RESULTS
# ================================================================
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["m_phi", "k_crit"])
    for m, kc in boundary_points:
        w.writerow([m, kc])

print("\nSaved fine boundary points →", OUT_CSV)


# ================================================================
# PLOT A/C BOUNDARY
# ================================================================
plt.figure(figsize=(7, 5))

if boundary_points:
    m_b, k_b = zip(*boundary_points)
    plt.plot(k_b, m_b, "o-", color="white",
             markeredgecolor="black", markersize=7,
             label="A/C fine boundary (Martin curve)")
else:
    print("WARNING: No boundary points found.")

plt.xlabel(r"$k_{\rm rot}$")
plt.ylabel(r"$m_\phi$")
plt.title(r"Fine A/C Boundary Curve in $(m_\phi, k_{\rm rot})$")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)
plt.show()

print("Saved figure →", OUT_FIG)
