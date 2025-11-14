"""
PHASE 2.4c — MICRO-FINE CRITICAL SCAN
Ultra-high-resolution boundary detection near the critical point:

    m_phi  ∈ [0.395, 0.405]   step = 0.0002
    k_rot  ∈ [0.380, 0.390]   step = 0.0002

This isolates the neighbourhood of the known critical point:
    (m_phi ≈ 0.400,  k_rot ≈ 0.3835)

We use Δφ_ini = 2.827 rad (antipodal), which maximizes the
sensitivity of the A/C transition.

Outputs:
    results_phase_sectors/boundary_microfine.csv
    results_phase_sectors/boundary_microfine.png
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from phase_evolution_ode import run_phase_evolution

# ------------------------------------------------------------------
# 1. SCAN PARAMETERS
# ------------------------------------------------------------------

m_values = np.arange(0.395, 0.405 + 1e-12, 0.0002)
k_values = np.arange(0.380, 0.390 + 1e-12, 0.0002)

delta_phi_ini = 2.827433388   # antipodal for maximum sensitivity
delta_phidot_ini = 0
q = 1.0
a_ini = 1e-3
a_max = 10.0
tail_frac = 0.15

outdir = "results_phase_sectors"
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, "boundary_microfine.csv")

# CSV header
with open(outfile, "w") as f:
    f.write("m_phi,k_rot,sector\n")

print("\n===============================================")
print("  PHASE 2.4c — MICRO-FINE CRITICAL SCAN")
print("===============================================\n")

boundary_points = []

def classify_sector(a, dphi):
    """Classify A/B/C using tail statistics."""
    tail = dphi[int(len(dphi)*(1-tail_frac)):]
    mean_tail = np.mean(tail)
    std_tail = np.std(tail)

    if std_tail < 0.05:
        if abs(mean_tail) < 2.5:
            return "A"
        else:
            return "B"
    else:
        return "C"


# ------------------------------------------------------------------
# 2. MAIN SCAN
# ------------------------------------------------------------------
for m_phi in m_values:
    print(f"Scanning m_phi = {m_phi:.6f} ...")
    last_sector = None

    for k_rot in k_values:
        # integrate
        a, dphi, _ = run_phase_evolution(
            m_phi=m_phi,
            k_rot=k_rot,
            q=q,
            delta_phi_ini=delta_phi_ini,
            delta_phidot_ini=delta_phidot_ini,
            a_ini=a_ini,
            a_max=a_max,
            n_steps=1500,
        )

        sector = classify_sector(a, dphi)

        # append row
        with open(outfile, "a") as f:
            f.write(f"{m_phi},{k_rot},{sector}\n")

        # detect boundary
        if last_sector is not None and sector != last_sector:
            boundary_points.append((m_phi, k_rot))
            print(f"  >>> boundary at k_rot = {k_rot:.6f}")

        last_sector = sector


# ------------------------------------------------------------------
# 3. PLOT BOUNDARY
# ------------------------------------------------------------------
plt.figure(figsize=(6,5))

if boundary_points:
    M, K = zip(*boundary_points)
    plt.scatter(M, K, c="red", s=22, label="Boundary A/C")
else:
    plt.text(0.397, 0.385, "No boundary detected", fontsize=12)

plt.xlabel(r"$m_\phi$")
plt.ylabel(r"$k_{\rm rot}$")
plt.title(r"Micro-fine A/C Boundary Near $(m_\phi,k_{\rm rot})\approx(0.40,0.3835)$")
plt.grid(True)
plt.tight_layout()

figfile = os.path.join(outdir, "boundary_microfine.png")
plt.savefig(figfile, dpi=160)

print("\nSaved CSV →", outfile)
print("Saved figure →", figfile)
print("\nDONE.\n")
