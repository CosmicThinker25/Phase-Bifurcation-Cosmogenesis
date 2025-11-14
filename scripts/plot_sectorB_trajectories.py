"""
plot_sectorB_trajectories.py
---------------------------------

This script visualizes Δφ(a) for several points in the plateau-B region near
(m_phi ~ 0.40, k_rot ~ 0.38–0.39), confirming the true behavior
of the "sector B" solutions.

CosmicThinker & Toko – Phase-Bifurcation Cosmogenesis Project (2025)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from phase_evolution_ode import run_phase_evolution


# ================================================================
# Configuration of the B-region scan
# ================================================================

# A small set of (m_phi, k_rot) pairs inside the observed B plateau
m_values = np.linspace(0.395, 0.405, 5)
k_values = np.linspace(0.380, 0.390, 5)

# Make 10 combinations
params_B = []
for m in m_values:
    for k in k_values:
        params_B.append((m, k))

# Reduce to 10 points
params_B = params_B[:10]

# Initial conditions known to produce B
delta_phi_ini = 1.57      # π/2
delta_phidot_ini = 0.0
q = 1.0                   # rotation falloff


# Output directory
outdir = "results_phase_sectors"
os.makedirs(outdir, exist_ok=True)


# ================================================================
# Produce the trajectories
# ================================================================

plt.figure(figsize=(10, 6))

for (m_phi, k_rot) in params_B:

    a, dphi, dphidot = run_phase_evolution(
        m_phi=m_phi,
        k_rot=k_rot,
        q=q,
        delta_phi_ini=delta_phi_ini,
        delta_phidot_ini=delta_phidot_ini,
        a_ini=1e-3,
        a_max=10.0,
        n_steps=2000,
        H0=1.0
    )

    label = f"m={m_phi:.4f}, k={k_rot:.4f}"
    plt.plot(a, dphi, lw=1.5, alpha=0.85, label=label)


# ================================================================
# Plot styling
# ================================================================

plt.xlabel("Scale factor a", fontsize=14)
plt.ylabel(r"$\Delta\phi(a)$", fontsize=14)
plt.title("Sector B Trajectories Near (mφ≈0.40, krot≈0.38–0.39)", fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=9)

outfile = os.path.join(outdir, "sectorB_trajectories.png")
plt.tight_layout()
plt.savefig(outfile, dpi=170)
plt.show()

print("\nSaved sector-B trajectory figure →", outfile)
