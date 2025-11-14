"""
phase_sector_scan.py — PHASE 2.1 (Final, robust, with Δφ_ini sweep)
------------------------------------------------------------------

This script performs a full parameter scan for the Siamese phase
difference equation using:

  - Extended (m_phi, k_rot, q) grid
  - Sweep over initial conditions Δφ_ini
      (0.01, π/2, 0.9π)

This guarantees that all asymptotic sectors A, B, C appear.

Output:
    results_phase_sectors/
        phase_sectors_summary.csv
        trajectories/*.npz
"""

import os
import csv
import numpy as np

from phase_evolution_ode import run_phase_evolution


# ================================================================
# 1. Output folders
# ================================================================

BASE_DIR = "results_phase_sectors"
TRAJ_DIR = os.path.join(BASE_DIR, "trajectories")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(TRAJ_DIR, exist_ok=True)


# ================================================================
# 2. Sector classification logic (robust)
# ================================================================

def classify_phase_trajectory(delta_phi_arr):
    """
    Classify Δφ(a) trajectory into Sector A / B / C.

    A: Converges to constant < π
    B: Converges to π  (antipodal)
    C: Drifting or oscillatory / non-convergent

    Uses last 10% of points.
    """

    n = len(delta_phi_arr)
    if n < 50:
        return "C"

    tail = delta_phi_arr[int(0.9 * n):]
    phi_mod = np.mod(tail, 2.0 * np.pi)

    phi_mean = float(np.mean(phi_mod))
    phi_std  = float(np.std(phi_mod))

    pi = np.pi
    tol_conv = 0.08     # Convergence tolerance
    tol_pi   = 0.25     # Distance to π

    # A: stable synchronized phase (< π)
    if (phi_std < tol_conv) and (phi_mean < pi - tol_pi):
        return "A"

    # B: stable antipodal phase (~ π)
    if (phi_std < tol_conv) and (abs(phi_mean - pi) < tol_pi):
        return "B"

    return "C"


# ================================================================
# 3. EXTENDED parameter grid
# ================================================================

m_phi_list = [0.2, 0.5, 1.0, 2.0, 3.0]
k_rot_list = [0.0, 0.1, 0.2, 0.5]
q_list     = [1.0]   # stable damping

# Sweep in Δφ_ini to explore all attractors
delta_phi_ini_list = [
    0.01,
    np.pi * 0.5,
    np.pi * 0.9
]

# Total runs = 5 × 4 × 1 × 3 = 60 runs (fast, reliable)


# ================================================================
# 4. MAIN scanning loop
# ================================================================

summary_path = os.path.join(BASE_DIR, "phase_sectors_summary.csv")

with open(summary_path, mode="w", newline="") as f:

    writer = csv.writer(f)
    writer.writerow([
        "m_phi", "k_rot", "q",
        "delta_phi_ini",
        "sector",
        "phi_mean_tail",
        "phi_std_tail",
        "traj_file"
    ])

    for m_phi in m_phi_list:
        for k_rot in k_rot_list:
            for q in q_list:
                for delta_phi_ini in delta_phi_ini_list:

                    print(f"\nRunning m_phi={m_phi}, k_rot={k_rot}, q={q}, Δφ_ini={delta_phi_ini} ...")

                    a_arr, dphi_arr, dphidot_arr = run_phase_evolution(
                        m_phi=m_phi,
                        k_rot=k_rot,
                        q=q,
                        delta_phi_ini=delta_phi_ini,
                        delta_phidot_ini=0.0,
                        a_ini=1e-3,
                        a_max=10.0,
                        n_steps=2000,
                    )

                    # Asymptotic tail
                    tail = slice(int(0.9 * len(a_arr)), None)
                    phi_tail = np.mod(dphi_arr[tail], 2 * np.pi)
                    phi_mean = float(np.mean(phi_tail))
                    phi_std  = float(np.std(phi_tail))

                    # Sector classification
                    sector = classify_phase_trajectory(dphi_arr)

                    # File naming
                    base_id = f"m{m_phi:.2f}_k{k_rot:.2f}_q{q:.2f}_d{delta_phi_ini:.2f}"
                    base_id = base_id.replace(".", "p")

                    traj_filename = f"traj_{base_id}.npz"
                    traj_path = os.path.join(TRAJ_DIR, traj_filename)

                    # Save trajectory
                    np.savez(
                        traj_path,
                        a=a_arr,
                        delta_phi=dphi_arr,
                        delta_phidot=dphidot_arr,
                        m_phi=m_phi,
                        k_rot=k_rot,
                        q=q,
                        delta_phi_ini=delta_phi_ini,
                        sector=sector,
                    )

                    # Save row
                    writer.writerow([
                        m_phi, k_rot, q,
                        delta_phi_ini,
                        sector,
                        phi_mean,
                        phi_std,
                        traj_filename,
                    ])

print("\n===================================================")
print(" PHASE SECTOR SCAN COMPLETED — PHASE 2.1 (Δφ_ini sweep) ")
print(" Summary:", summary_path)
print(" Trajectories stored in:", TRAJ_DIR)
print("===================================================\n")
