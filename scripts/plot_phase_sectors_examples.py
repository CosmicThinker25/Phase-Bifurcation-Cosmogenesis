"""
plot_phase_sectors_examples.py (robust version)
----------------------------------------------

Plots representative examples of sectors A/B/C,
but does NOT fail if one of the sectors is absent.

If some sectors are missing (e.g., no B),
they are simply skipped and a warning is printed.

Output:
    results_phase_sectors/phase_sectors_examples.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ====================================================
# 1. Locate trajectories
# ====================================================

BASE_DIR = "results_phase_sectors"
TRAJ_DIR = os.path.join(BASE_DIR, "trajectories")

if not os.path.isdir(TRAJ_DIR):
    raise RuntimeError("Trajectory directory not found. "
                       "Run phase_sector_scan.py first.")

examples = {"A": None, "B": None, "C": None}

# Search for one file per sector
for fname in os.listdir(TRAJ_DIR):
    if not fname.endswith(".npz"):
        continue

    path = os.path.join(TRAJ_DIR, fname)
    data = np.load(path)
    sector = str(data["sector"])

    if sector in examples and examples[sector] is None:
        examples[sector] = path

# ====================================================
# 2. Load trajectories that exist
# ====================================================

loaded = {}

for S in ["A", "B", "C"]:
    if examples[S] is None:
        print(f"WARNING: Sector {S} not found in dataset — skipping.")
    else:
        d = np.load(examples[S])
        loaded[S] = {
            "a": d["a"],
            "phi": d["delta_phi"],
            "m": float(d["m_phi"]),
            "k": float(d["k_rot"]),
            "q": float(d["q"]),
        }


# ====================================================
# 3. Plot (only available sectors)
# ====================================================

plt.figure(figsize=(8.5,5.5))

colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green"}

for S in loaded:
    a = loaded[S]["a"]
    phi = loaded[S]["phi"]
    m = loaded[S]["m"]
    k = loaded[S]["k"]
    q = loaded[S]["q"]

    plt.plot(a, phi, label=f"Sector {S} (m={m}, k={k}, q={q})",
             linewidth=2.0, color=colors[S])

plt.axhline(np.pi, color="gray", linestyle="--", linewidth=1.2, label=r"$\Delta\phi=\pi$")

plt.xlabel("Scale factor $a$")
plt.ylabel(r"$\Delta\phi(a)$")
plt.title("Representative Siamese Phase Trajectories (A/B/C)")

plt.legend(frameon=False)
plt.grid(alpha=0.2)
plt.tight_layout()

out_path = os.path.join(BASE_DIR, "phase_sectors_examples.png")
plt.savefig(out_path, dpi=300)
plt.show()

print("\nFIGURE GENERATED:", out_path)
