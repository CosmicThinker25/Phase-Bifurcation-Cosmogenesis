import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from phase_evolution_ode import run_phase_evolution
from phase_classifier import classify_sector   # clasificador A/B/C

# ================================
#  CONFIGURACIÓN DEL BARRIDO 2D
# ================================

mphi_values = np.linspace(0.20, 1.00, 40)          # 40 puntos
dphi_ini_values = np.linspace(0.0, np.pi, 80)      # 80 puntos
k_rot = 0.3835                                     # eje casi crítico

results = []

print("\n==============================================")
print("     PHASE 3 — 2D MAP (mφ, Δφ_ini)")
print("==============================================\n")

for mphi in mphi_values:
    print(f"Scanning m_phi = {mphi:.4f} ...")

    for dphi_ini in dphi_ini_values:

        # Integrar trayectoria con el solver correcto
        a, dphi, _ = run_phase_evolution(
            m_phi=mphi,
            k_rot=k_rot,
            q=1.0,
            delta_phi_ini=dphi_ini,
            delta_phidot_ini=0.0
        )

        # Clasificar
        sector = classify_sector(a, dphi)

        results.append([mphi, dphi_ini, sector])

# ================================
#   GUARDAR CSV
# ================================
df = pd.DataFrame(results, columns=["m_phi", "delta_phi_ini", "sector"])
df.to_csv("results_phase_sectors/phase_sector_map_delta.csv", index=False)

print("\nSaved CSV → results_phase_sectors/phase_sector_map_delta.csv")

# ================================
#   MAPA 2D
# ================================

sector_map = {"A": 0, "B": 1, "C": 2}

Z = np.array([sector_map[s] for s in df["sector"]]).reshape(
    len(mphi_values), len(dphi_ini_values)
)

plt.figure(figsize=(12, 6))
plt.imshow(
    Z,
    origin="lower",
    extent=[0, np.pi, mphi_values[0], mphi_values[-1]],
    aspect="auto",
    cmap="viridis"
)

plt.colorbar(
    ticks=[0, 1, 2],
    label="Sector"
).ax.set_yticklabels(["A", "B", "C"])

plt.xlabel(r"$\Delta\phi_{\rm ini}$")
plt.ylabel(r"$m_\phi$")
plt.title(r"Phase Sector Map in $(m_\phi,\,\Delta\phi_{\rm ini})$ at $k_{\rm rot}=0.3835$")

plt.tight_layout()
plt.savefig("results_phase_sectors/phase_sector_map_delta.png", dpi=180)

print("Saved figure → results_phase_sectors/phase_sector_map_delta.png")
print("\nDONE.\n")
