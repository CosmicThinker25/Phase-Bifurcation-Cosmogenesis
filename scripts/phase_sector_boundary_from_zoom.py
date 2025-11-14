"""
phase_sector_boundary_from_zoom.py
----------------------------------
Reconstruye la curva de frontera A–C usando los mapas de zoom fractal:

    Zone1.csv
    Zone2.csv
    Zone3.csv

Cada archivo debe contener columnas:
    m_phi, Delta_phi_ini, sector

El script:
  1. Carga los tres CSV
  2. Los concatena
  3. Busca puntos donde sector cambia entre A y C
  4. Extrae una curva límite aproximada
  5. Guarda boundary_zoom.csv
  6. Genera la figura phase_sector_boundary_from_zoom.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# 1. Carga de CSVs
# ============================================================

print("\n========================================")
print(" Loading Zone CSVs from zoom_fractal ...")
print("========================================\n")

base = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(base, "zoom_fractal")

zone_files = ["Zone1.csv", "Zone2.csv", "Zone3.csv"]
dfs = []

for zfile in zone_files:
    path = os.path.join(folder, zfile)
    if not os.path.exists(path):
        print(f"ERROR: No se encuentra {path}")
        exit(1)
    print(f" → Loaded: {zfile}")
    dfs.append(pd.read_csv(path))

# Concatenar
data = pd.concat(dfs, ignore_index=True)

# Normalizar columnas si vienen con nombres raros
data.columns = [c.strip() for c in data.columns]
if "m_phi" not in data.columns or "Delta_phi_ini" not in data.columns or "sector" not in data.columns:
    raise ValueError("ERROR: Las columnas deben ser m_phi, Delta_phi_ini, sector")

# ============================================================
# 2. Convertir sectores a código numérico A=0, B=1, C=2
# ============================================================

sector_map = {"A": 0, "B": 1, "C": 2}
data["sector_code"] = data["sector"].map(sector_map)

# ============================================================
# 3. Hallar frontera A–C (ignorando B)
# ============================================================

print("\n========================================")
print(" Detecting A–C boundary transitions ...")
print("========================================\n")

boundary_points = []

# Ordenar por m_phi
m_values = np.sort(data["m_phi"].unique())

for i, m in enumerate(m_values):
    df_m = data[data["m_phi"] == m].sort_values("Delta_phi_ini")

    dphi_vals = df_m["Delta_phi_ini"].values
    sec_vals = df_m["sector_code"].values

    # Buscar transición AC o CA
    for j in range(len(sec_vals) - 1):
        s1, s2 = sec_vals[j], sec_vals[j+1]
        if {s1, s2} == {0, 2}:  # A ↔ C
            # punto medio
            k_est = (dphi_vals[j] + dphi_vals[j+1]) / 2.0
            boundary_points.append((m, k_est))

    print(f"Row {i+1}/{len(m_values)} processed...")

boundary_points = np.array(boundary_points)

if boundary_points.size == 0:
    print("\nWARNING: No A–C boundary found.")
else:
    print(f"\nFound {len(boundary_points)} boundary points.")

# ============================================================
# 4. Guardar resultados
# ============================================================

out_csv = os.path.join(base, "results_phase_sectors", "boundary_zoom.csv")
os.makedirs(os.path.dirname(out_csv), exist_ok=True)

if boundary_points.size > 0:
    dfb = pd.DataFrame(boundary_points, columns=["m_phi", "Delta_phi_crit"])
    dfb.to_csv(out_csv, index=False)
    print(f"\nBoundary saved → {out_csv}")
else:
    print("\nNo boundary to save.")

# ============================================================
# 5. Figura
# ============================================================

fig = plt.figure(figsize=(10, 6))

if boundary_points.size > 0:
    plt.scatter(boundary_points[:, 1], boundary_points[:, 0],
                c="red", s=20, label="Boundary A–C")

plt.xlabel(r"$\Delta\phi_{\rm ini}$")
plt.ylabel(r"$m_\phi$")
plt.title("A–C Boundary Extracted from Fractal Zoom Maps")
plt.grid(True)

out_fig = os.path.join(base, "results_phase_sectors", "phase_sector_boundary_from_zoom.png")
plt.savefig(out_fig, dpi=200)
plt.close()

print(f"Figure saved → {out_fig}")

print("\n========================================")
print(" DONE — Zoom-based boundary extraction")
print("========================================\n")
