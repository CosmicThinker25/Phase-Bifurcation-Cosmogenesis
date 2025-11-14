#!/usr/bin/env python3
import os
import glob
import csv
import math
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results_phase_sectors")
OUTPUT_FIG = os.path.join(RESULTS_DIR, "PA_map_mphi_krot.png")


# ============================================================
# FUNCIÓN 1 — LECTURA ROBUSTA DE CSV
# Ignora boundary*, zoom*, y todo archivo sin datos completos
# ============================================================

def read_rows(results_dir):
    pattern = os.path.join(results_dir, "*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No se encontraron CSV en {results_dir}")

    rows = []
    total_files = len(files)

    print(f"[INFO] {total_files} CSV encontrados para el mapa 2D.")

    for idx, fname in enumerate(files, start=1):
        print(f"  -> ({idx}/{total_files}) Leyendo {os.path.basename(fname)}")

        with open(fname, "r", newline="") as f:
            reader = csv.DictReader(f)

            for r in reader:

                # Necesitamos al menos m_phi y k_rot
                try:
                    m_phi = float(r.get("m_phi"))
                    k_rot = float(r.get("k_rot", 0.0))
                except:
                    continue

                # Caso 1: CSV trae P_A
                if "P_A" in r and r["P_A"]:
                    try:
                        P_A = float(r["P_A"])
                        rows.append((m_phi, k_rot, P_A))
                    except:
                        pass
                    continue

                # Caso 2: calcular P_A desde N_A / N_total
                N_total_raw = r.get("N_total")
                N_A_raw = r.get("N_A")

                if not N_total_raw or not N_A_raw:
                    continue  # CSV boundary* → descartar

                try:
                    N_total = float(N_total_raw)
                    N_A = float(N_A_raw)
                except:
                    continue

                if N_total > 0:
                    P_A = N_A / N_total
                    rows.append((m_phi, k_rot, P_A))

    print(f"[INFO] Filas válidas para mapa 2D: {len(rows)}")
    return rows


# ============================================================
# FUNCIÓN 2 — CONSTRUIR LA REJILLA 2D
# m_phi → eje X
# k_rot → eje Y
# color → P_A promedio
# ============================================================

def build_grid(rows):
    grid = defaultdict(list)

    for m_phi, k_rot, P_A in rows:
        grid[(m_phi, k_rot)].append(P_A)

    mphi_vals = sorted({k[0] for k in grid})
    krot_vals = sorted({k[1] for k in grid})

    M = len(mphi_vals)
    K = len(krot_vals)

    PA_grid = np.full((K, M), np.nan)

    for i, k_rot in enumerate(krot_vals):
        for j, m_phi in enumerate(mphi_vals):
            vals = grid.get((m_phi, k_rot), [])
            if vals:
                PA_grid[i, j] = np.mean(vals)

    return np.array(mphi_vals), np.array(krot_vals), PA_grid


# ============================================================
# FUNCIÓN 3 — PINTAR EL MAPA 2D
# ============================================================

def plot_PA_map(mphis, krots, PA_grid, outfile):
    plt.figure(figsize=(10, 6))

    im = plt.imshow(
        PA_grid,
        origin="lower",
        aspect="auto",
        extent=[mphis.min(), mphis.max(), krots.min(), krots.max()],
        interpolation="nearest",
        cmap="viridis"
    )

    cbar = plt.colorbar(im)
    cbar.set_label("P_A")

    plt.xlabel("m_phi")
    plt.ylabel("k_rot")
    plt.title("Mapa 2D de probabilidad de sincronía  P_A(m_phi, k_rot)")
    plt.grid(False)
    plt.tight_layout()

    plt.savefig(outfile, dpi=200)
    print(f"[INFO] Mapa 2D guardado en {outfile}")


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main():
    rows = read_rows(RESULTS_DIR)
    mphi_vals, krot_vals, PA_grid = build_grid(rows)
    plot_PA_map(mphi_vals, krot_vals, PA_grid, OUTPUT_FIG)


# Ejecutar
if __name__ == "__main__":
    main()
