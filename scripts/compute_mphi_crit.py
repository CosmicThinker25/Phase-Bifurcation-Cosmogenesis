#!/usr/bin/env python3
import os
import glob
import json
import math
import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results_phase_sectors")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "mphi_crit_summary.json")
OUTPUT_FIG = os.path.join(RESULTS_DIR, "PA_vs_mphi.png")

TARGET_PA = 0.5   # Queremos localizar P_A = 0.5


# ============================================================
# FUNCIÓN 1 — LECTURA ROBUSTA DE TODOS LOS CSV
# Ignora automáticamente CSV que NO tengan N_total o P_A
# ============================================================

def read_sector_csvs(results_dir):
    pattern = os.path.join(results_dir, "*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No se encontraron CSV en {results_dir}")

    rows = []
    total_files = len(files)

    print(f"[INFO] Encontrados {total_files} CSV, empezando lectura...")

    for idx, fname in enumerate(files, start=1):
        print(f"  -> ({idx}/{total_files}) Leyendo {os.path.basename(fname)}")

        with open(fname, "r", newline="") as f:
            reader = csv.DictReader(f)

            for r in reader:

                # m_phi y k_rot son imprescindibles
                try:
                    m_phi = float(r.get("m_phi"))
                    k_rot = float(r.get("k_rot", 0.0))
                except:
                    continue  # fila inválida

                # Caso 1: Si el CSV trae P_A directamente, perfecto
                if "P_A" in r and r["P_A"]:
                    try:
                        P_A = float(r["P_A"])
                        rows.append({"m_phi": m_phi, "k_rot": k_rot, "P_A": P_A})
                    except:
                        pass
                    continue

                # Caso 2: Intentar calcular P_A mediante N_A / N_total
                N_total_raw = r.get("N_total")
                N_A_raw = r.get("N_A")

                if not N_total_raw or not N_A_raw:
                    continue  # CSV tipo boundary_* → ignorar

                try:
                    N_total = float(N_total_raw)
                    N_A = float(N_A_raw)
                except:
                    continue

                if N_total > 0:
                    P_A = N_A / N_total
                    rows.append({"m_phi": m_phi, "k_rot": k_rot, "P_A": P_A})

    print(f"[INFO] Filas válidas para análisis: {len(rows)}")
    return rows


# ============================================================
# FUNCIÓN 2 — PROMEDIOS P_A(m_phi)
# ============================================================

def compute_PA_vs_mphi(rows):
    groups = defaultdict(list)

    for r in rows:
        P_A = r["P_A"]
        if not math.isnan(P_A):
            groups[r["m_phi"]].append(P_A)

    mphi_vals = sorted(groups.keys())
    PA_means = []
    PA_stds = []

    print(f"[INFO] Calculando promedios para {len(mphi_vals)} valores de m_phi...")

    for mphi in mphi_vals:
        arr = np.array(groups[mphi])
        PA_means.append(arr.mean())
        PA_stds.append(arr.std(ddof=1) if len(arr) > 1 else 0.0)

    return np.array(mphi_vals), np.array(PA_means), np.array(PA_stds)


# ============================================================
# FUNCIÓN 3 — ENCONTRAR m_phi_crit (P_A = 0.5)
# ============================================================

def find_mphi_crit(mphi_vals, PA_means, target=0.5):
    idx_sort = np.argsort(mphi_vals)
    mphi = mphi_vals[idx_sort]
    PA = PA_means[idx_sort]

    diff = PA - target
    sign = np.sign(diff)

    for i in range(len(mphi) - 1):
        if sign[i] * sign[i+1] < 0:  # cruce exacto
            m1, m2 = mphi[i], mphi[i+1]
            p1, p2 = PA[i], PA[i+1]

            # interpolación lineal
            if p2 != p1:
                frac = (target - p1) / (p2 - p1)
                return m1 + frac * (m2 - m1)

    return None


# ============================================================
# FUNCIÓN 4 — GENERAR FIGURA
# ============================================================

def make_plot(mphi_vals, PA_means, PA_stds, mphi_crit, outfile):
    plt.figure()
    plt.errorbar(mphi_vals, PA_means, yerr=PA_stds, marker="o", capsize=4)

    plt.axhline(0.5, linestyle="--", color="gray")
    if mphi_crit:
        plt.axvline(mphi_crit, linestyle=":", color="red")

    plt.xlabel("m_phi")
    plt.ylabel("P_A")
    plt.title("Probabilidad de sincronía P_A vs m_phi")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)

    print(f"[INFO] Figura guardada en {outfile}")


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main():
    rows = read_sector_csvs(RESULTS_DIR)
    mphi_vals, PA_means, PA_stds = compute_PA_vs_mphi(rows)
    mphi_crit = find_mphi_crit(mphi_vals, PA_means, TARGET_PA)

    if mphi_crit:
        print(f"\n[RESULT] m_phi_crit ≈ {mphi_crit:.6f}\n")
    else:
        print("\n[RESULT] No se encontró cruce P_A = 0.5 (no hay transición clara)\n")

    # Guardar JSON
    summary = {
        "m_phi_vals": mphi_vals.tolist(),
        "PA_means": PA_means.tolist(),
        "PA_stds": PA_stds.tolist(),
        "target_PA": TARGET_PA,
        "m_phi_crit": mphi_crit,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Resumen guardado en {OUTPUT_JSON}")

    # Figura
    make_plot(mphi_vals, PA_means, PA_stds, mphi_crit, OUTPUT_FIG)


# Ejecutar
if __name__ == "__main__":
    main()
