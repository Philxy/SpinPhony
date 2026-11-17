import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Magnon-phonon coupling G_mp in SI units versus temperature normalised by the Curie temperature.
#
#   G [W m^-3 K^-1] = G_cell [meV K^-1 ps^-1 per cell] * (1.602176634e-10 W per meV/ps) / V_cell [m^3]
#
# Input: G_mp_temperature_scan.csv written by SpinPhony.py (columns Temperature_K,
# G_mp_meV_per_K_ps_per_cell, ...). Default curves: CrI3 (T_C = 61 K, bulk rhombohedral primitive cell
# 281.794 A^3) and fccNi (T_C = 627 K, fcc primitive cell a^3/4, a = 3.516822176 A). Other curves can be given as
#   --curve LABEL:CSV:T_C_in_K:CELL_VOLUME_in_A3     (repeatable; replaces the defaults)

MEV_PER_PS_TO_W = 1.602176634e-10
ANG3_TO_M3 = 1e-30
A_NI = 3.516822176

DEFAULT_CURVES = [
    ("CrI3", "Outputs/CrI3_reprodizierbar./G_mp_temperature_scan.csv", 61.0, 281.794),
    ("fccNi", "Outputs/fccNi_reprodizierbar/G_mp_temperature_scan.csv", 627.0, A_NI ** 3 / 4.0),
]


def load_curve(label, path, tc, vol_A3):
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found")
    df = pd.read_csv(path)
    G_cell = df["G_mp_meV_per_K_ps_per_cell"].to_numpy(float)
    out = pd.DataFrame({
        "material": label,
        "T_K": df["Temperature_K"].to_numpy(float),
        "T_over_Tc": df["Temperature_K"].to_numpy(float) / tc,
        "G_meV_per_K_ps_per_cell": G_cell,
        "G_W_m3K": G_cell * MEV_PER_PS_TO_W / (vol_A3 * ANG3_TO_M3),
        "T_C_K": tc,
        "V_cell_A3": vol_A3,
    })
    return out


def main():
    ap = argparse.ArgumentParser(description="G_mp in SI units versus T / T_Curie")
    ap.add_argument("--curve", action="append", default=None, metavar="LABEL:CSV:TC:V",
                    help="material curve; repeatable. Default: CrI3 and fccNi from Outputs/*_reprodizierbar*/")
    ap.add_argument("--out", default="Gmp_SI_vs_TTc.png")
    ap.add_argument("--linear_x", action="store_true", help="linear T/T_C axis (default: logarithmic)")
    ap.add_argument("--linear_y", action="store_true", help="linear G axis (default: logarithmic)")
    ap.add_argument("--gmin", type=float, default=1e6,
                    help="do not plot points below this G (W m^-3 K^-1); the exponentially small low-T tail "
                         "otherwise squeezes the interesting range (default 1e6; the CSV keeps all points)")
    ap.add_argument("--above_tc", action="store_true", help="also plot points above T_C (default: T <= T_C only)")
    ap.add_argument("--exp", default="Inputs/exp/Ca9La5Cu24O41_Gmp.csv",
                    help="experimental G_mp points (columns T_K, G); ladder compound Ca9La5Cu24O41. 'none' to skip")
    ap.add_argument("--exp_scale", type=float, default=1e15,
                    help="multiplier taking the experimental G column to W m^-3 K^-1 (1e15, as in the earlier plotting script)")
    ap.add_argument("--exp_tref", type=float, default=None,
                    help="divide the experimental T by this reference (K) and plot on the T/T_C axis; "
                         "default: no normalisation, drawn against its own top axis in K (the ladder has no T_C)")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    if a.curve:
        curves = []
        for text in a.curve:
            label, rest = text.split(":", 1)
            path, tc, vol = rest.rsplit(":", 2)
            curves.append((label, path, float(tc), float(vol)))
    else:
        curves = DEFAULT_CURVES

    frames = [load_curve(*c) for c in curves]
    table = pd.concat(frames, ignore_index=True)
    csv_out = os.path.splitext(a.out)[0] + ".csv"
    table.to_csv(csv_out, index=False)
    print(f" -> {csv_out}")
    for f in frames:
        pos = f[f.G_W_m3K > 0]
        i = pos.G_W_m3K.idxmax()
        print(f"{f.material.iloc[0]:6s} T_C = {f.T_C_K.iloc[0]:.0f} K, V = {f.V_cell_A3.iloc[0]:.3f} A^3: "
              f"G_max = {pos.G_W_m3K.max():.3e} W m^-3 K^-1 at T/T_C = {pos.T_over_Tc.loc[i]:.2f}")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for f in frames:
        d = f[(f.G_W_m3K > a.gmin) & np.isfinite(f.G_W_m3K)]
        if not a.above_tc:
            d = d[d.T_over_Tc <= 1.0]
        ax.plot(d.T_over_Tc, d.G_W_m3K, "o-", lw=1.8, ms=5, label=f"{f.material.iloc[0]} ($T_C$ = {f.T_C_K.iloc[0]:.0f} K)")
    if a.above_tc:
        ax.axvline(1.0, color="0.5", ls="--", lw=1)
    else:
        ax.set_xlim(right=1.05)
    if a.exp and a.exp.lower() != "none" and os.path.exists(a.exp):
        e = pd.read_csv(a.exp, skipinitialspace=True)
        Ge = e["G"].to_numpy(float) * a.exp_scale
        lab = r"exp. Ca$_9$La$_5$Cu$_{24}$O$_{41}$ (G $\times$ %.0e)" % a.exp_scale
        if a.exp_tref:
            ax.plot(e["T_K"] / a.exp_tref, Ge, "k^", ms=6, mfc="none", label=lab + fr", $T/{a.exp_tref:g}$ K")
        else:
            axt = ax.twiny()
            axt.plot(e["T_K"], Ge, "k^", ms=6, mfc="none", label=lab)
            axt.set_xscale("log" if not a.linear_x else "linear")
            axt.set_xlabel("T (K), experiment (top axis)", fontsize=11)
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = axt.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="lower right")
            legend_done = True
        exp_out = os.path.splitext(a.out)[0] + "_exp.csv"
        pd.DataFrame({"T_K": e["T_K"], "G_W_m3K": Ge}).to_csv(exp_out, index=False)
    ax.set_xlabel(r"$T / T_C$", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"$G_{mp}$ (W m$^{-3}$ K$^{-1}$)", fontsize=12, fontweight="bold")
    if not a.linear_x:
        ax.set_xscale("log")
    if not a.linear_y:
        ax.set_yscale("log")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    if "legend_done" not in dir():
        ax.legend(fontsize=10)
    ax.set_title(r"Magnon-phonon coupling versus reduced temperature", fontsize=13)
    fig.tight_layout()
    fig.savefig(a.out, dpi=300)
    print(f" -> {a.out}")
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
