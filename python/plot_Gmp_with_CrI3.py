"""
G_mp(T) of CrI3 (from full-BZ hybrid lifetime files) together with bcc Fe and fcc Ni, log-log.

    python python/plot_Gmp_with_CrI3.py [--cri3 DIR [DIR ...]] [--sigma_spread DIR [DIR ...]] [--tscan FOLDER[:material] ...]
                                        [--tmax 800] [--out_dir Outputs/paper]

CrI3: every DIR holds a full-BZ hybrid_path_lifetimes.csv (12^3 q-points, 26 hybrid modes); the temperature of the run is read from the
occupations (n_occ column; T = E / (k_B ln(1 + 1/n))), the rate is the rate of the hybrid mode and the phonon weight its phonon character:
    G_mp = (1/N_q) sum_{q,modes} w_ph E gamma dn_BE/dT ,   w_ph = phon_character
(same convention as python/plot-hybrid-lifetimesCrI3.py), SI via the rhombohedral primitive cell V = 281.794 A^3 (8 atoms).
Default CrI3 set = the three runs with the same settings (hybrid, full BZ, sigma = 0.2 meV): T = 15, 25, 40 K.
--sigma_spread: further 15 K runs with other broadenings (open markers) to show how much G_mp depends on sigma at fixed T.
Fe / Ni as in plot_Gmp_rel_paper.py (N = 40, sigma = 0.2 meV, T <= T_C and <= --tmax). Output gmp_vs_T_with_CrI3.png/pdf gmp_vs_T_over_Tc.png/pdf (the same against T/T_C; T_C = 1043 / 627 / 61 K for Fe / Ni / CrI3) and gmp_CrI3.csv.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_Gmp_rel_paper as P

KB = 0.08617333262
V_CRI3 = 281.794                 # A^3, rhombohedral primitive cell, 8 atoms
TC_CRI3 = 61.0                   # K, bulk CrI3
COL_CRI3 = "#2ca02c"
ROOT = "Outputs"
DEF_CRI3 = [f"{ROOT}/CrI3_Path_Hyrbid_Full_sig_0.2_15K", f"{ROOT}/CrI3_Path_Hyrbid_Full_sig_0.2_25K", f"{ROOT}/CrI3_Path_Hyrbid_Full_sig_0.2"]
DEF_SPREAD = [f"{ROOT}/CrI3_Path_Full_Hyrbid_sig_0.01_15K", f"{ROOT}/CrI3_Full_Hyrbid_sig_1.0_15K", f"{ROOT}/CrI3_Full_Hyrbid_sig_2.0_15K"]


def cri3_point(d):
    df = pd.read_csv(os.path.join(d, "hybrid_path_lifetimes.csv"), comment="#")
    s = df[(df.energy_meV > 0.5) & (df.n_occ > 1e-8)]
    T = float(np.median(s.energy_meV / (KB * np.log1p(1.0 / s.n_occ))))
    E = df.energy_meV.to_numpy(float)
    w = np.clip(df.phon_character.to_numpy(float), 0.0, 1.0)
    g = np.nan_to_num(pd.to_numeric(df["gamma_ps-1"], errors="coerce").to_numpy(float))
    k = E > 0
    G = np.sum(w[k] * E[k] * g[k] * P.dn_dT(E[k], T)) / df.q_idx.nunique()
    return dict(run=os.path.basename(d), T=T, nq=int(df.q_idx.nunique()), G_mp=G * P.MEV_PER_PS_TO_W / (V_CRI3 * P.ANG3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cri3", nargs="+", default=DEF_CRI3)
    ap.add_argument("--sigma_spread", nargs="*", default=DEF_SPREAD)
    ap.add_argument("--tscan", nargs="+", default=["Outputs/convergence_rel/Tscan_rel_20261112:bccFe", "Outputs/convergence_rel/Tscan_rel_NEW:fccNi",
                                                    "Outputs/convergence_rel/Tscan_rel_low"])
    ap.add_argument("--tscan_N", type=int, default=40)
    ap.add_argument("--tscan_s", type=float, default=0.2)
    ap.add_argument("--tmax", type=float, default=800.0)
    ap.add_argument("--out_dir", default="Outputs/paper")
    a = ap.parse_args()
    P.style()
    main_pts = pd.DataFrame([cri3_point(d) for d in a.cri3]).sort_values("T")
    spread = pd.DataFrame([cri3_point(d) for d in a.sigma_spread]) if a.sigma_spread else pd.DataFrame(columns=["T", "G_mp", "run"])
    print("CrI3 (hybrid, full BZ):\n", main_pts.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    if len(spread):
        print("sigma spread at fixed T:\n", spread.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    tab = P.load_scan(a.tscan)
    tab = tab[(tab.N == a.tscan_N) & (abs(tab.smearing - a.tscan_s) < 1e-9) & (tab["T"] <= a.tmax + 1e-6) & np.isfinite(tab.G_mp) & (tab.G_mp > 0)]

    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    for m in ("bccFe", "fccNi"):
        d = tab[tab.material == m].sort_values("T")
        d = d[d["T"] <= P.TC[m] + 1e-6]
        ax.plot(d["T"], d.G_mp, "o-", color=P.COL[m], label=P.NAME[m])
    ax.plot(main_pts["T"], main_pts.G_mp, "s-", color=COL_CRI3, label=r"CrI$_3$")
    if len(spread):
        ax.plot(spread["T"], spread.G_mp, "s", mfc="none", mec=COL_CRI3, ms=4, lw=0, label=r"CrI$_3$, other $\sigma$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([10, 20, 50, 100, 200, 500]); ax.set_xticklabels(["10", "20", "50", "100", "200", "500"]); ax.minorticks_off(); ax.tick_params(axis="y", which="minor", length=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"$T$ (K)"); ax.set_ylabel(r"$G_{mp}$ (W m$^{-3}$ K$^{-1}$)")
    ax.legend(frameon=False, loc="lower right", fontsize=6.5)
    fig.tight_layout()
    P.save(fig, a.out_dir, "gmp_vs_T_with_CrI3")
    # ---- reduced temperature T / T_C
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    for m in ("bccFe", "fccNi"):
        d = tab[tab.material == m].sort_values("T")
        d = d[d["T"] <= P.TC[m] + 1e-6]
        ax.plot(d["T"] / P.TC[m], d.G_mp, "o-", color=P.COL[m], label=P.NAME[m] + rf" ($T_C={P.TC[m]:g}$ K)")
    ax.plot(main_pts["T"] / TC_CRI3, main_pts.G_mp, "s-", color=COL_CRI3, label=rf"CrI$_3$ ($T_C={TC_CRI3:g}$ K)")
    if len(spread):
        ax.plot(spread["T"] / TC_CRI3, spread.G_mp, "s", mfc="none", mec=COL_CRI3, ms=4, lw=0, label=r"CrI$_3$, other $\sigma$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5, 1.0]); ax.set_xticklabels(["0.02", "0.05", "0.1", "0.2", "0.5", "1"]); ax.minorticks_off()
    ax.set_xlabel(r"$T/T_C$"); ax.set_ylabel(r"$G_{mp}$ (W m$^{-3}$ K$^{-1}$)")
    ax.legend(frameon=False, loc="lower right", fontsize=6)
    fig.tight_layout()
    P.save(fig, a.out_dir, "gmp_vs_T_over_Tc")
    out = pd.concat([main_pts.assign(sigma_set="main (sigma=0.2 meV)"), spread.assign(sigma_set="other sigma, 15 K")])
    out.to_csv(os.path.join(a.out_dir, "gmp_CrI3.csv"), index=False)
    print(f" -> {os.path.join(a.out_dir, 'gmp_CrI3.csv')}")


if __name__ == "__main__":
    main()
