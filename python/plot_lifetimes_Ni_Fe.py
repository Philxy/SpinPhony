"""
Scattering rate vs energy: bcc Fe (left) and fcc Ni (right), magnons and phonons together in each panel, shared y axis. Low alpha so the point density shows. E^2 guide line in the magnon panels.

    python python/plot_lifetimes_Ni_Fe.py [--fe DIR] [--ni DIR] [--E0 50] [--out Outputs/lifetimes_Ni_Fe.png]

--fe / --ni are run folders holding path_lifetimes.csv (T = 300 K runs).
Plotted rate: gamma_exch_ps-1 (non-relativistic, magnon-number-conserving magnon-phonon rate) if the file has it
(runs with --relativistic), otherwise gamma_ps-1, which then contains no relativistic part.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--fe", default="Outputs/compare_newSLC/old/Tscan_20261110/bccFe_N40_s0.2_T300K")
ap.add_argument("--ni", default="Outputs/compare_newSLC/new/fccNi_N40_s0.2")
ap.add_argument("--E0", type=float, default=50.0, help="meV, anchor of the E^2 line (median magnon rate within +-10%)")
ap.add_argument("--alpha", type=float, default=0.08)
ap.add_argument("--out", default="Outputs/lifetimes_Ni_Fe.png")
a = ap.parse_args()

rows = (("bcc Fe", a.fe), ("fcc Ni (new SLC)", a.ni))
fig, axs = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
for ax, (name, d) in zip(axs, rows):
    df = pd.read_csv(os.path.join(d, "path_lifetimes.csv"), comment="#")
    # only the non-relativistic, magnon-number-conserving magnon-phonon rate (exchange-striction vertex V^{+-}):
    # with --relativistic the file also holds gamma_exch_ps-1 (conserving part) next to the total gamma_ps-1
    col_g = "gamma_exch_ps-1" if "gamma_exch_ps-1" in df.columns else "gamma_ps-1"
    print(f"{name}: using column {col_g}" + ("" if col_g == "gamma_exch_ps-1" else "  (no relativistic channels in this file, so it is already the conserving rate)"))
    g = pd.to_numeric(df[col_g], errors="coerce")
    ok = np.isfinite(g) & (g > 0) & (df.energy_meV > 0.2)
    for part, mk, col in (("phonon", "o", "#1f77b4"), ("magnon", "^", "#d62728")):
        s = ok & (df.particle == part)
        ax.scatter(df.energy_meV[s], g[s], marker=mk, s=6, color=col, alpha=a.alpha, linewidths=0, rasterized=True)
        ax.scatter([], [], marker=mk, s=30, color=col, label=part + "s")  # opaque legend handle
        if part == "magnon":
            m = s & (abs(df.energy_meV - a.E0) < 0.1 * a.E0)
            if m.any():
                g0 = float(np.median(g[m])); E = np.linspace(a.E0, df.energy_meV[s].max(), 100)
                ax.plot(E, g0 * (E / a.E0) ** 2, "k--", lw=2, label=rf"$\propto E^2$ (from {a.E0:g} meV)")
    ax.set_yscale("log"); #ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.set_xscale("log")
    ax.set_title(name, fontsize=12); ax.set_xlabel("Energy (meV)"); ax.legend(fontsize=9, loc="upper left")
axs[0].set_ylabel(r"Scattering rate $\gamma$ (ps$^{-1}$)")
fig.tight_layout()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
fig.savefig(a.out, dpi=250)
print(f" -> {a.out}")
