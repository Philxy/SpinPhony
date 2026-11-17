"""
G_mp of a convergence scan: G_mp vs grid size N (one curve per broadening) and vs broadening (one curve per N), with the literature value.

    python python/convergence_Gmp.py Outputs/convergence/<tag> [--material bccFe fccNi] [--T 300]

G_mp = (1/N_q) sum_{q,lambda} E gamma_mp dn_BE/dT |_T over all phonon modes of the full-BZ run (bare phonons: weight 1), gamma_mp the magnon-phonon
channel only (the three-phonon rate of bccFe is not included). Units meV/(ps K)/cell and W m^-3 K^-1 (cell volumes in conv_common.CELL_VOLUME).
Writes <scan>/convergence_Gmp.csv and <scan>/figures/Gmp_convergence_<material>.png. The literature value (bccFe: plot-hybrid-lifetimesbccFe.py) is
the arithmetic bin average of the literature phonon rates on the run's phonon energies; fccNi has none unless Inputs/fccNi/lit_phonon_rates.csv exists.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conv_common as cc

ap = argparse.ArgumentParser()
ap.add_argument("scan")
ap.add_argument("--material", nargs="*", default=None)
ap.add_argument("--T", type=float, default=None, help="temperature of G_mp (default: the run's)")
ap.add_argument("--no_show", action="store_true")
a = ap.parse_args()
if a.no_show:
    matplotlib.use("Agg")
os.makedirs(os.path.join(a.scan, "figures"), exist_ok=True)
man = cc.load_scan(a.scan)
rows = []
for _, r in man[man.ok].iterrows():
    lit = cc.literature(r.material)["phonon"]
    G_cell, G, Gl = cc.gmp_of_run(cc.load_run(r.path), r.material, a.T if a.T else r["T"], lit)
    rows.append(dict(run=r.run, material=r.material, N=r.N, smearing=r.smearing, T=a.T if a.T else r["T"], G_meV_ps_K_cell=G_cell, G_W_m3_K=G, lit_G_W_m3_K=Gl))
res = pd.DataFrame(rows).sort_values(["material", "smearing", "N"])
res.to_csv(os.path.join(a.scan, "convergence_Gmp.csv"), index=False)
print(res.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

for mat in (a.material or sorted(res.material.unique())):
    r = res[res.material == mat]
    if r.empty:
        continue
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    cm = plt.get_cmap("viridis")
    Ss, Ns = sorted(r.smearing.unique()), sorted(r.N.unique())
    for j, s in enumerate(Ss):
        q = r[np.isclose(r.smearing, s)].sort_values("N")
        ax[0].plot(q.N, q.G_W_m3_K, "o-", color=cm(j / max(1, len(Ss) - 1)), label=f"smearing {s:g} meV")
    for j, n in enumerate(Ns):
        q = r[r.N == n].sort_values("smearing")
        ax[1].plot(q.smearing, q.G_W_m3_K, "o-", color=cm(j / max(1, len(Ns) - 1)), label=f"N = {n}$^3$")
    gl = r.lit_G_W_m3_K.dropna()
    if len(gl):
        for a_ in ax:
            a_.axhline(gl.median(), color="C3", ls="--", lw=1.4, label="literature (arith.)")
    ax[0].set_xlabel("grid points per direction N"); ax[0].set_xticks(Ns)
    ax[1].set_xlabel("Gaussian broadening factor (meV)"); ax[1].set_xscale("log")
    for a_ in ax:
        a_.set_ylabel(r"$G_{mp}$ (W m$^{-3}$ K$^{-1}$)"); a_.set_yscale("log"); a_.grid(alpha=0.3, which="both"); a_.legend(fontsize=8)
    ax[0].set_title(f"{mat}: G_mp vs N"); ax[1].set_title(f"{mat}: G_mp vs broadening")
    fig.suptitle(f"{mat}, T = {r['T'].iloc[0]:g} K (magnon-phonon channel)")
    fig.tight_layout(); out = os.path.join(a.scan, "figures", f"Gmp_convergence_{mat}.png"); fig.savefig(out, dpi=130); plt.close(fig)
    print("wrote", out)
