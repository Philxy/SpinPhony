"""
Binned median of the bare magnon-phonon scattering rate (or lifetime) vs energy, phonon branches separated; three-phonon scattering is not drawn.

    python python/median_branches.py Outputs/convergence/<tag> [--N 40] [--smearing 0.1] [--tau] [--linear_x] [--bins 22] [--min_count 5]

With --pol the phonons are grouped as transverse / longitudinal from the eigenvectors of the mesh file (|e.q^|^2 > 0.5 = longitudinal).

Per energy bin (log-spaced, or linear with --linear_x) the median over all modes of a branch is plotted (line), the 10-90 % range of the same modes as band;
bins with fewer than --min_count modes are left out. Magnons: the single branch. Phonons: branch 0, 1, 2 as numbered in the mesh file (sorted by energy at each q).
Literature (bccFe) as diamonds. Output: <scan>/gallery/<material>_N<N>_s<smearing>_median_branches[_tau][_linx].{png,pdf}
"""
import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conv_common as cc

ap = argparse.ArgumentParser()
ap.add_argument("scan")
ap.add_argument("--N", type=int, default=40)
ap.add_argument("--smearing", type=float, default=0.1)
ap.add_argument("--material", nargs="*", default=None)
ap.add_argument("--tau", action="store_true")
ap.add_argument("--linear_x", action="store_true")
ap.add_argument("--pol", action="store_true",
                help="group the phonons by polarisation instead of branch: longitudinal if |e.q^|^2 > 0.5, else transverse (needs the mesh file of the run)")
ap.add_argument("--bins", type=int, default=22)
ap.add_argument("--min_count", type=int, default=5)
a = ap.parse_args()

plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "legend.fontsize": 9, "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True, "axes.linewidth": 0.9})
BR = ["#0072B2", "#D55E00", "#009E73"]                 # Okabe-Ito, phonon branches 0, 1, 2
MAGC = "#CC79A7"
man = cc.load_scan(a.scan)
f = (lambda x: 1.0 / x) if a.tau else (lambda x: x)
ylab = r"$\tau$ (ps)" if a.tau else r"$\gamma$ (ps$^{-1}$)"


def binned(E, g):
    ok = (g > 1e-12) & (E > 0)
    E, g = E[ok], g[ok]
    edges = np.linspace(E.min(), E.max(), a.bins + 1) if a.linear_x else np.geomspace(E.min(), E.max(), a.bins + 1)
    idx = np.clip(np.digitize(E, edges) - 1, 0, a.bins - 1)
    out = []
    for i in range(a.bins):
        v = g[idx == i]
        if len(v) >= a.min_count:
            c = 0.5 * (edges[i] + edges[i + 1]) if a.linear_x else np.sqrt(edges[i] * edges[i + 1])
            out.append((c, np.median(v), np.percentile(v, 10), np.percentile(v, 90)))
    return np.array(out).T if out else np.zeros((4, 0))


for mat in (a.material or sorted(man.material.unique())):
    r = man[(man.material == mat) & (man.N == a.N) & np.isclose(man.smearing, a.smearing) & man.ok]
    if r.empty:
        print(f"{mat}: no finished run"); continue
    d = cc.load_run(r.iloc[0].path)
    lit = cc.literature(mat)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3))
    m = d[d.particle == "magnon"]
    c, med, lo, hi = binned(m.energy_meV.to_numpy(), m.gamma_mp.to_numpy())
    ax[0].fill_between(c, f(lo), f(hi), color=MAGC, alpha=0.2, linewidth=0); ax[0].plot(c, f(med), color=MAGC, lw=1.8, label="magnons")
    ph = d[d.particle == "phonon"].copy()
    if a.pol:
        sys.path.insert(0, os.path.join(cc.ROOT, "python"))
        import phph
        mesh_path = os.path.join(cc.ROOT, r.iloc[0].mesh)
        if not os.path.exists(mesh_path):
            print(f"{mat}: mesh file {mesh_path} not found (copy it from the cluster)"); plt.close(fig); continue
        h = phph.harmonic_from_h5(mesh_path)
        # Cartesian q of the mesh point folded into the first BZ, in the frame of the eigenvectors (the qx,qy,qz columns of the csv are not that)
        fold = h.q_int / h.mesh
        fold = fold - np.round(fold)
        q = (2 * np.pi * fold @ h.B)[ph.q_idx.to_numpy(int)]
        qn = np.linalg.norm(q, axis=1)
        e = h.eig[ph.q_idx.to_numpy(int), ph.branch.to_numpy(int)].real
        fl = np.where(qn > 1e-9, (e * q).sum(1) ** 2 / np.maximum(qn, 1e-12) ** 2, np.nan)
        ph["cls"] = np.where(fl > 0.5, "longitudinal", "transverse")
        print(f"{mat}: longitudinal fraction of all phonon modes {np.nanmean(fl > 0.5):.3f} (ideal 1/3); mean |e.q|^2 per branch "
              + ", ".join(f"{b}: {np.nanmean(fl[ph.branch.to_numpy() == b]):.2f}" for b in range(3)))
        for cls, col in (("transverse", BR[0]), ("longitudinal", BR[1])):
            p = ph[ph.cls == cls]
            c, med, lo, hi = binned(p.energy_meV.to_numpy(), p.gamma_mp.to_numpy())
            ax[1].fill_between(c, f(lo), f(hi), color=col, alpha=0.2, linewidth=0); ax[1].plot(c, f(med), color=col, lw=1.8, label=f"phonons, {cls}")
    else:
      for b in range(int(ph.branch.max()) + 1):
        p = ph[ph.branch == b]
        c, med, lo, hi = binned(p.energy_meV.to_numpy(), p.gamma_mp.to_numpy())
        ax[1].fill_between(c, f(lo), f(hi), color=BR[b], alpha=0.2, linewidth=0); ax[1].plot(c, f(med), color=BR[b], lw=1.8, label=f"phonon branch {b}")
    for k, kind in enumerate(("magnon", "phonon")):
        L = lit.get(kind)
        if L is not None and len(L):
            ax[k].scatter(L.E, f(L.g), marker="D", s=16, facecolor="none", edgecolor="#222222", linewidth=0.9, zorder=5, label="literature")
        ax[k].set_xscale("linear" if a.linear_x else "log"); ax[k].set_yscale("log"); ax[k].grid(alpha=0.25)
        if not a.linear_x:
            ax[k].xaxis.set_minor_formatter(NullFormatter())
        ax[k].yaxis.set_minor_formatter(NullFormatter())
        ax[k].set_xlabel("energy (meV)"); ax[k].set_ylabel(ylab)
        ax[k].text(0.03, 0.93, f"({'ab'[k]})", transform=ax[k].transAxes, va="top")
    h0, l0 = ax[0].get_legend_handles_labels(); h1, l1 = ax[1].get_legend_handles_labels()
    seen = dict(zip(l0 + l1, h0 + h1))
    fig.legend(list(seen.values()), list(seen.keys()), loc="upper center", ncol=len(seen), frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    base = os.path.join(a.scan, "gallery", f"{mat}_N{a.N}_s{a.smearing:g}_median_branches" + ("_pol" if a.pol else "") + ("_tau" if a.tau else "") + ("_linx" if a.linear_x else ""))
    os.makedirs(os.path.dirname(base), exist_ok=True)
    fig.savefig(base + ".png", dpi=220); fig.savefig(base + ".pdf"); plt.close(fig)
    print("wrote", base + ".png")
