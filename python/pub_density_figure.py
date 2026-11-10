"""
Publication figure (PRB style) of the bare magnon-phonon scattering rate: linear energy, logarithmic rate, mode density as colour, median line, literature.
Three-phonon scattering is not shown.

    python python/pub_density_figure.py Outputs/convergence/<tag> [--N 40] [--smearing 0.1] [--material bccFe fccNi] [--pol] [--single_column]

Panels: (a) magnons, (b) phonons; with --pol (needs the mesh file of the run) (b) transverse and (c) longitudinal phonons (|e.q^|^2 > 0.5).
Colour = number of modes per (energy, log10 rate) cell (viridis, logarithmic scale, empty cells white), black line = median in energy bins,
open diamonds = literature (bccFe). Output: <scan>/gallery/<material>_N<N>_s<smearing>_prb_density[_pol].{pdf,png}
"""
import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, NullFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conv_common as cc

ap = argparse.ArgumentParser()
ap.add_argument("scan")
ap.add_argument("--N", type=int, default=40)
ap.add_argument("--smearing", type=float, default=0.1)
ap.add_argument("--material", nargs="*", default=None)
ap.add_argument("--pol", action="store_true")
ap.add_argument("--single_column", action="store_true", help="3.4 in wide stacked version (panels above each other)")
ap.add_argument("--bins_x", type=int, default=70)
ap.add_argument("--bins_y", type=int, default=60)
a = ap.parse_args()

# PRB: 3.4 in (one column) / 7.0 in (two columns), text >= 8 pt at final size, sans-serif, ticks inward on all sides, no title, panel letters
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
                     "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "xtick.minor.visible": True,
                     "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42})
GAMMA = r"$\gamma$ (ps$^{-1}$)"
man = cc.load_scan(a.scan)
os.makedirs(os.path.join(a.scan, "gallery"), exist_ok=True)


def median_line(E, g, nb=22):
    edges = np.linspace(E.min(), E.max(), nb + 1)
    idx = np.clip(np.digitize(E, edges) - 1, 0, nb - 1)
    c, m = [], []
    for i in range(nb):
        v = g[idx == i]
        if len(v) >= 5:
            c.append(0.5 * (edges[i] + edges[i + 1])); m.append(np.median(v))
    return np.array(c), np.array(m)


def panel(ax, E, g, lit, letter, label, ylim, xlim):
    ok = (g > 1e-12) & (E > 0)
    E, g = E[ok], g[ok]
    xe = np.linspace(xlim[0], xlim[1], a.bins_x + 1)
    ye = np.linspace(np.log10(ylim[0]), np.log10(ylim[1]), a.bins_y + 1)
    H, _, _ = np.histogram2d(E, np.log10(g), bins=[xe, ye])
    H = np.ma.masked_equal(H.T, 0)
    pc = ax.pcolormesh(xe, 10 ** ye, H, norm=LogNorm(vmin=1, vmax=max(10, H.max())), cmap="viridis", rasterized=True, shading="flat")
    c, m = median_line(E, g)
    ax.plot(c, m, color="k", lw=1.0, zorder=4, label="median")
    if lit is not None and len(lit):
        ax.scatter(lit.E, lit.g, marker="D", s=9, facecolor="white", edgecolor="k", linewidth=0.6, zorder=5, label="literature")
    ax.set_yscale("log"); ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(LogLocator(numticks=12)); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("energy (meV)"); ax.set_ylabel(GAMMA)
    ax.text(0.04, 0.95, f"({letter}) {label}", transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.85))
    return pc


def decades(g, lo_q=0.05, hi_q=99.95):
    g = g[g > 1e-12]
    return 10 ** np.floor(np.log10(np.percentile(g, lo_q))), 10 ** np.ceil(np.log10(np.percentile(g, hi_q)))


for mat in (a.material or sorted(man.material.unique())):
    r = man[(man.material == mat) & (man.N == a.N) & np.isclose(man.smearing, a.smearing) & man.ok]
    if r.empty:
        print(f"{mat}: no finished run"); continue
    d = cc.load_run(r.iloc[0].path)
    lit = cc.literature(mat)
    mg, ph = d[d.particle == "magnon"], d[d.particle == "phonon"].copy()
    groups = [(mg, lit.get("magnon"), "magnons")]
    pol = a.pol
    if pol:
        import phph
        mp = os.path.join(cc.ROOT, r.iloc[0].mesh)
        if not os.path.exists(mp):
            print(f"{mat}: mesh file {mp} missing -> no polarisation split"); pol = False
    if pol:
        h = phph.harmonic_from_h5(mp)
        fold = h.q_int / h.mesh; fold = fold - np.round(fold)
        q = (2 * np.pi * fold @ h.B)[ph.q_idx.to_numpy(int)]
        qn = np.linalg.norm(q, axis=1)
        e = h.eig[ph.q_idx.to_numpy(int), ph.branch.to_numpy(int)].real
        fl = np.where(qn > 1e-9, (e * q).sum(1) ** 2 / np.maximum(qn, 1e-12) ** 2, np.nan)
        groups += [(ph[fl <= 0.5], lit.get("phonon"), "transverse phonons"), (ph[fl > 0.5], lit.get("phonon"), "longitudinal phonons")]
    else:
        groups.append((ph, lit.get("phonon"), "phonons"))
    n = len(groups)
    if a.single_column:
        fig, axs = plt.subplots(n, 1, figsize=(3.4, 2.15 * n + 0.1), squeeze=False)
        axs = axs[:, 0]
    else:
        fig, axs = plt.subplots(1, n, figsize=(7.0, 2.55), squeeze=False)
        axs = axs[0]
    ylim_m = decades(mg.gamma_mp.to_numpy())
    ylim_p = decades(ph.gamma_mp.to_numpy())
    pcs = []
    for k, (dd, L, label) in enumerate(groups):
        mag = k == 0
        xlim = (0, 10 * np.ceil(1.02 * dd.energy_meV.max() / 10)) if mag else (0, 5 * np.ceil(1.02 * ph.energy_meV.max() / 5))
        pcs.append(panel(axs[k], dd.energy_meV.to_numpy(), dd.gamma_mp.to_numpy(), L, "abc"[k], label, ylim_m if mag else ylim_p, xlim))
        if not mag and k > 1:
            axs[k].set_ylabel("")                                   # shared y axis of the phonon panels
            axs[k].tick_params(labelleft=False)
    h_, l_ = axs[0].get_legend_handles_labels()
    if lit.get("magnon") is not None or lit.get("phonon") is not None:
        axs[-1].legend(loc="lower right", frameon=True, framealpha=0.85, edgecolor="none", borderpad=0.3, handlelength=1.2)
    else:
        axs[-1].legend(loc="lower right", frameon=True, framealpha=0.85, edgecolor="none", borderpad=0.3, handlelength=1.2)
    fig.tight_layout(w_pad=0.6, h_pad=0.6, rect=[0, 0, 0.93 if not a.single_column else 0.88, 1])
    cax = fig.add_axes([0.945, 0.2, 0.012, 0.65]) if not a.single_column else fig.add_axes([0.9, 0.12, 0.025, 0.76])
    cb = fig.colorbar(pcs[0], cax=cax)
    cb.set_label("modes per cell", fontsize=8); cb.ax.tick_params(labelsize=7, direction="in")
    base = os.path.join(a.scan, "gallery", f"{mat}_N{a.N}_s{a.smearing:g}_prb_density" + ("_pol" if pol else "") + ("_1col" if a.single_column else ""))
    fig.savefig(base + ".pdf"); fig.savefig(base + ".png", dpi=400); plt.close(fig)
    print("wrote", base + ".png")
