"""
Bare lifetimes of a convergence scan (Submit_convergence/): one figure per run, overviews of the N-scan and of the broadening scan,
and median curves, all with the literature reference.

    python python/convergence_lifetimes.py Outputs/convergence/<tag> [--ref_smearing 0.1] [--ref_N 20] [--rate] [--material bccFe]

Per run (<scan>/figures/<run>_lifetimes.png): magnons (left) and phonons (right), tau = 1/gamma vs energy, log-log; magnon-phonon channel
(blue), three-phonon channel (green, bccFe), literature (diamonds; fccNi only if Inputs/fccNi/lit_*_rates.csv exist).
Overviews: <scan>/figures/overview_Nscan_<material>.png (columns: N at the reference broadening), overview_smearscan_<material>.png (columns: broadening at the
reference N), median_<material>.png (binned median tau for every setting).
"""
import argparse
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conv_common as cc

ap = argparse.ArgumentParser()
ap.add_argument("scan")
ap.add_argument("--material", nargs="*", default=None)
ap.add_argument("--ref_smearing", type=float, default=None, help="broadening of the N-scan overview (default: the available value closest to 0.1)")
ap.add_argument("--ref_N", type=int, default=None, help="grid of the broadening-scan overview (default: the largest N with all broadenings finished)")
ap.add_argument("--rate", action="store_true", help="plot the scattering rate instead of the lifetime")
ap.add_argument("--no_show", action="store_true")
a = ap.parse_args()
if a.no_show:
    matplotlib.use("Agg")
figdir = os.path.join(a.scan, "figures")
os.makedirs(figdir, exist_ok=True)
man = cc.load_scan(a.scan)
print(man[["run", "status", "ok"]].to_string(index=False))
Y = (lambda g: g) if a.rate else (lambda g: 1.0 / g)
ylab = r"scattering rate $\gamma$ (ps$^{-1}$)" if a.rate else r"lifetime $\tau$ (ps)"
cache = {}


def run_df(r):
    if r.run not in cache:
        cache[r.run] = cc.load_run(r.path)
    return cache[r.run]


def draw(ax, d, kind, lit, label=True):
    s = d[d.particle == kind]
    for col, c, lab in (("gamma_mp", cc.COLORS["mp"], "magnon-phonon"), ("gamma_phph", cc.COLORS["phph"], "phonon-phonon")):
        m = s[col] > 1e-12
        if m.any():
            ax.scatter(s.energy_meV[m], Y(s[col][m]), s=4, color=c, alpha=0.35, rasterized=True, label=f"{lab} ({int(m.sum())})" if label else None)
    L = lit.get(kind)
    if L is not None and len(L):
        ax.scatter(L.E, Y(L.g), marker="D", s=22, facecolor="none", edgecolor=cc.COLORS["lit_mag" if kind == "magnon" else "lit_phon"], linewidth=1.2,
                   zorder=5, label=f"literature ({len(L)})" if label else None)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")


for mat in (a.material or sorted(man.material.unique())):
    mm = man[(man.material == mat) & man.ok].copy()
    if mm.empty:
        print(f"{mat}: no finished runs"); continue
    lit = cc.literature(mat)
    # ---- one figure per run
    for _, r in mm.iterrows():
        d = run_df(r)
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
        for k, kind in enumerate(("magnon", "phonon")):
            draw(ax[k], d, kind, lit)
            ax[k].set_title(f"{kind}s"); ax[k].set_xlabel("energy (meV)"); ax[k].set_ylabel(ylab); ax[k].legend(fontsize=8, markerscale=3)
        fig.suptitle(f"{mat}  N = {r.N}^3  smearing {r.smearing:g} meV  T = {r['T']:g} K")
        fig.tight_layout(); fig.savefig(os.path.join(figdir, f"{r.run}_lifetimes.png"), dpi=130); plt.close(fig)
    Ns, Ss = sorted(mm.N.unique()), sorted(mm.smearing.unique())
    s0 = a.ref_smearing if a.ref_smearing is not None else min(Ss, key=lambda x: abs(x - 0.1))
    n0 = a.ref_N
    if n0 is None:
        full = [n for n in Ns if len(mm[mm.N == n]) == len(Ss)]
        n0 = max(full) if full else max(Ns)
    # ---- overviews
    for name, sel, key, vals in (("Nscan", mm[np.isclose(mm.smearing, s0)], "N", Ns), ("smearscan", mm[mm.N == n0], "smearing", Ss)):
        sel = sel.sort_values(key)
        if sel.empty:
            continue
        fig, ax = plt.subplots(2, len(sel), figsize=(3.6 * len(sel) + 1, 7), sharex=True, sharey="row", squeeze=False)
        for j, (_, r) in enumerate(sel.iterrows()):
            d = run_df(r)
            for i, kind in enumerate(("magnon", "phonon")):
                draw(ax[i, j], d, kind, lit, label=(j == 0))
                if i == 0:
                    ax[i, j].set_title(f"N = {r.N}^3" if key == "N" else f"smearing {r.smearing:g} meV", fontsize=10)
                if j == 0:
                    ax[i, j].set_ylabel(f"{kind}s: " + ylab)
                if i == 1:
                    ax[i, j].set_xlabel("energy (meV)")
        ax[0, 0].legend(fontsize=7, markerscale=3); ax[1, 0].legend(fontsize=7, markerscale=3)
        fig.suptitle(f"{mat}: " + (f"grid scan at smearing {s0:g} meV" if key == "N" else f"broadening scan at N = {n0}^3") + f", T = {sel['T'].iloc[0]:g} K")
        fig.tight_layout(); fig.savefig(os.path.join(figdir, f"overview_{name}_{mat}.png"), dpi=120); plt.close(fig)
    # ---- binned medians
    fig, ax = plt.subplots(2, 2, figsize=(13, 9), sharex="col")
    for row, (key, sel, vals) in enumerate((("N", mm[np.isclose(mm.smearing, s0)], Ns), ("smearing", mm[mm.N == n0], Ss))):
        cm = plt.get_cmap("viridis")
        for jv, v in enumerate(vals):
            r = sel[np.isclose(sel[key], v)]
            if r.empty:
                continue
            d = run_df(r.iloc[0])
            c = cm(jv / max(1, len(vals) - 1))
            for col, kind in enumerate(("magnon", "phonon")):
                s = d[d.particle == kind]
                for chan, ls in (("gamma_mp", "-"), ("gamma_phph", "--")):
                    m = (s[chan] > 1e-12) & (s.energy_meV > 0)
                    if m.sum() < 10:
                        continue
                    edges = np.geomspace(s.energy_meV[m].min(), s.energy_meV[m].max(), 26)
                    idx = np.digitize(s.energy_meV[m], edges)
                    cen = [np.sqrt(edges[i - 1] * edges[i]) for i in range(1, len(edges)) if (idx == i).sum() >= 3]
                    med = [np.median(Y(s[chan][m].to_numpy()[idx == i])) for i in range(1, len(edges)) if (idx == i).sum() >= 3]
                    ax[row, col].plot(cen, med, ls, color=c, lw=1.6, label=(f"{key} = {v:g}" if chan == "gamma_mp" else None))
        for col, kind in enumerate(("magnon", "phonon")):
            L = lit.get(kind)
            if L is not None and len(L):
                ax[row, col].scatter(L.E, Y(L.g), marker="D", s=14, facecolor="none", edgecolor="0.4", linewidth=0.9, label="literature")
            ax[row, col].set_xscale("log"); ax[row, col].set_yscale("log"); ax[row, col].grid(alpha=0.3, which="both")
            ax[row, col].set_title(f"{kind}s, median per energy bin ({'N scan' if key == 'N' else 'broadening scan'}; dashed: phonon-phonon)", fontsize=9)
            ax[row, col].set_ylabel(ylab)
            ax[row, col].legend(fontsize=7)
        ax[1, 0].set_xlabel("energy (meV)"); ax[1, 1].set_xlabel("energy (meV)")
    fig.suptitle(f"{mat}: N scan at smearing {s0:g} meV (top), broadening scan at N = {n0}^3 (bottom)")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, f"median_{mat}.png"), dpi=130); plt.close(fig)
    print(f"{mat}: figures in {figdir}")
