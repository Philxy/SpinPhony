"""
Gallery of candidate publication figures of the bare lifetimes of one convergence run (axes, quantity, layout, style), plus a contact sheet.

    python python/pub_lifetime_gallery.py Outputs/convergence/<tag> [--N 40] [--smearing 0.1] [--material bccFe fccNi] [--out gallery]

Variants (per material): A/B loglog tau/gamma, C/D linear energy + log y, E linear-linear gamma, F/G magnons and phonons in one panel,
H density (hexbin) loglog gamma, I binned median with 10-90 % band. Literature (diamonds) where available.
"""
import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.ticker import NullFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conv_common as cc

ap = argparse.ArgumentParser()
ap.add_argument("scan")
ap.add_argument("--N", type=int, default=40)
ap.add_argument("--smearing", type=float, default=0.1)
ap.add_argument("--material", nargs="*", default=None)
ap.add_argument("--out", default="gallery")
a = ap.parse_args()

# Okabe-Ito (colour-blind safe)
C_MP, C_PH, C_MAG, C_LIT = "#0072B2", "#009E73", "#D55E00", "#222222"
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "legend.fontsize": 9, "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True, "axes.linewidth": 0.9, "figure.dpi": 100})
LT = r"$\tau$ (ps)"
GM = r"$\gamma$ (ps$^{-1}$)"
man = cc.load_scan(a.scan)
os.makedirs(os.path.join(a.scan, a.out), exist_ok=True)


def data(d):
    """list of (label, color, particle, E, gamma) series"""
    out = []
    m = d[d.particle == "magnon"]
    out.append(("magnon-phonon", C_MP, "magnon", m.energy_meV.to_numpy(), m.gamma_mp.to_numpy()))
    p = d[d.particle == "phonon"]
    out.append(("magnon-phonon", C_MP, "phonon", p.energy_meV.to_numpy(), p.gamma_mp.to_numpy()))
    if (p.gamma_phph > 0).any():
        out.append(("phonon-phonon", C_PH, "phonon", p.energy_meV.to_numpy(), p.gamma_phph.to_numpy()))
    return out


def style(ax, xlog, ylog):
    ax.set_xscale("log" if xlog else "linear"); ax.set_yscale("log" if ylog else "linear")
    ax.grid(alpha=0.25, which="major")
    if xlog:
        ax.xaxis.set_minor_formatter(NullFormatter())      # no crowded 2x10^1, 3x10^1 ... labels
    if ylog:
        ax.yaxis.set_minor_formatter(NullFormatter())


def top_legend(fig, axes, one_ax_markerscale=3):
    """one legend above the panels (unique labels), so that it never covers data"""
    h, l, seen = [], [], set()
    for a_ in axes:
        for hh, ll in zip(*a_.get_legend_handles_labels()):
            if ll not in seen:
                seen.add(ll); h.append(hh); l.append(ll)
    for a_ in axes:
        if a_.get_legend() is not None:
            a_.get_legend().remove()
    if l:
        fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, markerscale=one_ax_markerscale, fontsize=9, bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=[0, 0, 1, 0.92])
    else:
        fig.tight_layout()


def draw_scatter(ax, series, kind, lit, tau, ylog, xlog, clip=None, s=3):
    for lab, c, part, E, g in series:
        if part != kind:
            continue
        ok = (g > 1e-12) & (E > 0)
        y = 1.0 / g[ok] if tau else g[ok]
        ax.scatter(E[ok], y, s=s, color=c, alpha=0.35, linewidths=0, rasterized=True, label=lab)
    L = lit.get(kind)
    if L is not None and len(L):
        ax.scatter(L.E, 1.0 / L.g if tau else L.g, marker="D", s=20, facecolor="none", edgecolor=C_LIT, linewidth=1.0, zorder=5, label="literature")
    style(ax, xlog, ylog)
    if not ylog:
        top = np.percentile(np.concatenate([g[(g > 1e-12)] for _, _, p_, _, g in series if p_ == kind]), 99.7)
        ax.set_ylim(0, 1.1 * top * (1 if not tau else 1))


def two_panel(series, lit, tau, xlog, ylog, title):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for k, kind in enumerate(("magnon", "phonon")):
        draw_scatter(ax[k], series, kind, lit, tau, ylog, xlog)
        ax[k].set_xlabel("energy (meV)"); ax[k].set_ylabel(LT if tau else GM)
        ax[k].text(0.03, 0.93, f"({'ab'[k]}) {kind}s", transform=ax[k].transAxes, va="top", fontsize=10)
    top_legend(fig, ax)
    return fig


def one_panel(series, lit, tau):
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    for kind, mk in (("magnon", "o"), ("phonon", "^")):
        for lab, c, part, E, g in series:
            if part != kind:
                continue
            ok = (g > 1e-12) & (E > 0)
            ax.scatter(E[ok], 1.0 / g[ok] if tau else g[ok], s=3, color=c, alpha=0.3, linewidths=0, rasterized=True, marker=mk, label=f"{kind}s: {lab}")
        L = lit.get(kind)
        if L is not None and len(L):
            ax.scatter(L.E, 1.0 / L.g if tau else L.g, marker="D", s=16, facecolor="none", edgecolor=C_LIT, linewidth=0.9, zorder=5,
                       label=f"literature {kind}s")
    style(ax, True, True); ax.set_xlabel("energy (meV)"); ax.set_ylabel(LT if tau else GM)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, markerscale=3, fontsize=7, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    return fig


def hexbin_panel(series, lit):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for k, kind in enumerate(("magnon", "phonon")):
        for lab, c, part, E, g in series:
            if part != kind:
                continue
            ok = (g > 1e-12) & (E > 0)
            cm = {"magnon-phonon": "Oranges" if kind == "magnon" else "Blues", "phonon-phonon": "Greens"}[lab]
            ax[k].hexbin(E[ok], g[ok], xscale="log", yscale="log", gridsize=55, bins="log", cmap=cm, mincnt=1, alpha=0.8, linewidths=0.1)
        L = lit.get(kind)
        if L is not None and len(L):
            ax[k].scatter(L.E, L.g, marker="D", s=16, facecolor="none", edgecolor=C_LIT, linewidth=0.9, zorder=5, label="literature")
        ax[k].set_xlabel("energy (meV)"); ax[k].set_ylabel(GM); style(ax[k], True, True)
        ax[k].text(0.97, 0.05, f"({'ab'[k]}) {kind}s", transform=ax[k].transAxes, ha="right", fontsize=10)
    top_legend(fig, ax)
    return fig


def median_panel(series, lit, tau=False):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for k, kind in enumerate(("magnon", "phonon")):
        for lab, c, part, E, g in series:
            if part != kind:
                continue
            ok = (g > 1e-12) & (E > 0)
            e, gg = E[ok], g[ok]
            edges = np.geomspace(e.min(), e.max(), 22)
            idx = np.digitize(e, edges)
            cen, med, lo, hi = [], [], [], []
            for i in range(1, len(edges)):
                v = gg[idx == i]
                if len(v) >= 5:
                    cen.append(np.sqrt(edges[i - 1] * edges[i])); med.append(np.median(v)); lo.append(np.percentile(v, 10)); hi.append(np.percentile(v, 90))
            cen, med, lo, hi = map(np.array, (cen, med, lo, hi))
            f = (lambda x: 1.0 / x) if tau else (lambda x: x)
            ax[k].fill_between(cen, f(lo), f(hi), color=c, alpha=0.18, linewidth=0)
            ax[k].plot(cen, f(med), color=c, lw=1.8, label=lab)
        L = lit.get(kind)
        if L is not None and len(L):
            ax[k].scatter(L.E, 1.0 / L.g if tau else L.g, marker="D", s=16, facecolor="none", edgecolor=C_LIT, linewidth=0.9, zorder=5, label="literature")
        style(ax[k], True, True); ax[k].set_xlabel("energy (meV)"); ax[k].set_ylabel(LT if tau else GM)
        ax[k].text(0.03, 0.93, f"({'ab'[k]}) {kind}s", transform=ax[k].transAxes, va="top", fontsize=10)
    top_legend(fig, ax)
    return fig


for mat in (a.material or sorted(man.material.unique())):
    r = man[(man.material == mat) & (man.N == a.N) & np.isclose(man.smearing, a.smearing) & man.ok]
    if r.empty:
        print(f"{mat}: no finished run N={a.N} smearing {a.smearing}"); continue
    d = cc.load_run(r.iloc[0].path)
    lit = cc.literature(mat)
    ser = data(d)
    figs = {
        "A_loglog_tau": two_panel(ser, lit, True, True, True, ""),
        "B_loglog_gamma": two_panel(ser, lit, False, True, True, ""),
        "C_linx_logy_tau": two_panel(ser, lit, True, False, True, ""),
        "D_linx_logy_gamma": two_panel(ser, lit, False, False, True, ""),
        "E_linlin_gamma": two_panel(ser, lit, False, False, False, ""),
        "F_onepanel_loglog_gamma": one_panel(ser, lit, False),
        "G_onepanel_loglog_tau": one_panel(ser, lit, True),
        "H_density_loglog_gamma": hexbin_panel(ser, lit),
        "I_median_band_gamma": median_panel(ser, lit, False),
        "J_median_band_tau": median_panel(ser, lit, True),
    }
    paths = []
    for name, fig in figs.items():
        p = os.path.join(a.scan, a.out, f"{mat}_N{a.N}_s{a.smearing:g}_{name}.png")
        fig.savefig(p, dpi=220); fig.savefig(p.replace(".png", ".pdf")); plt.close(fig); paths.append((name, p))
    fig, axs = plt.subplots(4, 3, figsize=(17, 14))
    for ax_, (name, p) in zip(axs.flat, paths):
        ax_.imshow(mpimg.imread(p)); ax_.set_title(name, fontsize=12); ax_.axis("off")
    for ax_ in list(axs.flat)[len(paths):]:
        ax_.axis("off")
    fig.suptitle(f"{mat}  N = {a.N}^3, smearing {a.smearing:g} meV", fontsize=14)
    fig.tight_layout(); cs = os.path.join(a.scan, a.out, f"{mat}_contact_sheet.png"); fig.savefig(cs, dpi=110); plt.close(fig)
    print(f"{mat}: {len(paths)} variants + {cs}")
