"""
Bare magnon-phonon lifetimes before/after the vertex audit (theory/bare_vertex_audit.md), log-log, plus the independent numpy reference.

    python python/plot_bare_audit.py LEGACY_path_lifetimes.csv FIXED_path_lifetimes.csv [REF.npz] [--out fig.png] [--no_show]

REF.npz from theory/checks/magph/ref_points.py. Left: magnons, right: phonons (all branches); tau = 1/gamma vs energy.
Second figure (<out>_ratio.png): gamma_fixed / gamma_legacy per mode.
"""
import argparse
import io
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("legacy"); ap.add_argument("fixed"); ap.add_argument("ref", nargs="?")
ap.add_argument("--out", default="bare_audit_lifetimes.png")
ap.add_argument("--rate", action="store_true", help="plot the scattering rate gamma instead of tau")
ap.add_argument("--lit_script", default="python/plot-hybrid-lifetimesbccFe.py", help="script holding the literature strings (energy meV, rate ps^-1)")
ap.add_argument("--no_show", action="store_true")
a = ap.parse_args()
if a.no_show:
    matplotlib.use("Agg")
L, F = pd.read_csv(a.legacy, comment="#"), pd.read_csv(a.fixed, comment="#")
R = np.load(a.ref) if a.ref else None
txt = open(a.lit_script).read()
def _lit(name):
    m = re.search(name + r'\s*=\s*"""(.*?)"""', txt, re.S)
    return pd.read_csv(io.StringIO(m.group(1)), names=["E", "g"])
LIT = {"magnon": _lit("literature_scattering_rates"), "phonon": _lit("literature_phonon_scatt_rates_per_ps")}
Y = (lambda g: g) if a.rate else (lambda g: 1.0 / g)
ylab = r"bare scattering rate $\gamma$ (ps$^{-1}$)" if a.rate else r"bare lifetime $\tau$ (ps)"
fig, ax = plt.subplots(1, 2, figsize=(14, 5.3))
for k, kind in enumerate(("magnon", "phonon")):
    l, f = L[L.particle == kind], F[F.particle == kind]
    for d, c, lab, z in ((l, "0.6", "before (legacy kernels)", 1), (f, "C3", "after (LaTeX vertex + absorption)", 2)):
        m = d["gamma_ps-1"] > 1e-12
        ax[k].scatter(d.energy_meV[m], Y(d["gamma_ps-1"][m]), s=4, color=c, alpha=0.35, rasterized=True, zorder=z, label=lab + f" ({int(m.sum())})")
    if R is not None:
        if kind == "magnon":
            E, g = R["E_mag"], R["g_mag"]
        else:
            E, g = np.repeat(R["E_ph"][:, :1], 1, 1)[:, 0], None
            E = R["E_ph"].ravel() if R["E_ph"].ndim == 1 else R["E_ph"].ravel()
            g = R["g_ph"].ravel()
            E = np.array([e for e in np.array(R["E_ph"])]).reshape(-1)
        if kind == "phonon":
            # E_ph is stored per q for all three branches: (n,3)
            Ep = np.load(a.ref)["E_ph"]
            E = Ep.ravel(); g = R["g_ph"].ravel()
        ok = g > 1e-12
        ax[k].scatter(E[ok], Y(g[ok]), marker="x", s=30, color="k", zorder=3, label="independent reference (numpy, from the LaTeX)")
    lit = LIT[kind]
    ax[k].scatter(lit.E, Y(lit.g), marker="D", s=22, facecolor="none", edgecolor="C2", linewidth=1.3, zorder=4, label=f"literature ({len(lit)}, {a.lit_script.split('/')[-1]})")
    ax[k].set_xscale("log"); ax[k].set_yscale("log"); ax[k].set_xlabel("energy (meV)"); ax[k].set_ylabel(ylab)
    ax[k].set_title(kind + "s"); ax[k].grid(alpha=0.3, which="both"); ax[k].legend(fontsize=8, markerscale=3)
fig.tight_layout(); fig.savefig(a.out, dpi=150); print("wrote", a.out)
fig2, ax2 = plt.subplots(1, 2, figsize=(14, 4.6))
for k, kind in enumerate(("magnon", "phonon")):
    l, f = L[L.particle == kind].reset_index(drop=True), F[F.particle == kind].reset_index(drop=True)
    ok = (l["gamma_ps-1"] > 1e-12) & (f["gamma_ps-1"] > 1e-12)
    r = f["gamma_ps-1"][ok] / l["gamma_ps-1"][ok]
    ax2[k].scatter(l.energy_meV[ok], r, s=4, alpha=0.35, color="C3", rasterized=True)
    ax2[k].axhline(1, color="k", lw=0.8); ax2[k].set_xscale("log"); ax2[k].set_yscale("log"); ax2[k].grid(alpha=0.3, which="both")
    ax2[k].set_xlabel("energy (meV)"); ax2[k].set_ylabel(r"$\gamma_{after}/\gamma_{before}$"); ax2[k].set_title(kind + "s")
    print(f"{kind}s: {int(ok.sum())} modes, ratio after/before: median {r.median():.2f}, 10-90% {np.percentile(r, [10, 90]).round(2)}, "
          f"sum gamma after/before {f['gamma_ps-1'][ok].sum() / l['gamma_ps-1'][ok].sum():.2f}; modes zero before/after: "
          f"{int((l['gamma_ps-1'] <= 1e-12).sum())}/{int((f['gamma_ps-1'] <= 1e-12).sum())}")
fig2.tight_layout(); fig2.savefig(a.out.replace(".png", "_ratio.png"), dpi=150)
if not a.no_show:
    plt.show()
