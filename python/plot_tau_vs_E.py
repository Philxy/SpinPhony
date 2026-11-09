"""
Lifetime (or, with --rate, scattering rate) vs energy, log-log, magnons and phonons in one plot. Colour = scattering channel, marker = particle.

    python python/plot_tau_vs_E.py Outputs/.../path_lifetimes.csv [--out fig.png] [--rate] [--no_total] [--no_show]

Channels (from the columns written with --phph / --relativistic): magnon-phonon exchange tau = 1/gamma_exch, three-phonon
tau = 1/gamma_phph; the relativistic channel is not drawn. The total tau_ps (all channels) is the grey underlay.
Modes with zero rate in a channel are not drawn for that channel.
"""
import argparse
import io
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--out", default=None)
ap.add_argument("--rate", action="store_true", help="plot the scattering rate gamma (ps^-1) instead of the lifetime")
ap.add_argument("--no_lit", action="store_true", help="do not draw the literature points")
ap.add_argument("--lit_script", default="python/plot-hybrid-lifetimesbccFe.py", help="script holding the literature strings (energy meV, rate ps^-1)")
ap.add_argument("--no_total", action="store_true", help="do not draw the grey total-lifetime underlay")
ap.add_argument("--no_show", action="store_true")
args = ap.parse_args()
if args.no_show:
    matplotlib.use("Agg")

nskip = sum(1 for l in open(args.csv) if l.startswith("#"))
with open(args.csv) as f:
    hdr = next(l for l in f if not l.startswith("#")).strip().split(",")
if "gamma_exch_ps-1" not in hdr:
    raise SystemExit("no gamma_exch_ps-1 column: the csv needs SpinPhony.py --phph (or --relativistic)")
col = {h: i for i, h in enumerate(hdr)}
raw = np.genfromtxt(args.csv, delimiter=",", skip_header=1 + nskip, dtype=str)
part = raw[:, col["particle"]]
E = raw[:, col["energy_meV"]].astype(float)
tau = raw[:, col["tau_ps"]].astype(float)
g = lambda name: raw[:, col[name]].astype(float) if name in col else np.zeros(len(raw))
gam = {"exch": g("gamma_exch_ps-1"), "phph": g("gamma_phph_ps-1")}

fig, ax = plt.subplots(figsize=(8.5, 5.8))
mk = {"magnon": "o", "phonon": "^"}
if not args.no_total:
    m = np.isfinite(tau) & (tau > 0) & (E > 0)
    ax.scatter(E[m], (1.0 / tau[m]) if args.rate else tau[m], s=14, color="0.8", marker="s", label="total (all channels)", rasterized=True, zorder=1)
chan = {"exch": ("C0", "magnon-phonon"), "phph": ("C2", "phonon-phonon")}
for kind in ("magnon", "phonon"):
    for key, (c, lab) in chan.items():
        m = (part == kind) & (gam[key] > 1e-10) & (E > 0)
        if m.any():
            ax.scatter(E[m], gam[key][m] if args.rate else 1.0 / gam[key][m], s=7, color=c, marker=mk[kind], alpha=0.5,
                       label=f"{kind}s, {lab} ({int(m.sum())})", rasterized=True, zorder=2)
if not args.no_lit:
    txt = open(args.lit_script).read()
    for name, lab, c in (("literature_scattering_rates", "literature magnons", "C4"), ("literature_phonon_scatt_rates_per_ps", "literature phonons", "k")):
        lit = pd.read_csv(io.StringIO(re.search(name + r'\s*=\s*"""(.*?)"""', txt, re.S).group(1)), names=["E", "g"])
        ax.scatter(lit.E, lit.g if args.rate else 1.0 / lit.g, marker="D", s=24, facecolor="none", edgecolor=c, linewidth=1.3, zorder=5, label=f"{lab} ({len(lit)})")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("energy (meV)"); ax.set_ylabel(r"scattering rate $\gamma$ (ps$^{-1}$)" if args.rate else r"lifetime $\tau$ (ps)")
ax.grid(alpha=0.3, which="both"); ax.legend(markerscale=2.5, fontsize=8.5)
fig.tight_layout()
out = args.out or args.csv.rsplit(".", 1)[0] + ("_gamma_vs_E.png" if args.rate else "_tau_vs_E.png")
fig.savefig(out, dpi=150)
print("wrote", out)
if not args.no_show:
    plt.show()
