"""
Magnon-phonon scattering rate (gamma_exch_ps-1) vs energy on linear axes: magnons and phonons (coloured by branch).

    python python/plot_magph_gamma.py Outputs/.../path_lifetimes.csv [--out fig.png] [--ymax_ph 0.05] [--no_show]

Needs a path_lifetimes.csv with the gamma_exch_ps-1 column (written with --phph and/or --relativistic).
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--out", default=None)
ap.add_argument("--ymax_mag", type=float, default=None)
ap.add_argument("--ymax_ph", type=float, default=None)
ap.add_argument("--no_show", action="store_true")
args = ap.parse_args()
if args.no_show:
    matplotlib.use("Agg")

d = pd.read_csv(args.csv, comment="#")
if "gamma_exch_ps-1" not in d:
    raise SystemExit("no gamma_exch_ps-1 column: the csv needs SpinPhony.py --phph or --relativistic")
g = "gamma_exch_ps-1"
mag = d[(d.particle == "magnon") & (d[g] > 0)]
ph = d[(d.particle == "phonon") & (d[g] > 0)]
cols = {0: "C0", 1: "C2", 2: "C1"}

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].scatter(mag.energy_meV, mag[g], s=6, color="C3", alpha=0.5, rasterized=True, label=f"magnons ({len(mag)})")
ax[0].set_ylim(0, args.ymax_mag or 1.05 * mag[g].max())
for b in (0, 1, 2):
    s = ph[ph.branch == b]
    ax[1].scatter(s.energy_meV, s[g], s=5, color=cols[b], alpha=0.4, rasterized=True, label=f"phonon branch {b} ({len(s)})")
    print(f"phonon branch {b}: {len(s)} modes, median {s[g].median():.3g}, max {s[g].max():.3g} ps^-1")
ax[1].set_ylim(0, args.ymax_ph or 1.05 * ph[g].quantile(0.999))
print(f"magnons: {len(mag)} modes, E {mag.energy_meV.min():.1f}-{mag.energy_meV.max():.1f} meV, median {mag[g].median():.3g}, max {mag[g].max():.3g} ps^-1")
for a, t in zip(ax, ("magnons", "phonons")):
    a.set_xlabel("energy (meV)"); a.set_ylabel(r"magnon-phonon scattering rate $\gamma$ (ps$^{-1}$)"); a.set_title(t)
    a.grid(alpha=0.3); a.legend(markerscale=3, fontsize=9)
fig.tight_layout()
out = args.out or args.csv.rsplit(".", 1)[0] + "_magph_gamma_linear.png"
fig.savefig(out, dpi=150)
print("wrote", out)
if not args.no_show:
    plt.show()
