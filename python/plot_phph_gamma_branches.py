"""
Three-phonon scattering rate vs phonon energy, linear axes, coloured by branch.

    python python/plot_phph_gamma_branches.py Outputs/.../path_lifetimes.csv [--out fig.png] [--ymax 0.5] [--no_show]

Reads gamma_phph_ps-1 (written with SpinPhony.py --phph) of the phonon modes. One panel per branch plus all three overlaid.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--out", default=None)
ap.add_argument("--ymax", type=float, default=None)
ap.add_argument("--no_show", action="store_true")
args = ap.parse_args()
if args.no_show:
    matplotlib.use("Agg")

d = pd.read_csv(args.csv, comment="#")
if "gamma_phph_ps-1" not in d:
    raise SystemExit("no gamma_phph_ps-1 column: run SpinPhony.py with --phph")
p = d[(d.particle == "phonon") & (d["gamma_phph_ps-1"] > 0)]
names = {0: "branch 0", 1: "branch 1", 2: "branch 2"}
cols = {0: "C0", 1: "C2", 2: "C1"}
ymax = args.ymax or 1.05 * p["gamma_phph_ps-1"].quantile(0.999)

fig, ax = plt.subplots(1, 4, figsize=(19, 4.8), sharex=True, sharey=True)
for b in (0, 1, 2):
    s = p[p.branch == b]
    for a in (ax[0], ax[b + 1]):
        a.scatter(s.energy_meV, s["gamma_phph_ps-1"], s=5, color=cols[b], alpha=0.4, label=f"{names[b]} ({len(s)})", rasterized=True)
    print(f"{names[b]}: {len(s)} modes, E {s.energy_meV.min():.1f}-{s.energy_meV.max():.1f} meV, gamma median {s['gamma_phph_ps-1'].median():.3g}, max {s['gamma_phph_ps-1'].max():.3g} ps^-1")
ax[0].set_title("all branches"); [ax[b + 1].set_title(names[b]) for b in (0, 1, 2)]
for a in ax:
    a.set_xlabel("phonon energy (meV)"); a.set_ylim(0, ymax); a.grid(alpha=0.3)
ax[0].set_ylabel(r"$\gamma_{\rm phph}$ (ps$^{-1}$)"); ax[0].legend(markerscale=3, fontsize=9)
fig.tight_layout()
out = args.out or args.csv.rsplit(".", 1)[0] + "_phph_gamma_branches.png"
fig.savefig(out, dpi=150)
print("wrote", out)
if not args.no_show:
    plt.show()
