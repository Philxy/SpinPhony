"""
Bare lifetimes: number-conserving (exchange) vs relativistic (V++/V--) channels.

    python python/plot_rel_lifetimes.py Outputs_Fe_rel/path_lifetimes.csv [--out fig.png]

Needs a path_lifetimes.csv written with  SpinPhony.py --relativistic  (columns gamma_exch_ps-1, gamma_rel_ps-1).
Left: scattering rate vs mode energy, exchange (blue) and relativistic (red), magnons and phonons separately.
Right: relativistic fraction gamma_rel / (gamma_exch + gamma_rel) vs energy. Modes with zero rate are not drawn.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--out", default=None)
args = ap.parse_args()

d = pd.read_csv(args.csv, comment="#")
if "gamma_rel_ps-1" not in d:
    raise SystemExit("no gamma_rel_ps-1 column: run SpinPhony.py with --relativistic")
ge, gr = d["gamma_exch_ps-1"].values, d["gamma_rel_ps-1"].values
E = d["energy_meV"].values
print(f"modes: {len(d)}, with relativistic rate > 0: {int((gr > 0).sum())}; "
      f"sum gamma_rel / sum gamma_exch = {gr.sum() / max(ge.sum(), 1e-300):.3e}")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
for a, kind in zip(ax[:2], ("magnon", "phonon")):
    m = (d["particle"] == kind).values
    for g, c, lab in ((ge, "C0", "exchange (V+-)"), (gr, "C3", "relativistic (V++/V--)")):
        s = m & (g > 0)
        a.scatter(E[s], g[s], s=6, color=c, alpha=0.5, label=f"{lab} ({int(s.sum())})")
    a.set_yscale("log"); a.set_xlabel("energy (meV)"); a.set_ylabel(r"$\gamma$ (ps$^{-1}$)")
    a.set_title(f"{kind}s"); a.grid(alpha=0.3, which="both"); a.legend(fontsize=8)
a = ax[2]
tot = ge + gr
for kind, c in (("magnon", "C2"), ("phonon", "C1")):
    s = (d["particle"] == kind).values & (tot > 0)
    a.scatter(E[s], gr[s] / tot[s], s=6, color=c, alpha=0.5, label=kind)
a.set_xlabel("energy (meV)"); a.set_ylabel(r"$\gamma_{rel}/\gamma_{tot}$"); a.set_ylim(-0.02, 1.02)
a.set_title("relativistic fraction"); a.grid(alpha=0.3); a.legend()
ax[0].set_xscale("log")
ax[1].set_xscale("log")
fig.tight_layout()
out = args.out or args.csv.rsplit(".", 1)[0] + "_rel.png"
fig.savefig(out, dpi=150)
plt.show()
print("wrote", out)
