"""
Bare lifetimes by channel: magnon-phonon exchange (V+-), relativistic (V++/V--) and three-phonon (--phph) scattering rates.

    python python/plot_rel_lifetimes.py Outputs/.../path_lifetimes.csv [--out fig.png] [--x energy|path] [--linear_x] [--no_show]

Needs a path_lifetimes.csv written with  SpinPhony.py --relativistic  and/or  --phph  (columns gamma_exch_ps-1, gamma_rel_ps-1, gamma_phph_ps-1).
Left: magnon scattering rate; middle: phonon scattering rate, one colour per channel; right: the fraction of every non-exchange channel,
gamma_channel / gamma_total, for the phonons. --x path plots against the path distance (high-symmetry path) instead of the energy.
Modes with zero rate are not drawn.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--out", default=None)
ap.add_argument("--x", choices=["energy", "path"], default="energy", help="abscissa: mode energy (default) or the path distance")
ap.add_argument("--linear_x", action="store_true", help="linear energy axis (default: log)")
ap.add_argument("--no_show", action="store_true", help="only write the figure")
args = ap.parse_args()
if args.no_show:
    matplotlib.use("Agg")

d = pd.read_csv(args.csv, comment="#")
if "gamma_exch_ps-1" not in d:
    raise SystemExit("no gamma_exch_ps-1 column: run SpinPhony.py with --relativistic and/or --phph")
zero = np.zeros(len(d))
chan = {
    "exchange (magnon-phonon V+-)": (d["gamma_exch_ps-1"].values, "C0"),
    "relativistic (V++/V--)": (d["gamma_rel_ps-1"].values if "gamma_rel_ps-1" in d else zero, "C3"),
    "phonon-phonon (fc3)": (d["gamma_phph_ps-1"].values if "gamma_phph_ps-1" in d else zero, "C2"),
}
chan = {k: v for k, v in chan.items() if k.startswith("exch") or (v[0] > 0).any()}
E = d["energy_meV"].values
X = E if args.x == "energy" else d["path_dist"].values
xlabel = "energy (meV)" if args.x == "energy" else "path distance"
tot = sum(v[0] for v in chan.values())
for kind in ("magnon", "phonon"):
    m = (d["particle"] == kind).values
    print(f"{kind}s: {int(m.sum())} modes; " + "; ".join(f"{k}: sum {v[0][m].sum():.3e} ps^-1, {int((v[0][m] > 0).sum())} with rate > 0" for k, v in chan.items()))

fig, ax = plt.subplots(1, 3, figsize=(17, 4.9))
for a, kind in zip(ax[:2], ("magnon", "phonon")):
    m = (d["particle"] == kind).values
    for lab, (g, c) in chan.items():
        s = m & (g > 1e-10)      # below 1e-10 ps^-1 is numerical zero
        if s.any():
            a.scatter(X[s], g[s], s=7, color=c, alpha=0.5, label=f"{lab} ({int(s.sum())})")
    a.set_yscale("log"); a.set_xlabel(xlabel); a.set_ylabel(r"$\gamma$ (ps$^{-1}$)")
    a.set_title(f"{kind}s"); a.grid(alpha=0.3, which="both"); a.legend(fontsize=8)
a = ax[2]
m = (d["particle"] == "phonon").values & (tot > 0)
for lab, (g, c) in chan.items():
    if lab.startswith("exch") or not (g[m] > 0).any():
        continue
    a.scatter(X[m], g[m] / tot[m], s=7, color=c, alpha=0.5, label=lab)
a.set_xlabel(xlabel); a.set_ylabel(r"$\gamma_{channel}/\gamma_{tot}$ (phonons)"); a.set_ylim(-0.02, 1.02)
a.set_title("share of the non-exchange channels"); a.grid(alpha=0.3); a.legend(fontsize=8)
if args.x == "energy" and not args.linear_x:
    ax[0].set_xscale("log"); ax[1].set_xscale("log")
fig.tight_layout()
out = args.out or args.csv.rsplit(".", 1)[0] + "_channels.png"
fig.savefig(out, dpi=150)
print("wrote", out)
if not args.no_show:
    plt.show()
