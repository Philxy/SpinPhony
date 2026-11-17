"""
Effective magnon-phonon coupling G_eff = P_mp / (T_mag - T_phon) at t = 0 for the runs of a dyn_scan.sh scan, split by channel type:
    conserving      : the bare run                     (number-conserving exchange-striction channels)
    non-conserving  : bare_rel - bare                  (the relativistic channels, additive in the rate; same state at t = 0)
    hybrid          : the hybrid run                   (all channels of the polaritons)
At t = 0 all bases start from the same occupations, so the flux difference isolates the added channels exactly.

    python python/plot_gmp_channels.py Outputs/dyn/<SCAN> [--cell_volume 11.777]

Figure gmp_channels.png: left G_eff(0) in meV K^-1 ps^-1 per cell (right axis: W m^-3 K^-1) per initial pair, bars on a log axis;
right: the same G_eff(t) over time for the bare and hybrid runs (conserving vs hybrid). gmp_channels.csv holds the numbers.
"""
import argparse, csv, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MEV_PER_PS_TO_W, ANG3 = 1.602176634e-10, 1e-30

def read_table(p):
    with open(p) as f: names = f.readline().split()
    d = np.atleast_2d(np.genfromtxt(p, skip_header=1)); return {n: d[:, i] for i, n in enumerate(names)}

def params(p):
    r = {}
    for line in open(p):
        if "=" in line: k, v = line.strip().split("=", 1); r[k] = v
    return r

ap = argparse.ArgumentParser()
ap.add_argument("scan_dir"); ap.add_argument("--cell_volume", type=float, default=2.8665 ** 3 / 2.0, help="A^3 (default bcc Fe primitive)")
ap.add_argument("--out_dir", default=None)
a = ap.parse_args()
sw = MEV_PER_PS_TO_W / (a.cell_volume * ANG3)

runs = {}
for obs in glob.glob(os.path.join(a.scan_dir, "*", "observables_dynamics.txt")):
    d = os.path.dirname(obs); p = params(os.path.join(d, "params.txt"))
    runs[(p["mode"], float(p["T_mag"]), float(p["T_phon"]))] = read_table(obs)
pairs = sorted({(k[1], k[2]) for k in runs}, key=lambda x: (-(x[0] - x[1]), x[0]))
pairs = [p for p in pairs if abs(p[0] - p[1]) > 1e-9]                       # G_eff undefined without a temperature difference

rows = []
for tm, tp in pairs:
    r = {"T_mag": tm, "T_phon": tp}
    for mode in ("bare", "bare_rel", "hybrid"):
        d = runs.get((mode, tm, tp))
        r["P_" + mode] = d["P_mp_cell"][0] if d is not None else np.nan
        r["dT"] = (d["T_mag_fit_K"][0] - d["T_phon_fit_K"][0]) if d is not None else r.get("dT", np.nan)
    r["G_conserving"] = r["P_bare"] / r["dT"]
    r["G_nonconserving"] = (r["P_bare_rel"] - r["P_bare"]) / r["dT"]
    r["G_hybrid"] = r["P_hybrid"] / r["dT"]
    rows.append(r)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
x = np.arange(len(rows)); w = 0.27
for i, (key, lab, col) in enumerate((("G_conserving", "conserving (bare)", "#1f77b4"),
                                     ("G_nonconserving", "non-conserving (relativistic, bare_rel - bare)", "#d62728"),
                                     ("G_hybrid", "hybrid", "#2ca02c"))):
    v = np.abs(np.array([r[key] for r in rows], float))
    ax.bar(x + (i - 1) * w, np.where(v > 0, v, np.nan), w, color=col, label=lab)
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{r['T_mag']:g}:{r['T_phon']:g}" for r in rows])
ax.set_xlabel("initial T_mag : T_phon (K)"); ax.set_ylabel(r"$|G_{eff}(t=0)|$ (meV K$^{-1}$ ps$^{-1}$ per cell)")
sec = ax.secondary_yaxis("right", functions=(lambda y: y * sw, lambda y: y / sw)); sec.set_ylabel(r"W m$^{-3}$ K$^{-1}$")
ax.legend(fontsize=8); ax.grid(True, which="both", axis="y", ls=":", alpha=0.5); ax.set_title(r"$G_{eff}$ by channel type at $t=0$")

for (mode, tm, tp), d in sorted(runs.items()):
    if mode in ("bare", "hybrid") and (tm, tp) in pairs:
        ax2.plot(d["Time_ps"], np.abs(d["G_eff_cell"]), ls="-" if mode == "bare" else "--", lw=1.6,
                 color=plt.cm.viridis(pairs.index((tm, tp)) / max(len(pairs) - 1, 1)),
                 label=f"{mode} {tm:g}:{tp:g}")
ax2.set_yscale("log"); ax2.set_xlabel("Time (ps)"); ax2.set_ylabel(r"$|G_{eff}(t)|$ (meV K$^{-1}$ ps$^{-1}$ per cell)")
ax2.legend(fontsize=6, ncol=2); ax2.grid(True, which="both", ls=":", alpha=0.5); ax2.set_title("solid: bare, dashed: hybrid")
fig.tight_layout()
out = a.out_dir or a.scan_dir
fig.savefig(os.path.join(out, "gmp_channels.png"), dpi=250)
with open(os.path.join(out, "gmp_channels.csv"), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
print(f" -> {os.path.join(out, 'gmp_channels.png')}\n -> {os.path.join(out, 'gmp_channels.csv')}")
for r in rows:
    print(f"{r['T_mag']:5g}:{r['T_phon']:<5g} dT={r['dT']:+7.1f}  G_cons={r['G_conserving']:.3e}  G_noncons={r['G_nonconserving']:.3e} "
          f"(ratio {r['G_nonconserving']/r['G_conserving']:.2e})  G_hybrid={r['G_hybrid']:.3e}  [W/m3K: cons {r['G_conserving']*sw:.2e}, hyb {r['G_hybrid']*sw:.2e}]")
