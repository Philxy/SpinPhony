"""
G_mp versus temperature for bcc Fe in SI units, with and without the number-conserving contribution.

    python python/plot_gmp_T_channels.py Outputs/dyn/<SCAN> [--linear_csv Outputs/G_mp_temperature_scan_bccFe.csv]

  line    : linear-response G_mp(T) from the lifetimes (conserving channels; --linear_csv, columns Temperature_K, G_mp_meV_per_K_ps_per_cell)
  points  : G_eff(t=0) = P_mp / (T_mag - T_phon) of the dynamics runs of the scan, plotted at the mean temperature (T_mag + T_phon)/2
              total            = bare_rel   (conserving + non-conserving)
              conserving       = bare
              non-conserving   = bare_rel - bare   ("without the conserving contribution")
              hybrid           = hybrid run
The dynamics points belong to large temperature differences (nonlinear), so they only coincide with the linear-response line for a small T_mag - T_phon;
for a proper curve run dyn_scan.sh with several equilibrium temperatures and a small difference (see the printed hint).
Output: gmp_vs_T_channels.png / .csv in the scan folder.
"""
import argparse, csv, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SW = 1.602176634e-10 / (2.8665 ** 3 / 2.0 * 1e-30)      # meV K^-1 ps^-1 per cell -> W m^-3 K^-1 (bcc Fe primitive cell)

def read_table(p):
    with open(p) as f: names = f.readline().split()
    d = np.atleast_2d(np.genfromtxt(p, skip_header=1)); return {n: d[:, i] for i, n in enumerate(names)}
def params(p):
    r = {}
    for line in open(p):
        if "=" in line: k, v = line.strip().split("=", 1); r[k] = v
    return r

ap = argparse.ArgumentParser()
ap.add_argument("scan_dir"); ap.add_argument("--linear_csv", default="Outputs/G_mp_temperature_scan_bccFe.csv")
ap.add_argument("--out_dir", default=None); ap.add_argument("--linear_y", action="store_true")
a = ap.parse_args()

runs = {}
for obs in glob.glob(os.path.join(a.scan_dir, "*", "observables_dynamics.txt")):
    d = os.path.dirname(obs); p = params(os.path.join(d, "params.txt"))
    runs[(p["mode"], float(p["T_mag"]), float(p["T_phon"]))] = read_table(obs)
pairs = sorted({(k[1], k[2]) for k in runs if abs(k[1] - k[2]) > 1e-9})
rows = []
for tm, tp in pairs:
    g = {}
    for mode in ("bare", "bare_rel", "hybrid"):
        d = runs.get((mode, tm, tp))
        g[mode] = None if d is None else d["P_mp_cell"][0] / (d["T_mag_fit_K"][0] - d["T_phon_fit_K"][0])
    if g["bare"] is None or g["bare_rel"] is None:
        continue
    rows.append(dict(T_mean=0.5 * (tm + tp), T_mag=tm, T_phon=tp, G_total=g["bare_rel"], G_conserving=g["bare"],
                     G_nonconserving=g["bare_rel"] - g["bare"], G_hybrid=g["hybrid"] if g["hybrid"] is not None else np.nan))

fig, ax = plt.subplots(figsize=(7.5, 5.6))
if os.path.exists(a.linear_csv):
    L = np.genfromtxt(a.linear_csv, delimiter=",", skip_header=1)
    ax.plot(L[:, 0], L[:, 1] * SW, color="k", lw=2, label="linear response (lifetimes, conserving channels)")
mk = dict(G_total=("o", "#1f77b4", "total (conserving + non-conserving)"), G_conserving=("s", "#2ca02c", "conserving only"),
          G_nonconserving=("v", "#d62728", "non-conserving only (without conserving)"), G_hybrid=("D", "#9467bd", "hybrid"))
for key, (m, c, lab) in mk.items():
    x = np.array([r["T_mean"] for r in rows]); y = np.abs(np.array([r[key] for r in rows], float)) * SW
    ok = np.isfinite(y) & (y > 0)
    ax.scatter(x[ok], y[ok], marker=m, s=50, color=c, edgecolors="k", linewidths=0.5, label=lab, zorder=3)
if not a.linear_y: ax.set_yscale("log")
ax.set_xlabel("temperature (K)   [dynamics points: mean of T_mag and T_phon]"); ax.set_ylabel(r"$G_{mp}$ (W m$^{-3}$ K$^{-1}$)")
ax.grid(True, which="both", ls=":", alpha=0.5); ax.legend(fontsize=8); ax.set_title("bcc Fe: G_mp by channel type")
fig.tight_layout()
out = a.out_dir or a.scan_dir
fig.savefig(os.path.join(out, "gmp_vs_T_channels.png"), dpi=250)
with open(os.path.join(out, "gmp_vs_T_channels.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f" -> {os.path.join(out, 'gmp_vs_T_channels.png')}")
for r in rows:
    print(f"T_mean {r['T_mean']:5.0f} K ({r['T_mag']:g}:{r['T_phon']:g}): total {r['G_total']*SW:.3e}  conserving {r['G_conserving']*SW:.3e}  "
          f"non-conserving {r['G_nonconserving']*SW:.3e}  hybrid {r['G_hybrid']*SW:.3e} W/m3K")
print("\nFor a G_mp(T) curve with the two channel classes at many temperatures (linear regime, fast: only the t=0 flux is needed):\n"
      '  MODES="bare bare_rel" TPAIRS="120:100 220:200 320:300 420:400 520:500 620:600" STEPS=2 OBS_EVERY=1 TIME_BARE=00:30:00 SCAN=bccFe_Gmp_T ./dyn_scan.sh')
