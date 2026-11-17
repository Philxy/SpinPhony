"""
Compare the dynamics of a scan written by dyn_scan.sh: several bases (bare / bare_rel / hybrid) and several
initial temperature pairs (T_mag, T_phon).

    python python/plot_dyn_compare.py Outputs/dyn/bccFe_dyn_g20_20261110 [--out_dir DIR] [--logt] [--tmax 20]

One column per basis, one colour per initial pair (T_mag:T_phon), reading observables_dynamics.txt of every run.
  row 1  fitted temperatures: solid = T_mag(t), dashed = T_phon(t)
           bare / bare_rel : T_mag from the (N, E) fit with a chemical potential, T_phon from E with mu = 0
           hybrid          : character-weighted subsystem temperatures of the theory (mu = 0, polariton number not conserved)
  row 2  magnon chemical potential mu_mag(t)   (bare: nonzero when the magnon number is conserved; hybrid: 0 by construction)
  row 3  relative change of the magnon number  N_mag(t)/N_mag(0) - 1   (bare: exactly 0 for the number-conserving
         exchange-striction vertex, nonzero with --relativistic; hybrid: change of the magnon content sum w_mag n)
  row 4  apparent mode temperature spread: T_eff(mu=0 assumption) of the magnons divided by T_mag_fit - the factor that the
         naive mu = 0 inversion over- (or under-) estimates; equals 1 for mu = 0
Also writes dyn_compare_summary.csv (final temperatures, mu, N_mag change, energy drift, tau_e-fold per run).
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_table(path):
    with open(path) as f:
        names = f.readline().split()
    data = np.atleast_2d(np.genfromtxt(path, skip_header=1))
    return {n: data[:, i] for i, n in enumerate(names)}


def read_params(path):
    p = {}
    if os.path.exists(path):
        for line in open(path):
            if "=" in line:
                k, v = line.strip().split("=", 1)
                p[k] = v
    return p


def efold(t, dT):
    ad = np.abs(dT)
    if len(ad) < 3 or not np.isfinite(ad[0]) or ad[0] <= 0:
        return float("nan")
    hit = np.nonzero(ad <= ad[0] / np.e)[0]
    return float(t[hit[0]]) if len(hit) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_dir")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--logt", action="store_true")
    ap.add_argument("--tmax", type=float, default=None)
    a = ap.parse_args()

    runs = []
    for obs in sorted(glob.glob(os.path.join(a.scan_dir, "*", "observables_dynamics.txt"))):
        d = os.path.dirname(obs)
        p = read_params(os.path.join(d, "params.txt"))
        if "mode" not in p:
            continue
        runs.append((p["mode"], float(p["T_mag"]), float(p["T_phon"]), read_table(obs), d))
    if not runs:
        raise SystemExit(f"no finished runs (params.txt + observables_dynamics.txt) below {a.scan_dir}")

    modes = [m for m in ("bare", "bare_rel", "hybrid") if any(r[0] == m for r in runs)]
    pairs = sorted({(r[1], r[2]) for r in runs}, key=lambda x: (-(x[0] - x[1]), x[0]))
    cmap = plt.cm.viridis(np.linspace(0.05, 0.9, max(len(pairs), 2)))
    color = {pr: cmap[i] for i, pr in enumerate(pairs)}
    titles = {"bare": "bare, number-conserving", "bare_rel": "bare + relativistic channels", "hybrid": "hybrid (magnon polaron)"}

    fig, axs = plt.subplots(4, len(modes), figsize=(4.6 * len(modes), 13), squeeze=False, sharex=True)
    rows = []
    for j, mode in enumerate(modes):
        for (m, tm, tp, d, path) in runs:
            if m != mode:
                continue
            t = d["Time_ps"]
            sel = t <= a.tmax if a.tmax else np.ones_like(t, bool)
            c = color[(tm, tp)]
            lab = f"{tm:g}:{tp:g} K"
            axs[0, j].plot(t[sel], d["T_mag_fit_K"][sel], color=c, lw=1.8, label=lab)
            axs[0, j].plot(t[sel], d["T_phon_fit_K"][sel], color=c, lw=1.4, ls="--")
            axs[1, j].plot(t[sel], d["mu_mag_meV"][sel], color=c, lw=1.8)
            axs[2, j].plot(t[sel], d["N_mag"][sel] / d["N_mag"][0] - 1.0, color=c, lw=1.8)
            ratio = np.where(d["T_mag_fit_K"] > 0, d["T_eff_mag_K"] / d["T_mag_fit_K"], np.nan)
            axs[3, j].plot(t[sel], ratio[sel], color=c, lw=1.8)
            E = d["E_tot_meV"]
            rows.append(dict(mode=mode, T_mag0=tm, T_phon0=tp, T_mag_end=d["T_mag_fit_K"][-1], T_phon_end=d["T_phon_fit_K"][-1],
                             mu_end_meV=d["mu_mag_meV"][-1], dNmag_rel=d["N_mag"][-1] / d["N_mag"][0] - 1.0,
                             dE_rel=(E[-1] - E[0]) / E[0], tau_efold_ps=efold(t, d["T_mag_fit_K"] - d["T_phon_fit_K"]), dir=path))
        axs[0, j].set_title(titles[mode], fontsize=11)
        axs[0, j].legend(fontsize=7, title="T_mag:T_phon", title_fontsize=7)
        axs[3, j].axhline(1.0, color="0.5", lw=0.7)
        axs[3, j].set_xlabel("Time (ps)")
        for i in range(4):
            axs[i, j].grid(True, ls=":", alpha=0.5)
            if a.logt:
                axs[i, j].set_xscale("log")
    axs[0, 0].set_ylabel("T (K)   solid: magnons, dashed: phonons")
    axs[1, 0].set_ylabel(r"$\mu_{mag}$ (meV)")
    axs[2, 0].set_ylabel(r"$N_{mag}(t)/N_{mag}(0)-1$")
    axs[3, 0].set_ylabel(r"$T_{eff}^{\mu=0}$ / $T_{mag}$ (fit)")
    fig.tight_layout()
    out = a.out_dir or a.scan_dir
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "dyn_compare.png"), dpi=250)
    print(f" -> {os.path.join(out, 'dyn_compare.png')}")

    import csv
    with open(os.path.join(out, "dyn_compare_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f" -> {os.path.join(out, 'dyn_compare_summary.csv')}")
    for r in rows:
        print(f"{r['mode']:8s} {r['T_mag0']:5g}:{r['T_phon0']:<5g} -> T_mag {r['T_mag_end']:7.2f}  T_phon {r['T_phon_end']:7.2f}  "
              f"mu {r['mu_end_meV']:+.3f} meV  dN_mag {r['dNmag_rel']:+.3e}  dE/E {r['dE_rel']:+.2e}  tau_e {r['tau_efold_ps']:.3g} ps")


if __name__ == "__main__":
    main()
