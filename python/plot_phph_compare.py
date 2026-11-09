"""
Bare dynamics with and without three-phonon scattering (--phph): temperatures, energy and phonon number versus time.

    python python/plot_phph_compare.py Outputs/phph_dyn/off Outputs/phph_dyn/on [--out fig.png] [--no_show]

Reads observables_dynamics.txt of both runs (first = reference, second = with --phph).
Panels: effective temperatures T_eff of the magnon / phonon subsystem, total energy drift, number of phonons, and the
difference of the phonon temperature between the two runs.
"""
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("ref_dir"); ap.add_argument("phph_dir")
ap.add_argument("--out", default="phph_dynamics_comparison.png")
ap.add_argument("--no_show", action="store_true")
ap.add_argument("--labels", nargs=2, default=["without phph", "with phph"])
args = ap.parse_args()
if args.no_show:
    matplotlib.use("Agg")


def read(path):
    with open(path) as f:
        names = f.readline().split()
    data = np.atleast_2d(np.genfromtxt(path, skip_header=1))
    return {n: data[:, i] for i, n in enumerate(names)}


runs = [(read(f"{d}/observables_dynamics.txt"), lab, c) for d, lab, c in zip((args.ref_dir, args.phph_dir), args.labels, ("C0", "C3"))]
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
for o, lab, c in runs:
    t = o["Time_ps"]
    ax[0, 0].plot(t, o["T_eff_mag_K"], c, ls="-", label=f"magnons, {lab}")
    ax[0, 0].plot(t, o["T_eff_phon_K"], c, ls="--", label=f"phonons, {lab}")
    ax[0, 1].plot(t, (o["E_tot_meV"] / o["E_tot_meV"][0] - 1.0), c, label=lab)
    ax[1, 0].plot(t, o["N_phon"], c, label=lab)
for a, (xl, yl) in zip(ax.flat, [("time (ps)", r"$T_{\rm eff}$ (K)"), ("time (ps)", r"$E_{tot}/E_{tot}(0)-1$"), ("time (ps)", r"$N_{\rm phon}$ per cell"), ("time (ps)", r"$\Delta T_{\rm phon}$ (K)")]):
    a.set_xlabel(xl); a.set_ylabel(yl); a.grid(alpha=0.3)
ax[0, 0].legend(fontsize=8); ax[0, 0].set_title("effective temperatures")
ax[0, 1].legend(fontsize=8); ax[0, 1].set_title("energy conservation")
ax[1, 0].legend(fontsize=8); ax[1, 0].set_title("phonon number (not conserved by 3-phonon processes)")
n = min(len(runs[0][0]["Time_ps"]), len(runs[1][0]["Time_ps"]))
ax[1, 1].plot(runs[0][0]["Time_ps"][:n], runs[1][0]["T_eff_phon_K"][:n] - runs[0][0]["T_eff_phon_K"][:n], "k", label="phonons")
ax[1, 1].plot(runs[0][0]["Time_ps"][:n], runs[1][0]["T_eff_mag_K"][:n] - runs[0][0]["T_eff_mag_K"][:n], "gray", label="magnons")
ax[1, 1].legend(fontsize=8); ax[1, 1].set_title("difference (with - without phph)")
fig.tight_layout(); fig.savefig(args.out, dpi=150); print("wrote", args.out)
if not args.no_show:
    plt.show()
