"""
Energy versus time of a hybrid-dynamics run: magnon-like, phonon-like and total energy.

    python python/plot_energy_time.py --dir Outputs/CrI3_pump [--n_cells 1728] [--logt]

Reads observables_dynamics.txt. In the hybrid dynamics the subsystem energies are character-weighted,
    E_mag  = sum_{k mu} w_mag(k mu) eps(k mu) n(k mu),   E_phon = sum w_ph eps n,   E_tot = sum eps n = E_mag + E_phon,
summed over the active modes; per cell = divided by the number of q-points (read from final_populations_hybrid.npz,
or --n_cells). Output: energy_vs_time.png (left: absolute energies, right: change since t = 0).
"""
import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="run folder with observables_dynamics.txt")
    ap.add_argument("--n_cells", type=int, default=None, help="number of q-points (default: from final_populations_hybrid.npz)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--logt", action="store_true")
    a = ap.parse_args()

    d = read_table(os.path.join(a.dir, "observables_dynamics.txt"))
    N = a.n_cells
    npz = os.path.join(a.dir, "final_populations_hybrid.npz")
    if N is None and os.path.exists(npz):
        N = int(np.load(npz)["n_hyb"].shape[0])
    unit = "meV per cell" if N else "meV (sum over the mesh)"
    s = 1.0 / N if N else 1.0

    t = d["Time_ps"]
    E = {"total": d["E_tot_meV"] * s, "magnon-like": d["E_mag_meV"] * s, "phonon-like": d["E_phon_meV"] * s}
    col = {"total": "k", "magnon-like": "#d62728", "phonon-like": "#1f77b4"}

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    for k, v in E.items():
        ax.plot(t, v, color=col[k], lw=2 if k != "total" else 1.6, ls="-" if k != "total" else "--", label=k)
    ax.set_yscale("log")
    ax.set_ylabel(f"energy ({unit})")
    ax.set_title("Energy content (character-weighted)")

    for k, v in E.items():
        ax2.plot(t, v - v[0], color=col[k], lw=2 if k != "total" else 1.6, ls="-" if k != "total" else "--", label=k)
    ax2.axhline(0, color="0.5", lw=0.7)
    ax2.set_ylabel(f"$E(t) - E(0)$ ({unit})")
    ax2.set_title("Energy change since t = 0")
    for x in (ax, ax2):
        x.set_xlabel("Time (ps)")
        x.grid(True, ls=":", alpha=0.5)
        x.legend(fontsize=9)
        if a.logt:
            x.set_xscale("log")
    fig.tight_layout()
    out = a.out or os.path.join(a.dir, "energy_vs_time.png")
    fig.savefig(out, dpi=250)
    print(f" -> {out}")
    dE_tot = (E["total"][-1] - E["total"][0]) / E["total"][0]
    print(f"E_tot drift: {dE_tot:+.3e} (relative);  dE_mag = {E['magnon-like'][-1] - E['magnon-like'][0]:+.4e}, "
          f"dE_phon = {E['phonon-like'][-1] - E['phonon-like'][0]:+.4e} ({unit})")


if __name__ == "__main__":
    main()
