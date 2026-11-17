"""
Phonon angular momentum and magnetization versus time for a hybrid-dynamics run.

    python python/plot_am_magnetization.py --dir Outputs/CrI3_pump [--logt]

Reads observables_dynamics.txt (columns Lz_cell, Sz_cell, Jz_cell, M_over_Meq, M_over_M0 and, for a pumped run, M_over_M0).
  Left   : change of the phonon angular momentum L^z, of the spin S^z and of J^z = S^z + L^z per cell (units of hbar).
           J^z is NOT conserved by the hybridization, so Delta J^z shows how much angular momentum the model creates/removes.
  Middle : magnetization M(t)/M(0), relative to the state at t = 0 (after the pump). The dashed line is the value M(T_eq)/M(0)
           of the fully equilibrated system at the energy-conserving temperature T_eq: the demagnetization the run is heading to.
  Right  : fraction of that equilibrium demagnetization reached,
           (1 - M(t)/M(0)) / (1 - M(T_eq)/M(0)),  and the relative change of the effective magnon number sum w_mag n.
Output: am_magnetization_vs_time.png
"""
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_table(path):
    with open(path) as f:
        names = f.readline().split()
    data = np.atleast_2d(np.genfromtxt(path, skip_header=1))
    return {n: data[:, i] for i, n in enumerate(names)}


ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--out", default=None)
ap.add_argument("--logt", action="store_true")
a = ap.parse_args()

d = read_table(os.path.join(a.dir, "observables_dynamics.txt"))
t = d["Time_ps"]
# M(t)/M(0): present in pumped runs; otherwise rebuild it from M/M(T_eq), which differs only by a constant
M0r = d["M_over_M0"] if "M_over_M0" in d else d["M_over_Meq"] / d["M_over_Meq"][0]
Meq_over_M0 = M0r[0] / d["M_over_Meq"][0]            # M(T_eq) / M(0)  (M0r[0] = 1)
depth_eq = 1.0 - Meq_over_M0                          # equilibrium demagnetization, fraction of M(0)

fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.8))
for key, lab, c, ls in (("Lz_cell", r"$L^z$ (phonon)", "#1f77b4", "-"), ("Sz_cell", r"$S^z$ (spin)", "#d62728", "-"),
                        ("Jz_cell", r"$J^z = S^z + L^z$", "k", "--")):
    ax.plot(t, d[key] - d[key][0], color=c, ls=ls, lw=2, label=lab)
ax.axhline(0, color="0.5", lw=0.7)
ax.set_ylabel(r"change per cell ($\hbar$)")
ax.set_title("Angular momentum")
ax.legend(fontsize=9)

ax2.plot(t, M0r, color="k", lw=2, label=r"$M(t)/M(0)$")
ax2.axhline(Meq_over_M0, color="#2ca02c", ls="--", label=fr"equilibrium $M(T_{{eq}}\!={d['T_eq_K'][0]:.1f}\,K)/M(0)$ = {Meq_over_M0:.4f}")
ax2.set_ylabel(r"$M / M(0)$")
ax2.set_title("Magnetization")
ax2.legend(fontsize=8)

frac = (1.0 - M0r) / depth_eq if depth_eq != 0 else np.full_like(M0r, np.nan)
ax3.plot(t, 100 * frac, color="#9467bd", lw=2, label="fraction of equilibrium demagnetization (%)")
ax3.set_ylabel("%")
ax3b = ax3.twinx()
ax3b.plot(t, 100 * (d["N_mag"] / d["N_mag"][0] - 1.0), color="0.4", ls=":", lw=1.8)
ax3b.set_ylabel(r"$\sum w_{mag} n$ change since t=0 (%), dotted", color="0.4")
ax3.set_title("Approach to the equilibrium magnetization")
ax3.legend(fontsize=8, loc="center right")
for x in (ax, ax2, ax3):
    x.set_xlabel("Time (ps)")
    x.grid(True, ls=":", alpha=0.5)
    if a.logt:
        x.set_xscale("log")
fig.tight_layout()
out = a.out or os.path.join(a.dir, "am_magnetization_vs_time.png")
fig.savefig(out, dpi=250)
print(f" -> {out}")
print(f"dL^z = {d['Lz_cell'][-1]-d['Lz_cell'][0]:+.4e}, dS^z = {d['Sz_cell'][-1]-d['Sz_cell'][0]:+.4e}, dJ^z = {d['Jz_cell'][-1]-d['Jz_cell'][0]:+.4e} hbar/cell "
      f"({100*(d['Jz_cell'][-1]-d['Jz_cell'][0])/abs(d['Jz_cell'][0]):+.1f}% of J^z(0))")
print(f"M/M(0): 1 -> {M0r[-1]:.5f};  equilibrium M(T_eq)/M(0) = {Meq_over_M0:.5f};  reached {100*frac[-1]:.1f}% of the equilibrium demagnetization")
