"""
Mode-resolved temperatures of the bare dynamics (needs occupation_snapshots_bare.npz, written by SpinPhony.py --dynamics ... --snapshots K).

    python python/plot_mode_temperatures.py RUN_DIR [--compare OTHER_RUN_DIR] [--times 0 1 5 20 100 250] [--out DIR] [--tmax 700]

Mode temperature  T_mode = E / (k_B ln(1 + 1/n))  from the occupation n of every mode (Gamma point and modes below 0.2 meV excluded).
Figures (written to --out, default RUN_DIR/figures_modeT):
  mode_T_hist.png         histograms of T_mode of the magnons and phonons at the chosen times (nearest snapshot); with --compare the second run dashed
  mode_T_vs_E.png         every mode as a dot, energy vs T_mode, one column per time (magnons / phonons)
  mode_T_dispersion.png   dispersion along Gamma-H-N-Gamma-P-H|P-N of the cubic bcc zone from the grid points on the lines, coloured by T_mode, one column per time
Same-energy modes at different q have different T_mode during the relaxation; the colour shows where in the zone the energy sits.
"""
import argparse
import itertools
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

KB = 0.08617333262
ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--compare", default=None)
ap.add_argument("--times", type=float, nargs="+", default=[0, 1, 5, 20, 100, 250])
ap.add_argument("--out", default=None)
ap.add_argument("--tmin", type=float, default=0.0)
ap.add_argument("--tmax", type=float, default=None, help="upper limit of the temperature axes (default: 1.05 x the largest of T_mag0, T_phon0)")
ap.add_argument("--ecut", type=float, default=0.2)
a = ap.parse_args()
plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True})


def load(run):
    z = np.load(os.path.join(run, "occupation_snapshots_bare.npz"))
    return {k: z[k] for k in z.files}


def mode_T(E, n):
    """T_mode (K) per mode; NaN where n <= 0 or E below the cut."""
    with np.errstate(divide="ignore", invalid="ignore"):
        T = E / (KB * np.log1p(1.0 / n))
    T[(n <= 0) | ~np.isfinite(T)] = np.nan
    return T


def pick(z, times):
    t = z["time_ps"]
    idx = [int(np.argmin(np.abs(t - x))) for x in times]
    out, seen = [], set()
    for i in idx:
        if i not in seen:
            seen.add(i); out.append(i)
    return out


Z = load(a.run)
Z2 = load(a.compare) if a.compare else None
out = a.out or os.path.join(a.run, "figures_modeT")
os.makedirs(out, exist_ok=True)
tmax = a.tmax or 1.05 * max(float(Z["T_mag_init"]), float(Z["T_phon_init"]))
sel = pick(Z, a.times)
gi = int(Z["gamma_idx"])
Em, Ep = Z["w_mag"][:, 0], Z["w_phon"]
keep_m = (Em > a.ecut) & (np.arange(len(Em)) != gi)
cm = plt.get_cmap("viridis")
cols = [cm(i / max(1, len(sel) - 1)) for i in range(len(sel))]
lab = lambda z, i: f"{z['time_ps'][i]:g} ps"


def T_of(z, i):
    Tm = mode_T(z["w_mag"][:, 0], z["n_mag"][i][:, 0].astype(float))
    Tp = np.stack([mode_T(z["w_phon"][:, b], z["n_phon"][i][:, b].astype(float)) for b in range(z["w_phon"].shape[1])], 1)
    Tm[~keep_m] = np.nan
    Tp[~((Ep > a.ecut) & (np.arange(len(Em))[:, None] != gi))] = np.nan
    return Tm, Tp


# ---- 1. histograms
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
bins = np.linspace(a.tmin, tmax, 90)
for zz, ls, tag in ((Z, "-", a.run), (Z2, "--", a.compare)):
    if zz is None:
        continue
    for j, i in enumerate(pick(zz, a.times)):
        Tm, Tp = T_of(zz, i)
        for k, T in enumerate((Tm, Tp.ravel())):
            ax[k].hist(T[np.isfinite(T)], bins=bins, histtype="step", color=cols[min(j, len(cols) - 1)], ls=ls, lw=1.3,
                       label=lab(zz, i) if zz is Z else None)
for k, name in enumerate(("magnons", "phonons")):
    ax[k].set_xlabel("mode temperature (K)"); ax[k].set_ylabel("number of modes"); ax[k].set_yscale("log"); ax[k].set_title(name)
    ax[k].set_xlim(a.tmin, tmax); ax[k].legend(fontsize=7, ncol=2)
if Z2 is not None:
    fig.suptitle(f"solid: {os.path.basename(a.run.rstrip('/'))}   dashed: {os.path.basename(a.compare.rstrip('/'))}", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(out, "mode_T_hist.png"), dpi=200); plt.close(fig)

# ---- 2. energy vs temperature dots
fig, ax = plt.subplots(2, len(sel), figsize=(2.6 * len(sel) + 0.6, 5.2), squeeze=False, sharey=True)
for j, i in enumerate(sel):
    Tm, Tp = T_of(Z, i)
    ax[0, j].scatter(Em[keep_m], Tm[keep_m], s=1.2, alpha=0.35, color="#D55E00", linewidths=0, rasterized=True)
    for b in range(Ep.shape[1]):
        ok = np.isfinite(Tp[:, b])
        ax[1, j].scatter(Ep[ok, b], Tp[ok, b], s=1.2, alpha=0.35, color=["#0072B2", "#009E73", "#CC79A7"][b % 3], linewidths=0, rasterized=True)
    ax[0, j].set_title(lab(Z, i), fontsize=9)
    ax[1, j].set_xlabel("energy (meV)")
    for r in (0, 1):
        ax[r, j].grid(alpha=0.25); ax[r, j].set_ylim(a.tmin, tmax)
ax[0, 0].set_ylabel("magnon mode temperature (K)"); ax[1, 0].set_ylabel("phonon mode temperature (K)")
fig.tight_layout(); fig.savefig(os.path.join(out, "mode_T_vs_E.png"), dpi=200); plt.close(fig)

# ---- 3. dispersion coloured by T_mode along the cubic-bcc path
mesh = Z["mesh"].astype(float)
B = Z["reciprocal_lattice"]                                       # rows b_i (1/A, without 2 pi)
fold = Z["q_grid"] / mesh
fold = fold - np.round(fold)
qc = fold @ B                                                      # cartesian q / (2 pi), 1/A
V = abs(np.linalg.det(np.linalg.inv(B).T))                         # primitive cell volume
alen = (2.0 * V) ** (1.0 / 3.0)                                    # cubic lattice constant of the bcc crystal
qn = qc * alen                                                     # q in units of 2 pi / a
G, H_, N_, P_ = np.zeros(3), np.array([1.0, 0, 0]), np.array([0.5, 0.5, 0]), np.array([0.5, 0.5, 0.5])
PATH = [("$\\Gamma$", G, "H", H_), ("H", H_, "N", N_), ("N", N_, "$\\Gamma$", G), ("$\\Gamma$", G, "P", P_), ("P", P_, "H", H_)]
images = []
for perm in itertools.permutations(range(3)):
    for sg in itertools.product([1, -1], repeat=3):
        images.append((perm, np.array(sg)))
tol = 0.6 / float(mesh.max())                                       # in 2 pi/a
xs, qi, ticks, x0 = [], [], [(0.0, "$\\Gamma$")], 0.0
for na, A_, nb, B_ in PATH:
    seg = np.linalg.norm(B_ - A_)
    best = {}
    for perm, sg in images:
        Ai, Bi = (A_[list(perm)] * sg), (B_[list(perm)] * sg)
        d = Bi - Ai
        t = np.clip(((qn - Ai) @ d) / (d @ d), 0, 1)
        dist = np.linalg.norm(qn - (Ai + t[:, None] * d), axis=1)
        for q_i in np.where(dist < tol)[0]:
            if q_i not in best or dist[q_i] < best[q_i][0]:
                best[q_i] = (dist[q_i], t[q_i])
    for q_i, (_, t) in best.items():
        xs.append(x0 + t * seg); qi.append(q_i)
    x0 += seg
    ticks.append((x0, nb))
xs, qi = np.array(xs), np.array(qi)
fig, ax = plt.subplots(2, len(sel), figsize=(2.9 * len(sel) + 0.9, 6.0), squeeze=False)
norm = Normalize(vmin=a.tmin if a.tmin > 0 else 0.0, vmax=tmax)
for j, i in enumerate(sel):
    Tm, Tp = T_of(Z, i)
    sc = ax[0, j].scatter(xs, Em[qi], c=Tm[qi], s=6, cmap="inferno", norm=norm, linewidths=0, rasterized=True)
    for b in range(Ep.shape[1]):
        sc = ax[1, j].scatter(xs, Ep[qi, b], c=Tp[qi, b], s=6, cmap="inferno", norm=norm, linewidths=0, rasterized=True)
    ax[0, j].set_title(lab(Z, i), fontsize=9)
    for r in (0, 1):
        ax[r, j].set_xticks([t for t, _ in ticks]); ax[r, j].set_xticklabels([n for _, n in ticks], fontsize=8)
        for t, _ in ticks:
            ax[r, j].axvline(t, color="0.8", lw=0.5, zorder=0)
        ax[r, j].set_xlim(0, x0)
ax[0, 0].set_ylabel("magnon energy (meV)"); ax[1, 0].set_ylabel("phonon energy (meV)")
fig.tight_layout(rect=[0, 0, 0.93, 1])
cax = fig.add_axes([0.945, 0.15, 0.012, 0.7]); fig.colorbar(sc, cax=cax).set_label("mode temperature (K)")
fig.savefig(os.path.join(out, "mode_T_dispersion.png"), dpi=200); plt.close(fig)
print(f"{len(sel)} times {[round(float(Z['time_ps'][i]), 3) for i in sel]}; {len(set(qi.tolist()))} grid points on the path; figures in {out}")
