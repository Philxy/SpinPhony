"""
Frequency-resolved coupling function G_mp(omega, T): which phonons carry the magnon-phonon heat transfer.

    python python/plot_gmp_spectral.py --dir Outputs/dyn/<SCAN>/bccFe_hybrid_Tm300_Tp300_g20 [--T 300] [--nbins 60]

Uses the same lifetimes as the G_mp calculation (plot-hybrid-lifetimes*.py): with the rate gamma of every phonon(-like) mode,
    G_mp(T) = (1/N) sum_{q,lambda} w_ph E gamma dn_BE/dT ,      dn_BE/dT = E / (k_B T^2) n (n + 1)
and the spectral function
    G(omega, T) = (1/N) sum_{modes with E in the bin} w_ph E gamma dn_BE/dT / (bin width),     G_mp(T) = integral G(omega) d omega .
Bare: the phonon rows of path_lifetimes.csv (w_ph = 1; branch index = energy order at each q: for bcc Fe branches 0, 1 are the transverse and
branch 2 the longitudinal acoustic one). Hybrid: hybrid_path_lifetimes.csv, w_ph = phon_character.
The rates are those of the run (equal-temperature run: T_mag = T_phon = T); a spectrum at another T needs a lifetime run at that T.

Output gmp_spectral.png (a: G(omega) bare by branch and hybrid, b: cumulative fraction, c: hybrid/bare) and gmp_spectral.csv in --dir.
Units: W m^-3 K^-1 per meV (cell volume: --cell_volume, default bcc Fe primitive cell).
"""
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KB = 0.08617333262
MEV_PER_PS_TO_W, ANG3 = 1.602176634e-10, 1e-30

def dn_dT(E, T):
    x = E / (KB * T); out = np.zeros_like(x); ok = (x > 0) & (x < 700)
    out[ok] = (x[ok] / T) / (4.0 * np.sinh(0.5 * x[ok]) ** 2)
    return out

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True); ap.add_argument("--T", type=float, default=None, help="K (default: from params.txt T_mag, else from n_occ)")
ap.add_argument("--nbins", type=int, default=60); ap.add_argument("--cell_volume", type=float, default=2.8665 ** 3 / 2.0)
ap.add_argument("--linear_y", action="store_true")
a = ap.parse_args()

b = pd.read_csv(os.path.join(a.dir, "path_lifetimes.csv"), comment="#")
h = pd.read_csv(os.path.join(a.dir, "hybrid_path_lifetimes.csv"), comment="#")
T = a.T
if T is None:
    pf = os.path.join(a.dir, "params.txt")
    if os.path.exists(pf):
        for line in open(pf):
            if line.startswith("T_mag="): T = float(line.split("=")[1])
    if T is None:
        s = (h.energy_meV > 0.5) & (h.n_occ > 1e-8)
        T = float(np.median(h.energy_meV[s] / (KB * np.log1p(1 / h.n_occ[s]))))
N = b.q_idx.nunique(); V_m3 = a.cell_volume * ANG3; sw = MEV_PER_PS_TO_W / V_m3

def contrib(E, gamma, w):
    g = np.where(np.isfinite(gamma), gamma, 0.0)
    return w * E * g * dn_dT(E, T) / N                    # meV K^-1 ps^-1 per cell, one entry per mode

bp = b[b.particle == "phonon"]; Eb = bp.energy_meV.to_numpy(float)
cb = contrib(Eb, bp["gamma_ps-1"].to_numpy(float), 1.0) * (Eb > 0.2)
Eh = h.energy_meV.to_numpy(float); wh = np.clip(h.phon_character.to_numpy(float), 0, 1)
ch = contrib(Eh, h["gamma_ps-1"].to_numpy(float), wh) * (Eh > 0.2)

emax = max(Eb.max(), Eh[wh > 0.5].max()) if (wh > 0.5).any() else Eb.max()
edges = np.linspace(0, emax * 1.0001, a.nbins + 1); dw = np.diff(edges); cen = 0.5 * (edges[1:] + edges[:-1])
def hist(E, c): return np.histogram(E, bins=edges, weights=c)[0] / dw * sw       # W m^-3 K^-1 meV^-1

branch_h = {k: hist(Eb[bp.branch.to_numpy() == k], cb[bp.branch.to_numpy() == k]) for k in sorted(bp.branch.unique())}
Gb = hist(Eb, cb); Gh = hist(Eh, ch)
Gb_tot, Gh_tot = cb.sum() * sw, ch.sum() * sw
print(f"T = {T:g} K, N = {N} q-points, V = {a.cell_volume:.3f} A^3")
print(f"G_mp bare (phonon rows)  = {Gb_tot:.4e} W m^-3 K^-1   ({cb.sum():.4e} meV/(K ps)/cell)")
print(f"G_mp hybrid (phonon-like) = {Gh_tot:.4e} W m^-3 K^-1   ({ch.sum():.4e} meV/(K ps)/cell)   ratio hybrid/bare = {Gh_tot/Gb_tot:.3g}")
for name, G in (("bare", Gb), ("hybrid", Gh)):
    cum = np.cumsum(G * dw); cum /= cum[-1]
    print(f"  {name:6s}: 50% of G_mp from phonons below {cen[np.searchsorted(cum, 0.5)]:.1f} meV, 90% below {cen[np.searchsorted(cum, 0.9)]:.1f} meV")
for k, g in branch_h.items():
    print(f"  bare branch {k}: {100 * (g * dw).sum() * 1.0 / (Gb * dw).sum():.1f}% of G_mp")

fig, axs = plt.subplots(1, 3, figsize=(17, 4.9))
ax = axs[0]
cols = plt.cm.Blues(np.linspace(0.45, 0.95, max(len(branch_h), 2)))
bottom = np.zeros_like(cen)
for (k, g), c in zip(branch_h.items(), cols):
    ax.fill_between(cen, bottom, bottom + g, step="mid", color=c, alpha=0.8, label=f"bare, branch {k}"); bottom = bottom + g
ax.step(cen, Gh, where="mid", color="#d62728", lw=2, label="hybrid (phonon-like, w_ph weighted)")
ax.set_xlabel("phonon energy (meV)"); ax.set_ylabel(r"$\mathcal{G}_{mp}(\omega)$  (W m$^{-3}$ K$^{-1}$ meV$^{-1}$)")
if not a.linear_y: ax.set_yscale("log")
ax.set_title(fr"Spectral coupling at $T$ = {T:g} K"); ax.legend(fontsize=7); ax.grid(True, which="both", ls=":", alpha=0.4)
ax = axs[1]
for name, G, c in (("bare", Gb, "#1f77b4"), ("hybrid", Gh, "#d62728")):
    cum = np.cumsum(G * dw); ax.plot(cen, cum / cum[-1], color=c, lw=2, label=name)
ax.set_xlabel("phonon energy (meV)"); ax.set_ylabel(r"cumulative fraction of $G_{mp}$"); ax.legend(); ax.grid(True, ls=":", alpha=0.5)
ax.set_title("Which phonons carry the transfer")
ax = axs[2]
ok = (Gb > 0) & (Gh > 0); ax.plot(cen[ok], Gh[ok] / Gb[ok], "o-", color="k", ms=4)
ax.axhline(1.0, color="0.5", lw=0.8); ax.set_yscale("log"); ax.set_xlabel("phonon energy (meV)"); ax.set_ylabel("hybrid / bare")
ax.set_title("Ratio of the spectral functions"); ax.grid(True, which="both", ls=":", alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(a.dir, "gmp_spectral.png"), dpi=250)
pd.DataFrame({"E_meV": cen, "G_bare_W_m3K_per_meV": Gb, "G_hybrid_W_m3K_per_meV": Gh,
              **{f"G_bare_branch{k}": g for k, g in branch_h.items()}}).to_csv(os.path.join(a.dir, "gmp_spectral.csv"), index=False)
print(f" -> {os.path.join(a.dir, 'gmp_spectral.png')}")
