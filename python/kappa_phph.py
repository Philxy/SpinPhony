"""
Lattice thermal conductivity in the single-mode relaxation-time approximation from three-phonon lifetimes.

    python python/kappa_phph.py --mesh Inputs/bccFe/combined_band_20x20x20.h5 LIFETIME_CSV [LIFETIME_CSV ...] [--T 300 ...]

kappa = 1/(3 V N) * sum_lambda C_lambda |v_lambda|^2 tau_lambda   (W/m/K), cubic average; C_lambda = k_B x^2 e^x/(e^x-1)^2, x = hbar w / k_B T.
Lifetime csv: phph_lifetimes_T*K.csv (python/phph.py) or a SpinPhony path_lifetimes.csv (--full_bz, column gamma_phph_ps-1; T = --T).
Group velocities: central differences of the sorted mesh energies, like the dispersion in the mesh file. Modes with rate 0 are skipped.
Also prints the cumulative kappa vs energy (what fraction comes from modes below E).
"""
import argparse, sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phph

HBAR, KB = 0.6582119569, 0.08617333262          # meV ps, meV/K
ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="+")
ap.add_argument("--mesh", required=True)
ap.add_argument("--tau_max", type=float, nargs="+", default=[100.0, 1000.0, 1e30],
                help="lifetime caps in ps (a few nearly isolated modes with tau ~ 1e5 ps dominate the uncapped sum; the cap stands for boundary/isotope scattering)")
ap.add_argument("--T", type=float, nargs="+", default=None, help="temperature of each csv (default: read from the file header / file name)")
args = ap.parse_args()

h = phph.harmonic_from_h5(args.mesh)
M = 2 * np.pi * np.linalg.inv(h.A).T                 # rows: dk_cart / df_i  (1/A)
V = abs(np.linalg.det(h.A))
g = h.gradient()                                      # dw/df, [N,3,3] meV
grad_k = np.linalg.solve(M.T[None, None], g[..., None])[..., 0]    # dw/dk_cart, meV A
v = grad_k / HBAR * 100.0                             # A/ps -> m/s
print(f"mesh {h.mesh.tolist()}, V = {V:.4f} A^3, rms |v| {np.sqrt((v**2).sum(-1).mean(0))} m/s per branch")

for n, path in enumerate(args.csv):
    d = pd.read_csv(path, comment="#")
    if args.T:
        T = args.T[n if len(args.T) > 1 else 0]
    else:
        import re
        T = float(re.search(r"T(\d+(?:\.\d+)?)K", path).group(1))
    if "particle" in d:
        d = d[d.particle == "phonon"]
        gam = d["gamma_phph_ps-1"].values
    else:
        gam = d["gamma_ps-1"].values
    iq, br = d.q_idx.values.astype(int), d.branch.values.astype(int)
    w = h.w[iq, br]
    ok = (gam > 1e-12) & (w > 0.1)
    x = w[ok] / (KB * T)
    C = KB * x**2 * np.exp(x) / np.expm1(x) ** 2 * 1.602176634e-22        # J/K per mode
    v2 = (v[iq[ok], br[ok]] ** 2).sum(-1)                                  # m^2/s^2
    tau0 = 1.0 / gam[ok]                                                   # ps
    caps = {c: (C * v2 * np.minimum(tau0, c) * 1e-12).sum() / (3 * V * 1e-30 * h.N) for c in args.tau_max}
    tau = np.minimum(tau0, args.tau_max[len(args.tau_max) // 2]) * 1e-12   # s, middle cap for the breakdown below
    contrib = C * v2 * tau / (3 * V * 1e-30 * h.N)                         # W/m/K per mode
    kappa = contrib.sum()
    o = np.argsort(w[ok]); cum = np.cumsum(contrib[o]) / kappa
    e50, e90 = w[ok][o][np.searchsorted(cum, 0.5)], w[ok][o][np.searchsorted(cum, 0.9)]
    lam = (np.sqrt(v2) * tau)                                              # mean free path (m)
    print(f"{os.path.basename(os.path.dirname(path)) or path}: T = {T:g} K, {int(ok.sum())} modes: kappa_L(SMRTA) with tau cap " + ", ".join(f"{c:g} ps: {k:.1f}" for c, k in caps.items()) + " W/m/K; breakdown below at the middle cap: "
          f"50% from E < {e50:.1f} meV, 90% from E < {e90:.1f} meV; mean free path median {np.median(lam) * 1e9:.1f} nm, max {lam.max() * 1e9:.0f} nm")
    print(f"   by branch: " + ", ".join(f"{b}: {contrib[br[ok] == b].sum():.2f}" for b in range(3)))
