"""
Self-test of python/partner_diag.py without a GPU (numba CUDA simulator):

    NUMBA_ENABLE_CUDASIM=1 python python/test_partner_diag.py

Random channel list with all four event types; the event sums of the diagnostic must reproduce the rates of the production kernel
phase_lifetime_path (SpinPhony.py) to round-off, the rate-weighted energies must match a plain numpy reference, and the sum of
the per-magnon G_mp shares must equal  sum_phonons (E dn/dT / N_q) gamma.
"""
import os
import sys
import tempfile

os.environ.setdefault("NUMBA_ENABLE_CUDASIM", "1")
import numpy as np
from numba import cuda

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import partner_diag as PD
import SpinPhony as SP

rng = np.random.default_rng(1)
Np, Ng, nbm, nbp, T = 12, 12, 2, 3, 150.0
wm_grid = rng.uniform(1, 80, (Ng, nbm)); wp_grid = rng.uniform(1, 35, (Ng, nbp))
wm_path = rng.uniform(1, 80, (Np, nbm)); wp_path = rng.uniform(1, 35, (Np, nbp))
nm = 1 / np.expm1(wm_grid / (PD.KB * T)); nph = 1 / np.expm1(wp_grid / (PD.KB * T))
M = 3000
ci = np.zeros((7, M), dtype=np.int32)
ci[0] = rng.choice([0, 2, 3, 4], M)
ci[1] = rng.integers(0, Np, M); ci[2] = rng.integers(0, Ng, M); ci[3] = rng.integers(0, Ng, M)
ci[4] = rng.integers(0, nbm, M); ci[5] = rng.integers(0, nbm, M); ci[6] = rng.integers(0, nbp, M)
cw = rng.uniform(0.01, 1.0, M)
d_ci, d_cw = cuda.to_device(ci), cuda.to_device(cw)
d_cnt = cuda.to_device(np.array([M], dtype=np.int64))
d_nm, d_np = cuda.to_device(nm), cuda.to_device(nph)

# production kernel
gm, gp, gmr, gpr = (cuda.to_device(np.zeros(Np * nbm)), cuda.to_device(np.zeros(Np * nbp)),
                    cuda.to_device(np.zeros(Np * nbm)), cuda.to_device(np.zeros(Np * nbp)))
SP.phase_lifetime_path[(M + 255) // 256, 256](d_ci, d_cw, d_cnt, d_nm, d_np, gm, gp, gmr, gpr, Ng)
gm, gp, gmr, gpr = (a.copy_to_host().reshape(Np, -1) for a in (gm, gp, gmr, gpr))

res = PD.run(d_ci, d_cw, d_cnt, d_nm, d_np, cuda.to_device(wm_grid), cuda.to_device(wm_path), wp_path, Ng, Np, T, gm, gmr, gp, gpr)
print("check:", {k: f"{v:.2e}" for k, v in res["check"].items()})
assert all(v < 1e-9 for v in res["check"].values()), res["check"]

# numpy reference for the rate-weighted energies (phonon rows, number-conserving c_type 2)
pref = (2 * np.pi / PD.HBAR) / Ng
R = np.zeros((Np, nbp)); Rhi = np.zeros((Np, nbp)); Rlo = np.zeros((Np, nbp)); Gm = np.zeros((Ng, nbm))
gw = wp_path * PD.dn_dT(wp_path, T) / Np
for j in np.where(ci[0] == 2)[0]:
    _, i, k, p, n, m, lam = ci[:, j]
    r = pref * cw[j] * (nm[p, m] - nm[k, n])
    e1, e2 = wm_grid[k, n], wm_grid[p, m]
    R[i, lam] += r; Rhi[i, lam] += r * max(e1, e2); Rlo[i, lam] += r * min(e1, e2)
    Gm[k, n] += 0.5 * r * gw[i, lam]; Gm[p, m] += 0.5 * r * gw[i, lam]
a = res["acc_phon"]
assert np.allclose(a[..., 0], R) and np.allclose(a[..., 1], Rhi) and np.allclose(a[..., 2], Rlo)
assert np.allclose(res["g_mag"][0], Gm)
assert np.isclose(res["g_mag"][0].sum(), (gw * gp).sum()) and np.isclose(res["g_mag"][1].sum(), (gw * gpr).sum())
with tempfile.TemporaryDirectory() as d:
    PD.write_files(res, d, np.zeros((Np, 3)), wm_path, wp_path, np.zeros((Ng, 3)), wm_grid)
    import pandas as pd
    t = pd.read_csv(f"{d}/path_partner_magnons.csv"); g = pd.read_csv(f"{d}/magnon_Gmp_contrib.csv")
    assert len(t) == Np * (nbm + nbp) and len(g) == Ng * nbm
    assert np.isclose(g["G_exch_meV_ps-1_K-1_per_cell"].sum(), (gw * gp).sum())
    assert (t.loc[t.particle == "phonon", "mag_E_hi_mean_meV"].dropna() >= t.loc[t.particle == "phonon", "mag_E_lo_mean_meV"].dropna()).all()
print("all checks passed (event sums = production rates, energies, G_mp shares, file layout)")
