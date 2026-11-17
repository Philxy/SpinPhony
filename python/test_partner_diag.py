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
ci[0] = rng.choice([0, 1, 2, 3, 4], M)
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

# diagnostic, fed in two chunks of path points (as in the chunked path scan: path_idx local to the chunk)
acc = PD.PartnerAccumulator(Np, Ng, nbm, nbp, wp_path, T)
d_wm_grid, d_wm_path = cuda.to_device(wm_grid), cuda.to_device(wm_path)
for i0, i1 in ((0, 5), (5, Np)):
    sel = (ci[1] >= i0) & (ci[1] < i1)
    cl = ci[:, sel].copy(); cl[1] -= i0
    wl = cw[sel].copy()
    acc.add(i0, i1, cuda.to_device(cl), cuda.to_device(wl), cuda.to_device(np.array([sel.sum()], dtype=np.int64)), d_nm, d_np, d_wm_grid, cuda.to_device(wm_path[i0:i1]))
res = acc.finish(gm, gmr, gp, gpr)
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
for a_, b_ in ((2, 0), (4, 0), (3, 1), (5, 1)):      # hi / lo attribution conserve the totals
    assert np.isclose(res["g_mag"][a_].sum(), res["g_mag"][b_].sum())
Ghi = np.zeros((Ng, nbm)); Glo = np.zeros((Ng, nbm))
for j in np.where(ci[0] == 2)[0]:
    _, i, k, p, n, m, lam = ci[:, j]
    r = pref * cw[j] * (nm[p, m] - nm[k, n]); c = r * gw[i, lam]
    kh = wm_grid[k, n] >= wm_grid[p, m]
    Ghi[(k, n) if kh else (p, m)] += c; Glo[(p, m) if kh else (k, n)] += c
assert np.allclose(res["g_mag"][2], Ghi) and np.allclose(res["g_mag"][4], Glo)
with tempfile.TemporaryDirectory() as d:
    PD.write_files(res, d, np.zeros((Np, 3)), wm_path, wp_path, np.zeros((Ng, 3)), wm_grid)
    import csv
    rd = lambda fn: list(csv.DictReader(open(fn)))
    t = rd(f"{d}/path_partner_magnons.csv"); g = rd(f"{d}/magnon_Gmp_contrib.csv")
    assert len(t) == Np * (nbm + nbp) and len(g) == Ng * nbm
    assert np.isclose(sum(float(r["G_exch_meV_ps-1_K-1_per_cell"]) for r in g), (gw * gp).sum())
    ph = [r for r in t if r["particle"] == "phonon" and r["mag_E_hi_mean_meV"] != ""]
    # (hi >= lo of the rate-weighted means holds for physical, positive event rates; the random synthetic occupations here give negative
    #  rates for some events, so only the layout and the filled rows are checked)
    assert ph
print("all checks passed (event sums = production rates, energies, G_mp shares, file layout)")
