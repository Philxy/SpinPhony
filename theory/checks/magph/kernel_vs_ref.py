"""Run the CUDA path kernels (phase_1_scan_path + phase_lifetime_path) on a few grid points and compare with the numpy reference.
   python theory/checks/magph/kernel_vs_ref.py [n_points] [smearing]         (repository root, needs a GPU)
   Prints per mode: kernel rate | replica of the kernel (reference code, same vertex bookkeeping) | LaTeX-consistent reference."""
import sys, math
import numpy as np
sys.path.insert(0, "python"); sys.path.insert(0, "theory/checks/magph")
import SpinPhony as sp
import ref_bare_rates as R
if "--legacy" in sys.argv:
    sys.argv.remove("--legacy"); sp.BARE_LEGACY = True
from numba import cuda

npts = int(sys.argv[1]) if len(sys.argv) > 1 else 8
smear = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
mesh = "Inputs/bccFe/combined_band_20x20x20.h5"
ref = R.Ref(mesh, 600.0, 300.0, slc=True)
cd = ref.cd
gpu = cd.push_to_gpu()
N = cd.N
rng = np.random.default_rng(5)
sel = np.sort(rng.choice(np.arange(1, N), npts, replace=False))
qf = cd.q_grid[sel].astype(np.float64) / cd.mesh
d_qf = cuda.to_device(qf); d_qc = cuda.to_device(cd.q_grid_cart[sel])
d_wp = cuda.to_device(cd.w_phon[sel]); d_wm = cuda.to_device(cd.w_mag[sel]); d_ep = cuda.to_device(cd.eig_phon[sel])
d_gq = cuda.to_device(cd.q_grid.astype(np.float64) / cd.mesh)
cap = 400_000_000 // 8
d_idx = cuda.device_array((7, cap), dtype=np.int32); d_w = cuda.device_array(cap, dtype=np.float64); d_cnt = cuda.to_device(np.zeros(1, dtype=np.int64))
gam = int(cd.grid_map[0, 0, 0])
thr = (16, 16)
sp.phase_1_scan_path[(math.ceil(npts / 16), math.ceil(N / 16)), thr](
    gpu["mesh"], d_gq, gpu["q_grid_cart"], gpu["grid_map"], d_qf, d_qc, d_wp, d_wm, d_ep,
    gpu["w_phon"], gpu["w_mag"], gpu["eig_phon"], gpu["grad_f_phon"], gpu["grad_f_mag"],
    gpu["slc_axis"], gpu["slc_rij"], gpu["slc_rik"], gpu["slc_J"], gpu["slc_types"], smear, d_idx, d_w, d_cnt,
    gpu["atom_masses"], gpu["mag_moments"], gam)
cuda.synchronize()
n = int(d_cnt.copy_to_host()[0])
n_mag = sp.init_bose_einstein(cd.w_mag, 600.0); n_ph = sp.init_bose_einstein(cd.w_phon, 300.0)
d_nm, d_np = cuda.to_device(n_mag), cuda.to_device(n_ph)
gm = cuda.to_device(np.zeros(npts)); gp = cuda.to_device(np.zeros(npts * 3)); z1 = cuda.to_device(np.zeros(npts)); z2 = cuda.to_device(np.zeros(npts * 3))
sp.phase_lifetime_path[math.ceil(n / 256), 256](d_idx[:, :n], d_w[:n], d_cnt, d_nm, d_np, gm, gp, z1, z2, N)
cuda.synchronize()
gm, gp = gm.copy_to_host(), gp.copy_to_host().reshape(npts, 3)
legacy = sp.BARE_LEGACY
mv, pv = ("kernel", "kernel") if legacy else ("tex", "ham")      # new kernels: magnon V_{q,q-k} (both terms), phonon V_{k,+q}
print(f"BARE_LEGACY={legacy}: {n:,} channels; mode | kernel | reference with the same bookkeeping ({mv}/{pv}) | LaTeX-consistent (V_H, emission + absorption)")
for i, q in enumerate(sel):
    print(f"magnon q={q:5d} E={ref.w_mg[q]:6.1f}: {gm[i]:10.4g} {R.magnon_rate(ref, q, mv, smear):10.4g} {R.magnon_rate(ref, q, 'ham', smear):10.4g}")
for i, q in enumerate(sel):
    lam = int(np.argmax(gp[i]))
    print(f"phonon q={q:5d} b={lam} E={ref.w_ph[q, lam]:5.1f}: {gp[i, lam]:10.4g} {R.phonon_rate(ref, q, lam, pv, smear):10.4g} {R.phonon_rate(ref, q, lam, 'ham', smear):10.4g}")
