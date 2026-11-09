"""Reference rates (independent numpy implementation of theory/mag_ph.txt) at a stratified sample of 20^3 grid modes, saved for plot_bare_audit.py.
   python theory/checks/magph/ref_points.py out.npz [n_mag] [n_phon] [smearing]"""
import sys
import numpy as np
sys.path.insert(0, "python"); sys.path.insert(0, "theory/checks/magph")
import ref_bare_rates as R
out = sys.argv[1]
nm = int(sys.argv[2]) if len(sys.argv) > 2 else 30
npn = int(sys.argv[3]) if len(sys.argv) > 3 else 30
smear = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
ref = R.Ref("Inputs/bccFe/combined_band_20x20x20.h5", 600.0, 300.0, slc=True)
order = np.argsort(ref.w_mg)
order = order[ref.w_mg[order] >= R.ECUT]
sel_m = order[np.linspace(0, len(order) - 1, nm).astype(int)]
rng = np.random.default_rng(11)
sel_p = rng.choice(np.arange(1, ref.N), npn, replace=False)
gm = np.array([R.magnon_rate(ref, q, "tex", smear) for q in sel_m])
gp = np.array([[R.phonon_rate(ref, q, b, "ham", smear) for b in range(3)] for q in sel_p])
np.savez(out, q_mag=sel_m, E_mag=ref.w_mg[sel_m], g_mag=gm, q_ph=sel_p, E_ph=ref.w_ph[sel_p], g_ph=gp)
print("saved", out)
