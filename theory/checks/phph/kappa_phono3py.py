"""Reference phono3py lattice thermal conductivity (SMRTA) of bcc Fe from Inputs/bccFe/PH3 (same fc2.hdf5 / fc3.hdf5 as SpinPhony --phph).
   Run from the repository root:  python theory/checks/phph/kappa_phono3py.py MESH out.txt [sigma_THz ...]
   No sigma: tetrahedron method (phono3py default); with sigma(s): Gaussian smearing in THz. Temperatures 100, 300, 600 K.
   Prints kappa_xx (W/m/K, cubic: xx = yy = zz) and the lifetimes' summary; no isotope scattering, no boundary scattering."""
import sys, numpy as np
from phono3py.cui.load import load
mesh = int(sys.argv[1]); out = sys.argv[2]
sigmas = [float(s) for s in sys.argv[3:]] or None
d = "Inputs/bccFe/PH3/"
ph3 = load(unitcell_filename=d + "POSCAR", supercell_matrix=[4, 4, 4], primitive_matrix="P", calculator="vasp", fc3_filename=d + "fc3.hdf5", fc2_filename=d + "fc2.hdf5", produce_fc=False,
           is_symmetry=True, symmetrize_fc=False, log_level=1)
ph3.mesh_numbers = [mesh] * 3
if sigmas:
    ph3.sigmas = sigmas
ph3.init_phph_interaction()
temps = [100, 300, 600]
ph3.run_thermal_conductivity(temperatures=temps, is_isotope=False, write_kappa=False)
kap = np.array(ph3.thermal_conductivity.kappa)              # (sigma, T, 6)
with open(out, "w") as f:
    f.write(f"# phono3py SMRTA kappa_L bcc Fe, mesh {mesh}^3, {'sigma ' + str(sigmas) + ' THz' if sigmas else 'tetrahedron'}\n# T  kappa_xx kappa_yy kappa_zz (W/m/K)\n")
    for i in range(kap.shape[0]):
        for j, T in enumerate(temps):
            f.write(f"{T} {kap[i, j, 0]:.3f} {kap[i, j, 1]:.3f} {kap[i, j, 2]:.3f}" + (f"  sigma={sigmas[i]}" if sigmas else "") + "\n")
# tau-capped kappa from the mode contributions: 1/tau = 4 pi Gamma [1/ps] (Gamma = linewidth HWHM in THz); a tau cap stands for boundary/isotope scattering
tc = ph3.thermal_conductivity
gam, mk, wts = np.array(tc.gamma), np.array(tc.mode_kappa), np.array(tc.grid_weights)          # (sig,T,gp,band), (sig,T,gp,band,6), (gp)
tau = np.where(gam > 0, 1.0 / (4 * np.pi * np.maximum(gam, 1e-300)), np.inf)
print("check: sum(mode_kappa)/N vs kappa:", (mk[..., 0].sum((2, 3)) / wts.sum())[:, :].ravel()[:3], kap[..., 0].ravel()[:3])
with open(out, "a") as f:
    for cap in (30.0, 100.0, 300.0, 1000.0):
        f.write(f"# tau capped at {cap:g} ps\n")
        fac = np.where(np.isfinite(tau), np.minimum(tau, cap) / np.where(np.isfinite(tau), tau, 1.0), 0.0)
        fac = np.where(gam > 0, fac, 0.0)
        kc = (mk[..., 0] * fac).sum((2, 3)) / wts.sum()
        for i in range(kc.shape[0]):
            for j, T in enumerate(temps):
                f.write(f"{T} {kc[i, j]:.3f}" + (f"  sigma={sigmas[i]}" if sigmas else "") + "\n")
    n0 = (gam <= 0).sum((2, 3))[0]
    f.write(f"# modes with gamma = 0 (excluded from capped sums): {n0.tolist()} per T\n")
print(open(out).read())
