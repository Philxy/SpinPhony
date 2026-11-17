"""
Three-phonon (phonon-phonon) scattering of bcc Fe from third-order force constants.

Implements theory/ph_ph_scattering.tex (vertex Psi, collision integral and the single-mode RTA lifetime)
for a one-atom primitive cell.  Input: phono3py fc3.hdf5 (+ fc2.hdf5, optional), 4x4x4 supercell, eV/A^3.
Harmonic frequencies / eigenvectors are taken either from the SpinPhony phonon mesh file (default, `existing`)
or from the phono3py fc2.hdf5 (`fc2`, self-consistent with fc3; used to validate against phono3py).

    python python/phph.py --material bccFe --T 300 --ibz --out_dir Outputs/phph

Units: energies meV, lengths A, force constants eV/A^3, time ps.

Conventions (tex): 1/tau_q,nu = pi/(hbar N) sum_k sum_{lam lam'} [ |V(-q nu, k lam, k' lam')|^2 (1+n+n') delta(E_q-E_k-E_k')
                                                  + 2 |V(-q nu, -k lam, k' lam')|^2 (n-n') delta(E_q+E_k-E_k') ],
V = (1/sqrt 8) sum e e e Phi~(k, k') prod_i sqrt(hbar^2/(m E_i)),  Phi~(k,k') = sum_{j,l} Phi[0,j,l] <e^{-i k R_j}> <e^{-i k' R_l}>.
Shortest-image averaging of the supercell vectors follows phono3py (each of j, l averaged separately).
"""
import argparse
import itertools
import os
import numpy as np

HBAR_MEV_PS = 0.6582119569                     # meV ps
KB_MEV = 0.08617333262                         # meV / K
HBAR2_OVER_AMU_MEV_A2 = 4.180137               # hbar^2 / amu in meV A^2   ((hbar c)^2 / (amu c^2))
EV_TO_MEV = 1.0e3
OMEGA_CONV = 15.633302 * 4.135667696           # sqrt(eV/A^2/amu) -> meV (same as CrystalDataSoA)


def n_bose(E, T):
    x = np.asarray(E, float) / (KB_MEV * T)
    out = np.zeros_like(x)
    ok = x > 1e-12
    out[ok] = 1.0 / np.expm1(np.minimum(x[ok], 700.0))
    out[x > 700.0] = 0.0
    return out


# ----------------------------------------------------------------------------------------------- harmonic data
class Harmonic:
    """Phonon frequencies (meV, ascending) and eigenvectors e[q, branch, xyz] on a Gamma-centred mesh of a one-atom cell."""

    def __init__(self, mesh, q_int, w, eig, mass, A):
        self.mesh = np.asarray(mesh, int)
        self.q_int = np.asarray(q_int, int)               # (N,3) integer mesh coordinates in [0, mesh)
        self.N = len(self.q_int)
        self.w, self.eig, self.mass, self.A = w, eig, float(mass), np.asarray(A, float)
        self.B = np.linalg.inv(self.A).T                  # rows: reciprocal vectors without 2 pi
        self.grid_map = np.full(tuple(self.mesh), -1, dtype=np.int64)
        self.grid_map[self.q_int[:, 0], self.q_int[:, 1], self.q_int[:, 2]] = np.arange(self.N)
        if np.any(self.grid_map < 0):
            raise ValueError("incomplete q mesh")
        self.q_cart = 2 * np.pi * (self.q_int / self.mesh) @ self.B

    def index(self, qi):
        """grid index of integer coordinates (any shape (...,3)), folded into the mesh."""
        qi = np.asarray(qi) % self.mesh
        return self.grid_map[qi[..., 0], qi[..., 1], qi[..., 2]]

    def gradient(self):
        """d(omega)/d(fractional q) by central differences on the mesh, [N,3 branch,3 dir] in meV per unit of fractional q
        (identical definition to CrystalDataSoA._compute_group_velocities)."""
        g = np.zeros((self.N, 3, 3))
        for i in range(3):
            d = np.zeros(3, int); d[i] = 1
            ip, im = self.index(self.q_int + d), self.index(self.q_int - d)
            g[:, :, i] = (self.w[ip] - self.w[im]) * (self.mesh[i] / 2.0)
        return g


def harmonic_from_h5(path, mass=None):
    import h5py
    with h5py.File(path, "r") as f:
        mesh = f["mesh"][:]
        N = int(f["nqpoint"][()])
        B = f["reciprocal_lattice"][:]
        q_frac = f["q_positions"][:]
        DM = f["dynamical_matrices"][:]
        m = float(f["atom_masses"][0]) if mass is None else mass
        if int(f["natom"][()]) != 1:
            raise ValueError("phph.py is implemented for one-atom cells")
    q_int = np.round(q_frac * mesh).astype(int) % mesh
    ev, vec = np.linalg.eigh(DM)
    w = np.sign(ev) * np.sqrt(np.abs(ev)) * OMEGA_CONV
    eig = np.transpose(vec, (0, 2, 1)).astype(complex)    # [q, branch, xyz]
    return Harmonic(mesh, q_int, w, eig, m, np.linalg.inv(B).T)


class Fc3Bcc:
    """phono3py force constants of a one-atom cell and their lattice Fourier transform on a mesh."""

    def __init__(self, ph3_dir, A, supercell=(4, 4, 4), need_fc2=True):
        import h5py
        self.A = np.asarray(A, float)
        with h5py.File(os.path.join(ph3_dir, "fc3.hdf5"), "r") as f:
            fc3 = f["fc3"][:]
        self.fc2 = None
        if need_fc2:
            with h5py.File(os.path.join(ph3_dir, "fc2.hdf5"), "r") as f:
                self.fc2 = f["fc2"][:]
        if fc3.shape[:3] != (64, 64, 64):
            raise ValueError(f"expected a full (64,64,64,3,3,3) fc3, got {fc3.shape}")
        self.phi = np.ascontiguousarray(fc3[0])             # Phi[0, j, l, a, b, c]; translation invariance gives all others
        self.fc3_full = fc3
        # supercell atom positions: read from phono3py.yaml header (supercell lattice + fractional coords)
        import yaml
        with open(os.path.join(ph3_dir, "phono3py.yaml")) as f:
            y = yaml.safe_load(f)
        L = np.array(y["supercell"]["lattice"], float)
        frac = np.array([p["coordinates"] for p in y["supercell"]["points"]], float)
        self.mass = float(y["primitive_cell"]["points"][0]["mass"])
        P_ph3 = np.array(y["primitive_cell"]["lattice"], float)
        # integer coordinates (basis of the PH3 primitive cell) of all shortest images of every supercell atom
        cart0 = frac @ L
        images = []
        for j in range(len(frac)):
            c = np.array([cart0[j] + np.array(t) @ L for t in itertools.product(range(-2, 3), repeat=3)])
            nrm = np.linalg.norm(c, axis=1)
            images.append(c[nrm < nrm.min() + 1e-5])
        # express R in the integer basis of the *requested* lattice A (rescaled onto the PH3 lattice constant)
        scale = np.linalg.norm(self.A[0]) / np.linalg.norm(P_ph3[0])
        Ainv = np.linalg.inv(self.A)
        self.images = []
        for im in images:
            n = (im * scale) @ Ainv
            if np.abs(n - np.round(n)).max() > 1e-3:
                raise ValueError("supercell vectors are not lattice vectors of the requested primitive basis (different lattice?)")
            self.images.append(np.round(n).astype(int))
        self.mult = np.array([len(i) for i in self.images])
        self.a_ph3 = 2 * P_ph3[0, 0]
        self.P_ph3 = P_ph3

    def phase(self, q_frac):
        """<exp(-2 pi i q.n)> over the shortest images of each of the 64 supercell atoms; q_frac (M,3) -> (M,64)."""
        out = np.zeros((len(q_frac), len(self.images)), dtype=complex)
        for j, im in enumerate(self.images):
            out[:, j] = np.exp(-2j * np.pi * (q_frac @ im.T)).mean(axis=1)
        return out

    def dynamical_matrix(self, q_frac):
        """D(q) = (1/m) sum_j Phi2[0,j] <e^{+i q R_j}>  [eV/A^2/amu]; q_frac (M,3)."""
        P = np.conj(self.phase(q_frac))
        return np.einsum("mj,jab->mab", P, self.fc2[0]) / self.mass


def harmonic_from_fc2(fc3obj, mesh):
    mesh = np.asarray(mesh, int)
    q_int = np.array(list(itertools.product(*[range(m) for m in mesh])), int)
    D = fc3obj.dynamical_matrix(q_int / mesh)
    D = 0.5 * (D + np.transpose(D.conj(), (0, 2, 1)))
    ev, vec = np.linalg.eigh(D)
    w = np.sign(ev) * np.sqrt(np.abs(ev)) * OMEGA_CONV
    return Harmonic(mesh, q_int, w, np.transpose(vec, (0, 2, 1)), fc3obj.mass, fc3obj.A)


# ----------------------------------------------------------------------------------------------- scattering
class PhPh:
    def __init__(self, fc3obj, harm, T, sigma=None, base_smearing=0.3, min_sigma=0.1, E_cut=4.14e-4, slot_mode="sym", gpu_like=False):
        """sigma: fixed Gaussian width (meV); None -> adaptive max(base_smearing*sqrt(var/12), min_sigma) like the magnon-phonon scans."""
        self.fc, self.h, self.T = fc3obj, harm, float(T)
        self.sigma, self.base, self.min_sigma, self.E_cut, self.slot_mode = sigma, base_smearing, min_sigma, E_cut, slot_mode
        self.gpu_like = gpu_like          # delta function of the SpinPhony kernels: Gaussian renormalised over +-2 sigma (0.4179/sigma), zero outside
        self.exclude_gamma = True
        if abs(fc3obj.mass - harm.mass) > 1e-3:
            print(f" warning: mass in fc3 data {fc3obj.mass} != harmonic data {harm.mass}")
        q_frac = harm.q_int / harm.mesh
        self.P = fc3obj.phase(q_frac)                        # (N,64)
        # T[n, j, abc] = sum_l P[n, l] Phi[j, l, abc]   (n runs over the third momentum)
        N = harm.N
        phi_l = np.transpose(fc3obj.phi, (1, 0, 2, 3, 4)).reshape(64, 64 * 27)
        self.Tn = (self.P @ phi_l).reshape(N, 64, 27)
        self.gamma_idx = int(harm.index(np.zeros(3, int)))   # the Gamma point is excluded as a partner (its numerical omega is not exactly 0 in the mesh file)
        self.nBE = n_bose(np.maximum(harm.w, 0.0), T)
        self.grad = harm.gradient() if sigma is None else None
        self.pref = np.where(harm.w > E_cut, np.sqrt(HBAR2_OVER_AMU_MEV_A2 / (harm.mass * np.maximum(harm.w, 1e-30))), 0.0)  # (N,3), Angstrom

    def vertex(self, i1, i2, i3):
        """|V|^2 [meV^2] over (n, nu1, lam2, lam3) for the three slot grid indices (i1 scalar or array, i2, i3 arrays).
        For wavevectors off the 4x4x4 supercell grid the truncated fc3 is not exactly translation covariant, so |V|^2 depends on which slot
        carries no lattice phase (reference slot).  slot_mode 'first': slot 1 is the reference; 'sym' (default): mean over the three choices,
        a permutation-symmetric vertex that reproduces phono3py's linewidths on non-commensurate meshes (validated, see theory/fc3_bccFe_data_notes.md)."""
        i2, i3 = np.atleast_1d(i2), np.atleast_1d(i3)
        i1 = np.broadcast_to(np.asarray(i1), i2.shape)
        a = self._vertex_ref(i1, i2, i3)
        if self.slot_mode == "first":
            return a
        b = np.transpose(self._vertex_ref(i2, i1, i3), (0, 2, 1, 3))
        c = np.transpose(self._vertex_ref(i3, i2, i1), (0, 3, 2, 1))
        return (a + b + c) / 3.0

    def _vertex_ref(self, i1, i2, i3):
        """|V|^2 with slot 1 as the reference slot (no lattice phase); i1, i2, i3 arrays of equal length."""
        h = self.h
        Phit = np.einsum("njx,nj->nx", self.Tn[i3], self.P[i2]).reshape(-1, 3, 3, 3) * EV_TO_MEV      # meV/A^3
        e1, e2, e3 = h.eig[i1], h.eig[i2], h.eig[i3]        # (n,3 branch,3 xyz)
        t = np.einsum("nabc,nxa->nxbc", Phit, e1)
        t = np.einsum("nxbc,nyb->nxyc", t, e2)
        t = np.einsum("nxyc,nzc->nxyz", t, e3)
        pref = (self.pref[i1][:, :, None, None] * self.pref[i2][:, None, :, None] * self.pref[i3][:, None, None, :]) / np.sqrt(8.0)
        return np.abs(t * pref) ** 2

    def _sigma(self, iq, k2, k3, sign2):
        """Gaussian width for the (iq, k2, k3) triples; arrays over the two branch indices (n,3,3) for decay (sign2=-1) /
        coalescence (sign2=+1: E_q + E_k - E_k')."""
        h = self.h
        if self.sigma is not None:
            return self.sigma
        g1 = self.grad[iq]                                 # (3 nu, 3 dir)
        g2, g3 = self.grad[k2], self.grad[k3]              # (n,3,3)
        # decay:  d = g_q - g_k - g_k';  coalescence: d = g_q + g_k - g_k'
        if sign2 < 0:
            d = g1[None, :, None, None, :] - g2[:, None, :, None, :] - g3[:, None, None, :, :]
        else:
            d = g1[None, :, None, None, :] + g2[:, None, :, None, :] - g3[:, None, None, :, :]
        var = ((d / h.mesh) ** 2).sum(-1)
        return np.maximum(self.base * np.sqrt(var / 12.0), self.min_sigma)

    def _delta(self, dE, sg):
        if self.gpu_like:
            return np.where(np.abs(dE) < 2.0 * sg, 0.4179 / sg * np.exp(-0.5 * (dE / sg) ** 2), 0.0)
        return np.exp(-0.5 * (dE / sg) ** 2) / (sg * 2.5066282746310002)

    def rate(self, iq, chunk=2000):
        """returns gamma [1/ps] of the three branches at grid index iq, split into (decay, coalescence)."""
        h = self.h
        kq = h.q_int[iq]
        i_mq = int(h.index(-kq))
        Eq = h.w[iq]
        gam_dec, gam_coa = np.zeros(3), np.zeros(3)
        for s in range(0, h.N, chunk):
            K = h.q_int[s:s + chunk]
            iK = np.arange(s, min(s + chunk, h.N))
            # ---------------- decay q -> k + k':  V(-q, k, k'),  k' = q - k
            i3 = h.index(kq - K)
            Ek, Ep = h.w[iK], h.w[i3]                      # (n,3)
            V2 = self.vertex(i_mq, iK, i3)
            dE = Eq[None, :, None, None] - Ek[:, None, :, None] - Ep[:, None, None, :]
            sg = self._sigma(iq, iK, i3, -1)
            ok = ((Ek[:, None, :, None] > self.E_cut) & (Ep[:, None, None, :] > self.E_cut) & (Eq[None, :, None, None] > self.E_cut)
              & (iK != self.gamma_idx)[:, None, None, None] & (i3 != self.gamma_idx)[:, None, None, None])
            gauss = self._delta(dE, sg) * ok
            occ = 1.0 + self.nBE[iK][:, None, :, None] + self.nBE[i3][:, None, None, :]
            gam_dec += (V2 * occ * gauss).sum(axis=(0, 2, 3))
            # ---------------- coalescence q + k -> k':  V(-q, -k, k'),  k' = q + k
            i2 = h.index(-K)
            i3 = h.index(kq + K)
            Ek, Ep = h.w[iK], h.w[i3]
            V2 = self.vertex(i_mq, i2, i3)
            dE = Eq[None, :, None, None] + Ek[:, None, :, None] - Ep[:, None, None, :]
            sg = self._sigma(iq, iK, i3, +1)
            ok = ((Ek[:, None, :, None] > self.E_cut) & (Ep[:, None, None, :] > self.E_cut) & (Eq[None, :, None, None] > self.E_cut)
              & (iK != self.gamma_idx)[:, None, None, None] & (i3 != self.gamma_idx)[:, None, None, None])
            gauss = self._delta(dE, sg) * ok
            occ = self.nBE[iK][:, None, :, None] - self.nBE[i3][:, None, None, :]
            gam_coa += 2.0 * (V2 * occ * gauss).sum(axis=(0, 2, 3))
        c = np.pi / (HBAR_MEV_PS * h.N)
        return c * gam_dec, c * gam_coa


def cubic_ops():
    ops = []
    for p in itertools.permutations(range(3)):
        for s in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3))
            for i in range(3):
                R[i, p[i]] = s[i]
            ops.append(R)
    return ops


def irreducible_points(h):
    """orbit representatives of the grid under the 48 cubic operations (valid for bcc/fcc/sc, Cartesian axes along the cube)."""
    rep = -np.ones(h.N, int)
    B = h.B
    for i in range(h.N):
        if rep[i] >= 0:
            continue
        rep[i] = i
        q = (h.q_int[i] / h.mesh) @ B
        for R in cubic_ops():
            f = (q @ R.T) @ np.linalg.inv(B) * h.mesh
            fi = np.round(f).astype(int)
            if np.abs(f - fi).max() < 1e-6:
                j = int(h.index(fi))
                if rep[j] < 0:
                    rep[j] = i
    return rep


def average_degenerate(w, g, tol=1e-3):
    g = g.copy()
    i = 0
    while i < len(w):
        j = i + 1
        while j < len(w) and abs(w[j] - w[i]) < tol:
            j += 1
        g[i:j] = g[i:j].mean()
        i = j
    return g


def main():
    ap = argparse.ArgumentParser(description="three-phonon RTA lifetimes of bcc Fe from phono3py fc3")
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--material", default="bccFe")
    ap.add_argument("--ph3_dir", default="Inputs/bccFe/PH3")
    ap.add_argument("--harmonic", choices=["existing", "fc2"], default="existing",
                    help="phonon frequencies/eigenvectors: the SpinPhony mesh file (default) or the phono3py fc2.hdf5")
    ap.add_argument("--mesh", type=int, default=None, help="mesh (only for --harmonic fc2; default: from the config mesh file)")
    ap.add_argument("--T", type=float, nargs="+", default=[300.0])
    ap.add_argument("--sigma", type=float, default=None, help="fixed Gaussian width in meV (default: adaptive like the magnon-phonon scans)")
    ap.add_argument("--smearing", type=float, default=None)
    ap.add_argument("--min_sigma", type=float, default=None)
    ap.add_argument("--E_cut", type=float, default=4.14e-4, help="exclude modes below this energy (meV)")
    ap.add_argument("--slot_mode", choices=["sym", "first"], default="sym", help="vertex reference slot: mean over the three slots (default) or slot 1 only")
    ap.add_argument("--ibz", action="store_true", help="compute irreducible q only and map by cubic symmetry")
    ap.add_argument("--stride", type=int, default=1, help="use every n-th q index only")
    ap.add_argument("--out_dir", default="Outputs")
    args = ap.parse_args()

    import tomllib
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    sim = cfg["simulation"]
    smearing = args.smearing if args.smearing is not None else sim.get("smearing", 0.3)
    min_sigma = args.min_sigma if args.min_sigma is not None else sim.get("min_sigma", 0.1)
    h5 = cfg["materials"][args.material]["mesh"]
    h = harmonic_from_h5(h5)
    fc = Fc3Bcc(args.ph3_dir, h.A)
    if args.harmonic == "fc2":
        h = harmonic_from_fc2(fc, [args.mesh or h.mesh[0]] * 3)
    print(f" mesh {h.mesh.tolist()} ({h.N} q), harmonic = {args.harmonic}, mass {h.mass}, lattice scale a_ph3 = {fc.a_ph3:.5f} A")
    os.makedirs(args.out_dir, exist_ok=True)
    rep = irreducible_points(h) if args.ibz else np.arange(h.N)
    todo = np.unique(rep[::args.stride]) if not args.ibz else np.unique(rep)
    print(f" q points to evaluate: {len(todo)} of {h.N}")
    for T in args.T:
        pp = PhPh(fc, h, T, sigma=args.sigma, base_smearing=smearing, min_sigma=min_sigma, E_cut=args.E_cut, slot_mode=args.slot_mode)
        gd, gc = np.zeros((h.N, 3)), np.zeros((h.N, 3))
        for n, iq in enumerate(todo):
            a, b = pp.rate(int(iq))
            gd[iq], gc[iq] = a, b
            if n % max(1, len(todo) // 10) == 0:
                print(f"   T={T:g} K  {n + 1}/{len(todo)}  q_idx {iq}  E = {np.round(h.w[iq], 2)}  gamma = {np.round(a + b, 6)} ps^-1", flush=True)
        if args.ibz:
            gam = gd + gc
            for i in range(h.N):
                if rep[i] != i:
                    gd[i], gc[i] = gd[rep[i]], gc[rep[i]]
            for i in range(h.N):
                gd[i] = average_degenerate(h.w[i], gd[i]); gc[i] = average_degenerate(h.w[i], gc[i])
            sel = np.arange(h.N)
        else:
            sel = todo
        path = os.path.join(args.out_dir, f"phph_lifetimes_T{T:g}K.csv")
        with open(path, "w") as f:
            f.write(f"# three-phonon SMRTA lifetimes, bcc Fe, harmonic={args.harmonic}, T={T} K, sigma={'adaptive' if args.sigma is None else args.sigma}, "
                    f"smearing={smearing}, min_sigma={min_sigma}, E_cut={args.E_cut}\n")
            f.write("q_idx,qx,qy,qz,branch,energy_meV,gamma_ps-1,tau_ps,gamma_decay_ps-1,gamma_coalesce_ps-1\n")
            for i in sel:
                for b in range(3):
                    g = gd[i, b] + gc[i, b]
                    f.write(f"{i},{h.q_cart[i, 0]:.6f},{h.q_cart[i, 1]:.6f},{h.q_cart[i, 2]:.6f},{b},{h.w[i, b]:.6f},{g:.8e},"
                            f"{(1.0 / g if g > 0 else float('inf')):.6e},{gd[i, b]:.8e},{gc[i, b]:.8e}\n")
        print(f" -> wrote {path}")


if __name__ == "__main__":
    main()
