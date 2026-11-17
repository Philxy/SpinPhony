"""
GPU self-test of the multi-sublattice bare vertices (U transformation of theory/bare_lifetimes.tex, 'Multi-Sublattice Eigenmode Basis').

    python python/test_bare_multisublattice.py [--config config.toml] [--material CrI3] [--cases 400]     (needs a GPU; run through Slurm)

 A  host: sublattice positions tau, U table, the umklapp phase  U(k+G) = exp(-i G.tau) U(k)  checked on the magnon matrix A(k) itself
    (for both signs of the phase, the right one must reproduce A(k+G) = Phi A(k) Phi^+ up to the 1 % rounding of the Jij bond vectors)
 B  device vs independent numpy reference: |V^{+-}_{nu nu'}|^2 of calc_vertex_V_path_band for random label sets (incl. labels outside the table
    cell, i.e. umklapp), and the invariance of |V|^2 under a common reciprocal-lattice shift of both magnon labels (same physical process)
 C  device vs numpy reference of the relativistic pair amplitude A_{nu nu'} (kind 0 and 1, with a synthetic anisotropic tensor, since the
    real CrI3 tensor has V^{++} = 0), and the exchange symmetry A_{nu nu'}(k1,k2) = A_{nu' nu}(k2,k1)
 D  one magnetic sublattice: calc_vertex_V_path_band with U = 1 reproduces calc_vertex_V_path exactly
"""
import argparse
import os
import sys
import tomllib

import numpy as np
from numba import cuda

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import SpinPhony as SP

HBAR = 0.6582119569
DALTON = 0.10364269


# ---------------------------------------------------------------- numpy reference of the sublattice vertices
class Ref:
    def __init__(self, cd):
        self.axis = np.asarray(cd.slc_axis); self.rij = np.asarray(cd.slc_rij); self.rik = np.asarray(cd.slc_rik)
        self.J = np.asarray(cd.slc_J); self.types = np.asarray(cd.slc_types)
        self.mass = np.asarray(cd.atom_masses) * DALTON; self.mom = np.asarray(cd.mag_moments)

    def Jt(self, p, q, n, m, l, mu):
        r = (self.axis == mu) & (self.types[:, 0] == n + 1) & (self.types[:, 1] == m + 1) & (self.types[:, 2] == l + 1)
        ph = self.rij[r] @ p + self.rik[r] @ q
        return (self.J[r] * np.exp(1j * ph)[:, None, None]).sum(0)

    def V_cons(self, kp, q, e, omega, n, m):                      # calc_vertex_V_path_c
        Sn, Sm = abs(self.mom[n] / 2), abs(self.mom[m] / 2)
        V = 0j
        for l in range(len(self.mass)):
            amp = np.sqrt(HBAR ** 2 / (2 * self.mass[l] * omega))
            for mu in range(3):
                Jd = self.Jt(kp - q, q, n, m, l, mu)
                Wd = (Jd[0, 0] + Jd[1, 1] - 1j * Jd[0, 1] + 1j * Jd[1, 0]) / np.sqrt(Sn * Sm)
                Ws = 0j
                if n == m:
                    for mp in range(len(self.mom)):
                        if abs(self.mom[mp]) > 1e-2:
                            Ws += 2.0 / Sn * self.Jt(np.zeros(3), q, n, mp, l, mu)[2, 2]
                V += amp * e[l, mu] * (Wd - Ws)
        return V

    def V_nc(self, kind, k, q, conj, e, omega, n, m):             # calc_vertex_nc
        Sn, Sm = abs(self.mom[n] / 2), abs(self.mom[m] / 2)
        sn, sm = (1.0 if self.mom[n] >= 0 else -1.0), (1.0 if self.mom[m] >= 0 else -1.0)
        a, sg = (k - q, 1.0) if kind == 0 else (-k - q, -1.0)
        V = 0j
        for l in range(len(self.mass)):
            amp = np.sqrt(HBAR ** 2 / (2 * self.mass[l] * omega))
            for mu in range(3):
                emu = np.conj(e[l, mu]) if conj else e[l, mu]
                J = self.Jt(a, q, n, m, l, mu)
                W = (J[0, 0] - sn * sm * J[1, 1] + sg * 1j * sm * J[0, 1] + sg * 1j * sn * J[1, 0]) / (2 * np.sqrt(Sn * Sm))
                V += amp * emu * W
        return V


def gauge(d, tau, n, sign=-1.0):
    return np.exp(sign * 1j * (d @ tau[n]))


# ---------------------------------------------------------------- device wrappers
@cuda.jit
def k_band(kp, q, lam, nu, nup, row1, c1, row2, c2, iph, U, tau, slc_axis, slc_rij, slc_rik, slc_J, slc_types, eig_phon, w_phon, masses, moms, out):
    i = cuda.grid(1)
    if i >= out.shape[0]:
        return
    out[i] = SP.calc_vertex_V_path_band(kp[i, 0], kp[i, 1], kp[i, 2], q[i, 0], q[i, 1], q[i, 2], 0.0, 0.0, 0.0, lam[i], nu[i], nup[i],
                                        U, row1[i], c1[i, 0], c1[i, 1], c1[i, 2], U, row2[i], c2[i, 0], c2[i, 1], c2[i, 2], tau,
                                        slc_axis, slc_rij, slc_rik, slc_J, slc_types, eig_phon[iph[i]], w_phon[iph[i], lam[i]], masses, moms)


@cuda.jit
def k_old(kp, q, lam, n, m, iph, slc_axis, slc_rij, slc_rik, slc_J, slc_types, eig_phon, w_phon, masses, moms, out):
    i = cuda.grid(1)
    if i >= out.shape[0]:
        return
    out[i] = SP.calc_vertex_V_path(kp[i, 0], kp[i, 1], kp[i, 2], q[i, 0], q[i, 1], q[i, 2], 0.0, 0.0, 0.0, lam[i], n[i], m[i],
                                   slc_axis, slc_rij, slc_rik, slc_J, slc_types, eig_phon[iph[i]], w_phon[iph[i], lam[i]], masses, moms)


@cuda.jit
def k_pair(kind, k1, k2, q, conj, lam, nu, nup, row1, c1, row2, c2, iph, U, tau, slc_axis, slc_rij, slc_rik, slc_J, slc_types, eig_phon, w_phon, masses, moms, out):
    i = cuda.grid(1)
    if i >= out.shape[0]:
        return
    out[i] = SP.calc_pair_amplitude_nc_band(kind, k1[i, 0], k1[i, 1], k1[i, 2], k2[i, 0], k2[i, 1], k2[i, 2], q[i, 0], q[i, 1], q[i, 2], conj,
                                            eig_phon[iph[i]], lam[i], w_phon[iph[i], lam[i]], nu[i], nup[i],
                                            U, row1[i], c1[i, 0], c1[i, 1], c1[i, 2], U, row2[i], c2[i, 0], c2[i, 1], c2[i, 2], tau,
                                            slc_axis, slc_rij, slc_rik, slc_J, slc_types, masses, moms, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.toml"); ap.add_argument("--material", default="CrI3"); ap.add_argument("--cases", type=int, default=300)
    a = ap.parse_args()
    cfg = tomllib.load(open(a.config, "rb"))["materials"][a.material]
    rng = np.random.default_rng(3)
    cd = SP.CrystalDataSoA(cfg["mesh"], cfg["jij"], cfg["slc_files"], cfg["lattice_constant"], cfg["anisotropy"], include_slc=False)
    nb = cd.n_mag_branches
    print(f"\n#### material {a.material}: {nb} magnetic sublattices, {cd.N} grid points")
    B = cd.reciprocal_lattice * 2 * np.pi
    U, info = cd.compute_bare_magnon_U(cd.q_grid_cart, cd.w_mag, cfg["anisotropy"], cfg["lattice_constant"])
    tau, dev = cd.compute_mag_tau(cfg["lattice_constant"])
    print(f"A  tau = {np.round(tau, 4).tolist()} A, bond deviation {dev:.4f};  U: anomalous block {info['max_anomalous']:.2e}, energy mismatch {info['max_energy_mismatch']:.2e} meV")
    # A. umklapp phase on the magnon matrix itself
    bi, bj, br, bJ = cd._mag_bonds(cfg["lattice_constant"])
    S = abs(cd.mag_moments[cd.mag_indices][0]) / 2
    J0 = np.zeros((nb, nb))
    for i, j, J in zip(bi, bj, bJ):
        J0[i, j] += J

    def Amat(k):
        Jk = np.zeros((nb, nb), complex)
        for i, j, r, J in zip(bi, bj, br, bJ):
            Jk[i, j] += J * np.exp(1j * (k @ r))
        return -S * (Jk - np.diag(J0.sum(1)))

    if nb > 1:
        for sign in (-1.0, +1.0):
            errs = []
            for _ in range(20):
                k = cd.q_grid_cart[rng.integers(cd.N)]
                G = (rng.integers(-1, 2, 3)) @ B
                Phi = np.diag([gauge(G, tau, n, sign) for n in range(nb)])
                A0, A1 = Amat(k), Amat(k + G)
                errs.append(np.linalg.norm(A1 - Phi @ A0 @ Phi.conj().T) / max(np.linalg.norm(A1), 1e-12))
            print(f"A  umklapp phase sign {sign:+.0f}: max rel. error of A(k+G) = Phi A(k) Phi^+ : {max(errs):.3e}")
            if sign == -1.0:
                err_used = max(errs)
        assert err_used < 0.15, "umklapp phase of the sublattice positions does not reproduce the magnon matrix"

    # device arrays
    g = lambda x: cuda.to_device(np.ascontiguousarray(x))
    d_U, d_tau = g(U), g(tau)
    slc = [g(cd.slc_axis), g(cd.slc_rij), g(cd.slc_rik), g(cd.slc_J), g(cd.slc_types)]
    d_eig, d_w, d_mass, d_mom = g(cd.eig_phon), g(cd.w_phon), g(cd.atom_masses), g(cd.mag_moments)
    ref = Ref(cd)
    M = a.cases
    Nn = cd.N
    ok_ph = [(cd.w_phon[i] > 3.0).nonzero()[0] for i in range(Nn)]
    valid = np.array([i for i in range(Nn) if len(ok_ph[i]) > 0])
    # random processes: grid points k (rows), phonon points
    r1, r2, iph = rng.integers(0, Nn, M), rng.integers(0, Nn, M), rng.choice(valid, M)
    lam = np.array([rng.choice(ok_ph[i]) for i in iph], dtype=np.int32)
    nu, nup = rng.integers(0, nb, M).astype(np.int32), rng.integers(0, nb, M).astype(np.int32)
    c1, c2 = cd.q_grid_cart[r1], cd.q_grid_cart[r2]
    # literal labels: first magnon = table point + a random reciprocal vector (umklapp), phonon label q arbitrary cartesian
    Gs = (rng.integers(-1, 2, (M, 3))) @ B
    kp = c1 + Gs
    q = cd.q_grid_cart[iph]
    d = lambda x, t=None: cuda.to_device(np.ascontiguousarray(x if t is None else x.astype(t)))
    # second table row must correspond to the label kp - q: use the grid point nearest to it (modulo G)
    sec = kp - q
    fr = (sec @ np.linalg.inv(B)) % 1.0
    grid_frac = cd.q_grid / cd.mesh
    row2 = np.array([int(np.argmin(np.sum(((grid_frac - f + 0.5) % 1.0 - 0.5) ** 2, axis=1))) for f in fr], dtype=np.int32)
    c2 = cd.q_grid_cart[row2]

    def dev_band(kp_, q_, nu_, nup_, r1_, c1_, r2_, c2_):
        out = d(np.zeros(M, np.float64))
        k_band[(M + 63) // 64, 64](d(kp_), d(q_), d(lam, np.int32), d(nu_, np.int32), d(nup_, np.int32), d(r1_, np.int32), d(c1_), d(r2_, np.int32), d(c2_), d(iph, np.int32),
                                   d_U, d_tau, *slc, d_eig, d_w, d_mass, d_mom, out)
        return out.copy_to_host()

    # B. vertex against numpy reference
    if nb > 1:
        V2 = dev_band(kp, q, nu, nup, r1, c1, row2, c2, )
        Vr = np.zeros(M)
        for i in range(M):
            e, om = cd.eig_phon[iph[i], lam[i]], cd.w_phon[iph[i], lam[i]]
            a1 = np.array([np.conj(U[r1[i], n, nu[i]]) * np.conj(gauge(kp[i] - c1[i], tau, n)) for n in range(nb)])
            a2 = np.array([U[row2[i], m, nup[i]] * gauge(kp[i] - q[i] - c2[i], tau, m) for m in range(nb)])
            V = sum(ref.V_cons(kp[i], q[i], e, om, n, m) * a1[n] * a2[m] for n in range(nb) for m in range(nb))
            Vr[i] = abs(V) ** 2
        rel = np.abs(V2 - Vr) / np.maximum(Vr, 1e-30 + 1e-9 * Vr.max())
        print(f"B  device vs numpy |V^{{+-}}_{{nu nu'}}|^2 ({M} cases incl. umklapp labels): max rel. dev {rel.max():.2e}, largest value {Vr.max():.3e}")
        assert rel.max() < 1e-8, "band vertex differs from its numpy reference"
        # B2. invariance under a common G shift of the literal labels (tables fixed)
        G2 = (rng.integers(-1, 2, (M, 3))) @ B
        V3 = dev_band(kp + G2, q, nu, nup, r1, c1, row2, c2)
        m = Vr > 1e-3 * Vr.max()
        inv = np.abs(V3[m] - V2[m]) / V2[m]
        print(f"B  |V|^2 under a common G shift of both magnon labels: median change {np.median(inv):.3e}, 90 % {np.percentile(inv, 90):.3e}, max {inv.max():.3e}"
              f"  (small = the umklapp gauge is right; the ~1 % mismatch of the lattice constant of the Jij file sets the floor)")
        assert np.median(inv) < 0.15

    # C. relativistic pair amplitude with a synthetic anisotropic tensor
    if nb > 1:
        Jsave = cd.slc_J.copy()
        cd.slc_J = Jsave.copy()
        rr = 0.5 + rng.random(len(Jsave))
        cd.slc_J[:, 0, 0] += 0.3 * rr; cd.slc_J[:, 1, 1] -= 0.3 * rr; cd.slc_J[:, 0, 1] += 0.2 * rr; cd.slc_J[:, 1, 0] += 0.2 * rr
        slcA = [g(cd.slc_axis), g(cd.slc_rij), g(cd.slc_rik), g(cd.slc_J), g(cd.slc_types)]
        ref2 = Ref(cd)
        for kind in (0, 1):
            ra, rb, ip = rng.integers(0, Nn, M), rng.integers(0, Nn, M), rng.choice(valid, M)
            lm = np.array([rng.choice(ok_ph[i]) for i in ip], dtype=np.int32)
            nu_, nup_ = rng.integers(0, nb, M).astype(np.int32), rng.integers(0, nb, M).astype(np.int32)
            qq = cd.q_grid_cart[ip]
            conj = 0 if kind == 0 else 1
            # labels: k1 + k2 = +q (kind 0) / -q (kind 1) literally
            k1 = cd.q_grid_cart[ra] + (rng.integers(-1, 2, (M, 3))) @ B
            k2 = (qq - k1) if kind == 0 else (-qq - k1)
            ca, cb = cd.q_grid_cart[ra], cd.q_grid_cart[rb]
            out = d(np.zeros(M, np.complex128))
            k_pair[(M + 63) // 64, 64](kind, d(k1), d(k2), d(qq), conj, d(lm, np.int32), d(nu_, np.int32), d(nup_, np.int32), d(ra, np.int32), d(ca), d(rb, np.int32), d(cb), d(ip, np.int32),
                                       d_U, d_tau, *slcA, d_eig, d_w, d_mass, d_mom, out)
            Ad = out.copy_to_host()
            Ar = np.zeros(M, complex)
            for i in range(M):
                e, om = cd.eig_phon[ip[i], lm[i]], cd.w_phon[ip[i], lm[i]]
                f1 = lambda n: (np.conj if kind == 0 else (lambda x: x))(U[ra[i], n, nu_[i]] * gauge(k1[i] - ca[i], tau, n))
                f2 = lambda n: (np.conj if kind == 0 else (lambda x: x))(U[rb[i], n, nup_[i]] * gauge(k2[i] - cb[i], tau, n))
                tot = 0j
                for n in range(nb):
                    for m in range(nb):
                        tot += ref2.V_nc(kind, k1[i], qq[i], conj, e, om, n, m) * f1(n) * f2(m)
                        tot += ref2.V_nc(kind, k2[i], qq[i], conj, e, om, n, m) * f2(n) * f1(m)
                Ar[i] = tot
            rel = np.abs(Ad - Ar) / np.maximum(np.abs(Ar), 1e-9 * np.abs(Ar).max())
            print(f"C  kind {kind}: device vs numpy pair amplitude A_{{nu nu'}}: max rel. dev {rel.max():.2e}, max |A| {np.abs(Ar).max():.3e}")
            assert rel.max() < 1e-8, "pair amplitude differs from its numpy reference"
            # exchange symmetry A_{nu nu'}(k1,k2) = A_{nu' nu}(k2,k1)
            out2 = d(np.zeros(M, np.complex128))
            k_pair[(M + 63) // 64, 64](kind, d(k2), d(k1), d(qq), conj, d(lm, np.int32), d(nup_, np.int32), d(nu_, np.int32), d(rb, np.int32), d(cb), d(ra, np.int32), d(ca), d(ip, np.int32),
                                       d_U, d_tau, *slcA, d_eig, d_w, d_mass, d_mom, out2)
            A2 = out2.copy_to_host()
            sym = np.abs(A2 - Ad) / np.maximum(np.abs(Ad), 1e-9 * np.abs(Ad).max())
            print(f"C  kind {kind}: exchange symmetry A_{{nu nu'}}(k1,k2) = A_{{nu' nu}}(k2,k1): max rel. dev {sym.max():.2e}")
            assert sym.max() < 1e-9
        cd.slc_J = Jsave

    # D. one-band identity
    U1 = np.ones((Nn, 1, 1), complex)
    d_U1, d_tau1 = g(U1), g(np.zeros((1, 3)))
    lam1 = lam; kp1 = kp; q1 = q
    out_a, out_b = d(np.zeros(M)), d(np.zeros(M))
    # a one-sublattice crystal is only emulated: compare calc_vertex_V_path (sublattice 0,0) with the band function on the first sublattice of the CURRENT crystal
    # with U = identity restricted to sublattice 0 (mag_U1.shape[1] = 1)
    k_old[(M + 63) // 64, 64](d(kp1), d(q1), d(lam1, np.int32), d(np.zeros(M), np.int32), d(np.zeros(M), np.int32), d(iph, np.int32), *slc, d_eig, d_w, d_mass, d_mom, out_a)
    out = d(np.zeros(M))
    k_band[(M + 63) // 64, 64](d(kp1), d(q1), d(lam1, np.int32), d(np.zeros(M), np.int32), d(np.zeros(M), np.int32), d(r1, np.int32), d(c1), d(r2 if False else row2, np.int32), d(c2), d(iph, np.int32),
                               d_U1, d_tau1, *slc, d_eig, d_w, d_mass, d_mom, out)
    va, vb = out_a.copy_to_host(), out.copy_to_host()
    print(f"D  one-band U = 1: |V|^2 of the band function vs calc_vertex_V_path: max abs dev {np.abs(va - vb).max():.2e} (max {va.max():.3e})")
    assert np.array_equal(va, vb)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
