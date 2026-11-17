"""
Independent numpy implementation of the bare (number-conserving) magnon-phonon vertex and RTA rates, written from theory/mag_ph.txt
(Hamiltonian, W^{+-}, V^{+-}) and theory/bare_scattering_rates_plus_relativistic.tex (rates 1.2 and 2.1). It does not import any kernel.

Three vertex bookkeepings are provided, all built from the same J~, e^mu_lambda and hbar^2/(2 M E_ph):

  V_H(K, Q)      direct from the Hamiltonian   H = sum (a_Q + a^+_-Q) V_{K,Q} b^+_K b_{K-Q},  W^{+-}(K,Q) = [J~xx + J~yy - i J~xy + i J~yx](K-Q, Q)/S - 2 J~zz(0,Q)/S
  rates in terms of V_H:
     magnon q -> k + ph(q-k)  and  q + ph(k-q) -> k:      |V_H(K=k, Q=k-q)|^2  (the same a_Q/a^+_-Q vertex for both)
     phonon q absorbed by magnon k-q -> k:                 |V_H(K=k, Q=q)|^2

  tex-literal labels of the relaxation section:
     magnon:  |V(K=q, Q=q-k)|^2  both terms (2.1)      phonon: |V(K=k, Q=-q)|^2 (1.2)

  kernel replica (what phase_1_scan_path hands to calc_vertex_V_path): magnon  V(K=q, Q=k-q);  phonon  V(K=k, Q=-q)
"""
import sys, math
import numpy as np
sys.path.insert(0, "python")
import SpinPhony as sp

HBAR = 0.6582119569
KB = 0.08617333262
DALTON = 0.10364269


class Ref:
    def __init__(self, mesh_h5, T_mag, T_phon, slc=True, material="bccFe"):
        files = ["Inputs/bccFe/Fe_full_tensor_ij-uk_%s_displacement.csv" % c for c in "xyz"]
        self.cd = cd = sp.CrystalDataSoA(mesh_h5, "Inputs/bccFe/Fe_Jij_scaled.csv", slc_files=files, lattice_constant=2.8665, anisotropy=0.005, include_slc=slc)
        self.N = cd.N
        self.mesh = np.asarray(cd.mesh)
        self.qc = cd.q_grid_cart.copy()
        self.q_int = cd.q_grid.copy()
        self.gmap = cd.grid_map
        self.w_ph, self.e_ph, self.w_mg = cd.w_phon, cd.eig_phon[:, :, 0, :], cd.w_mag[:, 0]
        self.g_mag = cd.grad_f_mag[:, 0, :]
        self.S = abs(cd.mag_moments[0] / 2.0)
        self.mass = cd.atom_masses[0] * DALTON
        self.nm = sp.init_bose_einstein(self.w_mg, T_mag)
        self.np_ = sp.init_bose_einstein(self.w_ph, T_phon)
        # SLC tensors per displacement axis mu
        self.slc = []
        for mu in range(3):
            s = cd.slc_axis == mu
            self.slc.append((cd.slc_rij[s], cd.slc_rik[s], cd.slc_J[s]))
        self.gamma_idx = int(self.gmap[0, 0, 0])

    def idx(self, qint):
        return self.gmap[tuple(np.mod(np.asarray(qint), self.mesh).T)]

    def Jt(self, p, q, mu):
        """J~^{ab,mu}(p, q) = sum J^{ab}_{ij,k} exp(i(p.rij + q.rik)), p,q (B,3) cartesian, returns (B,3,3) complex"""
        rij, rik, J = self.slc[mu]
        ph = np.exp(1j * (p @ rij.T + q @ rik.T))                  # (B, M)
        return np.einsum("bm,mxy->bxy", ph, J)

    def V(self, K, Q, iph, lam):
        """V^{+-}_{K,Q,lam} (without the 1/sqrt(N)), K, Q cartesian (B,3); iph: grid index of the phonon (eigenvector e(Q), omega(Q)), lam branch"""
        om = self.w_ph[iph, lam]
        amp = np.sqrt(HBAR ** 2 / (2.0 * self.mass * np.maximum(om, 1e-9)))
        V = 0.0
        for mu in range(3):
            Jd = self.Jt(K - Q, Q, mu)
            Js = self.Jt(np.zeros_like(K), Q, mu)
            W = (Jd[:, 0, 0] + Jd[:, 1, 1] - 1j * Jd[:, 0, 1] + 1j * Jd[:, 1, 0]) / self.S - 2.0 / self.S * Js[:, 2, 2]
            V = V + amp * self.e_ph[iph, lam, mu] * W
        return np.where(om < 0.1, 0.0, np.abs(V) ** 2)

    # ---- delta functions
    def sigma(self, kind, gmag_terms, gph_terms, smearing, floor):
        var = 0.0
        for i in range(3):
            d = sum(gmag_terms[i]) if gmag_terms is not None else 0.0
            var = var + (d / self.mesh[i]) ** 2
        return np.maximum(smearing * np.sqrt(var / 12.0), floor)


def gauss(dE, sg, trunc=True):
    w = 0.3989422804014327 / sg * np.exp(-0.5 * (dE / sg) ** 2)
    return np.where(np.abs(dE) < 2.0 * sg, 0.4179 / sg * np.exp(-0.5 * (dE / sg) ** 2), 0.0) if trunc else w


ECUT = 0.2          # BARE_MIN_SCATTERING_ENERGY (meV)


def _delta(ref, dE, kind, kk, kk2, smearing, sigma_fixed, lam=0):
    """kind 'mag' (magnon channel): d_g = -g_mag[kk] + g_ph[kk2] (g_ph is zero in the code); kind 'phon': d_g = g_mag[kk2] - g_mag[kk]"""
    if sigma_fixed is not None:
        return gauss(dE, sigma_fixed, trunc=False)
    d = (-ref.g_mag[kk] + ref.cd.grad_f_phon[kk2, lam]) if kind == "mag" else (ref.g_mag[kk2] - ref.g_mag[kk])
    var = ((d / ref.mesh) ** 2).sum(-1)
    sg = np.maximum(smearing * np.sqrt(var / 12.0), 1e-5)
    return gauss(dE, sg, trunc=True)


def magnon_rate(ref, iq, version, smearing=0.3, sigma=None, absorb=None):
    """RTA rate (1/ps) of the magnon at grid index iq (decay q -> k + ph, and for tex/ham also the absorption q + ph -> k)."""
    absorb = (version != "kernel") if absorb is None else absorb
    N = ref.N
    k = np.arange(N)
    ok0 = (k != ref.gamma_idx) & (ref.w_mg[k] >= ECUT) & (ref.w_mg[iq] >= ECUT)
    kq = ref.idx(ref.q_int[iq][None] - ref.q_int)          # q - k
    kmq = ref.idx(ref.q_int - ref.q_int[iq][None])         # k - q
    pref = 2 * np.pi / (HBAR * N)
    tot = 0.0
    for lam in range(3):
        w_em = ref.w_ph[kq, lam]
        w_ab = ref.w_ph[kmq, lam]
        qk = ref.qc[iq][None] - ref.qc                      # cartesian q - k
        kq_c = ref.qc - ref.qc[iq][None]                    # cartesian k - q
        Kq = np.broadcast_to(ref.qc[iq], (N, 3))
        # ---- emission
        dE = ref.w_mg[iq] - ref.w_mg - w_em
        if version == "kernel":
            Vv = ref.V(Kq, kq_c, kmq, lam)                  # kp = path, q = k - path, eigenvector at idx_kminq
        elif version == "tex":
            Vv = ref.V(Kq, qk, kq, lam)                     # V_{q, q-k}
        else:
            Vv = ref.V(ref.qc, kq_c, kmq, lam)              # V_{K=k, Q=k-q}
        okE = ok0 & (w_em >= ECUT)
        B = 1.0 + ref.np_[kq, lam] + ref.nm
        tot += (pref * Vv * B * _delta(ref, dE, "mag", k, kq, smearing, sigma, lam) * okE).sum()
        # ---- absorption (the kernels only ship the emission delta)
        if absorb:
            dE2 = ref.w_mg[iq] - ref.w_mg + w_ab
            okA = ok0 & (w_ab >= ECUT)
            B2 = ref.np_[kmq, lam] - ref.nm
            tot += (pref * Vv * B2 * _delta(ref, dE2, "mag", k, kmq, smearing, sigma, lam) * okA).sum()
    return tot


def phonon_rate(ref, iq, lam, version, smearing=0.3, sigma=None):
    """RTA rate (1/ps) of the phonon (iq, lam): absorption by a magnon k-q -> k (minus stimulated emission)."""
    N = ref.N
    k = np.arange(N)
    kmq = ref.idx(ref.q_int - ref.q_int[iq][None])         # k - q
    Eq = ref.w_ph[iq, lam]
    dE = Eq + ref.w_mg[kmq] - ref.w_mg
    ok = (k != ref.gamma_idx) & (ref.w_mg >= ECUT) & (ref.w_mg[kmq] >= ECUT) & (Eq >= ECUT)
    sgn = -1.0 if version in ("kernel", "tex") else 1.0
    Qc = np.broadcast_to(sgn * ref.qc[iq], (N, 3))
    Vv = ref.V(ref.qc, Qc, np.full(N, iq), lam)
    return (2 * np.pi / (HBAR * N) * Vv * (ref.nm[kmq] - ref.nm) * _delta(ref, dE, "phon", k, kmq, smearing, sigma) * ok).sum()
