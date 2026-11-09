"""
GPU kernels and host glue for three-phonon (phonon-phonon) scattering inside SpinPhony's bare dynamics (--phph).

Physics: theory/ph_ph_scattering.tex, validated against phono3py (theory/fc3_bccFe_data_notes.md, python/phph.py).
One-atom cells only (bcc Fe). Third-order force constants: phono3py fc3.hdf5 (+ phono3py.yaml with the supercell positions).

Event list (grid, used by the dynamics): ordered triples  a=(q,nu)  ->  b=(k,lam) + c=(p=q-k,lam')  with  w = |V|^2 delta
(|V|^2 = mean over the three reference slots of the lattice phases, see python/phph.py; delta: Gaussian of width
max(adaptive, min_sigma), normalised over +-2 sigma like every other bare scan). Both orderings of (b,c) are present, every event
updates the three populations by
        R = 1/2 (2 pi / hbar N) w F,   F = n_a (n_b + 1)(n_c + 1) - (n_a + 1) n_b n_c,     dn_a -= R,  dn_b += R,  dn_c += R .
Linearised: gamma_a = pref w (1 + n_b + n_c),  gamma_b = pref w (n_c - n_a),  gamma_c = pref w (n_b - n_a),  pref = (2 pi/hbar N)/2.
Event list (path, lifetimes only): for a path point q (exact wavevector, partners on the grid, third mode snapped to the grid)
        type 0  q -> k + p       gamma += 1/2 (2 pi/hbar N) w (1 + n_k + n_p)
        type 1  q + k -> s       gamma +=     (2 pi/hbar N) w (n_k - n_s)
Gamma (grid point 0) takes no part as a partner (its numerical omega is not exactly 0) and modes below min_energy are excluded.
"""
import math
import os

import numpy as np
from numba import cuda, complex128

import phph

HBAR = 0.6582119569                       # meV ps
HBAR2_OVER_AMU = 4.180137                 # hbar^2 / amu in meV A^2
EV_TO_MEV = 1.0e3
INV_SQRT8 = 0.35355339059327373
TWO_PI = 2.0 * math.pi
N_SC = 64                                 # atoms in the 4x4x4 supercell of the fc3 calculation


# ----------------------------------------------------------------------------------------------------- device helpers
@cuda.jit(device=True)
def _phi_PT(P, ip, T, it, out):
    """out[x] = sum_j P[ip,j] T[it,j,x]   (x = 3*3*3 Cartesian triple)"""
    for x in range(27):
        out[x] = 0.0 + 0.0j
    for j in range(64):
        pj = P[ip, j]
        for x in range(27):
            out[x] += pj * T[it, j, x]


@cuda.jit(device=True)
def _phi_Pvec_T(pvec, T, it, out):
    """out[abc] = sum_l pvec[l] T[it,l,a c b]   (= sum_{jl} Phi[j,l,abc] P_it[j] pvec[l], Phi[j,l,abc] = Phi[l,j,acb])"""
    for x in range(27):
        out[x] = 0.0 + 0.0j
    for l in range(64):
        pl = pvec[l]
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    out[a * 9 + b * 3 + c] += pl * T[it, l, a * 9 + c * 3 + b]


@cuda.jit(device=True)
def _phi_direct(pa, pb, phi, out):
    """out[x] = sum_{jl} pa[j] pb[l] Phi[j,l,x]"""
    for x in range(27):
        out[x] = 0.0 + 0.0j
    for j in range(64):
        for l in range(64):
            w = pa[j] * pb[l]
            for x in range(27):
                out[x] += w * phi[j, l, x]


@cuda.jit(device=True)
def _contract(Phi, er, es, et):
    """sum_{abc} er[a] es[b] et[c] Phi[a,b,c]"""
    tot = 0.0 + 0.0j
    for a in range(3):
        va = 0.0 + 0.0j
        for b in range(3):
            ub = 0.0 + 0.0j
            for c in range(3):
                ub += Phi[a * 9 + b * 3 + c] * et[c]
            va += ub * es[b]
        tot += va * er[a]
    return tot


@cuda.jit(device=True)
def _phase_vec(fx, fy, fz, img_atom, img_n, img_inv_mult, out):
    """out[j] = <exp(-2 pi i q.n)> over the shortest images of supercell atom j, q = (fx,fy,fz) in the fractional basis."""
    for j in range(64):
        out[j] = 0.0 + 0.0j
    for t in range(img_atom.shape[0]):
        arg = -TWO_PI * (fx * img_n[t, 0] + fy * img_n[t, 1] + fz * img_n[t, 2])
        out[img_atom[t]] += img_inv_mult[t] * (math.cos(arg) + 1j * math.sin(arg))


@cuda.jit(device=True)
def _load_e(eig, i, b, conj, out):
    for c in range(3):
        v = eig[i, b, 0, c]
        out[c] = v.conjugate() if conj else v


@cuda.jit(device=True)
def _V2_three(Pa, Pb, Pc, e1, e2, e3, pref):
    """mean over the three reference slots of |V|^2 given the three Phi~ tensors (27 complex each; layouts as in python/phph.py)."""
    va = _contract(Pa, e1, e2, e3)
    vb = _contract(Pb, e2, e1, e3)
    vc = _contract(Pc, e3, e2, e1)
    s = (va.real * va.real + va.imag * va.imag + vb.real * vb.real + vb.imag * vb.imag + vc.real * vc.real + vc.imag * vc.imag) / 3.0
    return s * pref * pref


# ----------------------------------------------------------------------------------------------------- scan: grid
@cuda.jit
def phph_scan_grid(mesh, q_grid, grid_map, neg_idx, w_phon, grad_f, eig, atom_mass, P, T,
                   base_smearing, min_sigma, e_cut, gamma_idx, store, idx_out, w_out, count):
    """thread = (a, b) grid pair, c = a - b. store == 0: only count the energy-resonant branch triples (upper bound of the event number)."""
    a_idx, b_idx = cuda.grid(2)
    N = q_grid.shape[0]
    if a_idx >= N or b_idx >= N:
        return
    if a_idx == gamma_idx or b_idx == gamma_idx:
        return
    c_idx = grid_map[(q_grid[a_idx, 0] - q_grid[b_idx, 0]) % mesh[0],
                     (q_grid[a_idx, 1] - q_grid[b_idx, 1]) % mesh[1],
                     (q_grid[a_idx, 2] - q_grid[b_idx, 2]) % mesh[2]]
    if c_idx == gamma_idx:
        return
    i1 = neg_idx[a_idx]
    mass = atom_mass[0]

    phi_a = cuda.local.array(27, dtype=complex128)
    phi_b = cuda.local.array(27, dtype=complex128)
    phi_c = cuda.local.array(27, dtype=complex128)
    e1 = cuda.local.array(3, dtype=complex128)
    e2 = cuda.local.array(3, dtype=complex128)
    e3 = cuda.local.array(3, dtype=complex128)
    have_phi = False

    for nu in range(w_phon.shape[1]):
        wa = w_phon[a_idx, nu]
        if wa < e_cut:
            continue
        for lam in range(w_phon.shape[1]):
            wb = w_phon[b_idx, lam]
            if wb < e_cut:
                continue
            for lam2 in range(w_phon.shape[1]):
                wc = w_phon[c_idx, lam2]
                if wc < e_cut:
                    continue
                dE = wa - wb - wc
                var = 0.0
                for i in range(3):
                    d_g = grad_f[a_idx, nu, i] - grad_f[b_idx, lam, i] - grad_f[c_idx, lam2, i]
                    sw = d_g / mesh[i]
                    var += sw * sw
                sig_raw = base_smearing * math.sqrt(var / 12.0)
                sigma = sig_raw if sig_raw > min_sigma else min_sigma
                if abs(dE) >= 2.0 * sigma:
                    continue
                if store == 0:
                    cuda.atomic.add(count, 0, 1)
                    continue
                if not have_phi:
                    _phi_PT(P, b_idx, T, c_idx, phi_a)        # ref = slot 1:  phases on slots 2 (b) and 3 (c)
                    _phi_PT(P, i1, T, c_idx, phi_b)           # ref = slot 2:  phases on slots 1 (-a) and 3 (c)
                    _phi_PT(P, b_idx, T, i1, phi_c)           # ref = slot 3:  phases on slots 2 (b) and 1 (-a)
                    have_phi = True
                _load_e(eig, i1, nu, False, e1)
                _load_e(eig, b_idx, lam, False, e2)
                _load_e(eig, c_idx, lam2, False, e3)
                pref = EV_TO_MEV * INV_SQRT8 * math.sqrt(HBAR2_OVER_AMU / (mass * wa)) * math.sqrt(HBAR2_OVER_AMU / (mass * wb)) \
                    * math.sqrt(HBAR2_OVER_AMU / (mass * wc))
                V2 = _V2_three(phi_a, phi_b, phi_c, e1, e2, e3, pref)
                if V2 > 0.0:
                    c_i = cuda.atomic.add(count, 0, 1)
                    if c_i < idx_out.shape[1]:
                        idx_out[0, c_i] = 0
                        idx_out[1, c_i] = a_idx
                        idx_out[2, c_i] = b_idx
                        idx_out[3, c_i] = c_idx
                        idx_out[4, c_i] = nu
                        idx_out[5, c_i] = lam
                        idx_out[6, c_i] = lam2
                        w_out[c_i] = V2 * (0.4179 / sigma) * math.exp(-0.5 * dE * dE / (sigma * sigma))


# ----------------------------------------------------------------------------------------------------- scan: path
@cuda.jit
def phph_scan_path(mesh, q_grid, grid_map, neg_idx, w_phon, grad_f, eig, atom_mass, P, T, phi,
                   img_atom, img_n, img_inv_mult,
                   path_q_frac, path_w_phon, path_eig,
                   base_smearing, min_sigma, e_cut, gamma_idx, store, idx_out, w_out, count):
    """thread = (path point, grid k). Exact path wavevector in the lattice phases, partner k on the grid, third mode snapped to the
    grid for its energy / eigenvector (its lattice phase uses the exact momentum). The path point's own energy gradient is not
    available: like the other path scans, the Gaussian width uses the partners' gradients only."""
    ip, k_idx = cuda.grid(2)
    N = q_grid.shape[0]
    if ip >= path_q_frac.shape[0] or k_idx >= N or k_idx == gamma_idx:
        return
    mass = atom_mass[0]
    nb = w_phon.shape[1]

    P1 = cuda.local.array(64, dtype=complex128)      # P(-q exact)
    P3 = cuda.local.array(64, dtype=complex128)      # P(slot 3, exact)
    phi_a = cuda.local.array(27, dtype=complex128)
    phi_b = cuda.local.array(27, dtype=complex128)
    phi_c = cuda.local.array(27, dtype=complex128)
    e1 = cuda.local.array(3, dtype=complex128)
    e2 = cuda.local.array(3, dtype=complex128)
    e3 = cuda.local.array(3, dtype=complex128)
    qf0 = path_q_frac[ip, 0]
    qf1 = path_q_frac[ip, 1]
    qf2 = path_q_frac[ip, 2]
    kf0 = q_grid[k_idx, 0] / mesh[0]
    kf1 = q_grid[k_idx, 1] / mesh[1]
    kf2 = q_grid[k_idx, 2] / mesh[2]
    p_computed = False
    for typ in range(2):
        # slot 3 momentum: p = q - k (decay) or s = q + k (coalescence); slot 2: k or -k
        sgn = -1.0 if typ == 0 else 1.0
        f0 = qf0 + sgn * kf0
        f1 = qf1 + sgn * kf1
        f2 = qf2 + sgn * kf2
        i3 = grid_map[int(math.floor((f0 - math.floor(f0)) * mesh[0] + 0.5)) % mesh[0],
                      int(math.floor((f1 - math.floor(f1)) * mesh[1] + 0.5)) % mesh[1],
                      int(math.floor((f2 - math.floor(f2)) * mesh[2] + 0.5)) % mesh[2]]
        if i3 == gamma_idx:
            continue
        i2 = k_idx if typ == 0 else neg_idx[k_idx]
        have_phi = False
        for nu in range(nb):
            wq = path_w_phon[ip, nu]
            if wq < e_cut:
                continue
            for lam in range(nb):
                wk = w_phon[k_idx, lam]
                if wk < e_cut:
                    continue
                for lam2 in range(nb):
                    w3 = w_phon[i3, lam2]
                    if w3 < e_cut:
                        continue
                    if typ == 0:
                        dE = wq - wk - w3
                    else:
                        dE = wq + wk - w3
                    var = 0.0
                    for i in range(3):
                        if typ == 0:
                            d_g = -grad_f[k_idx, lam, i] - grad_f[i3, lam2, i]
                        else:
                            d_g = grad_f[k_idx, lam, i] - grad_f[i3, lam2, i]
                        sw = d_g / mesh[i]
                        var += sw * sw
                    sig_raw = base_smearing * math.sqrt(var / 12.0)
                    sigma = sig_raw if sig_raw > min_sigma else min_sigma
                    if abs(dE) >= 2.0 * sigma:
                        continue
                    if store == 0:
                        cuda.atomic.add(count, 0, 1)
                        continue
                    if not have_phi:
                        if not p_computed:
                            _phase_vec(qf0, qf1, qf2, img_atom, img_n, img_inv_mult, P1)
                            for j in range(64):
                                P1[j] = P1[j].conjugate()
                            p_computed = True
                        _phase_vec(f0, f1, f2, img_atom, img_n, img_inv_mult, P3)
                        _phi_Pvec_T(P3, T, i2, phi_a)             # ref = slot 1 (-q):  phases slot 2 (grid), slot 3 (exact)
                        _phi_direct(P1, P3, phi, phi_b)           # ref = slot 2:       phases slot 1 (-q), slot 3
                        _phi_Pvec_T(P1, T, i2, phi_c)             # ref = slot 3:       phases slot 2, slot 1
                        have_phi = True
                    _load_e(path_eig, ip, nu, True, e1)           # e(-q) = conj e(q)
                    _load_e(eig, i2, lam, False, e2)
                    _load_e(eig, i3, lam2, False, e3)
                    pref = EV_TO_MEV * INV_SQRT8 * math.sqrt(HBAR2_OVER_AMU / (mass * wq)) * math.sqrt(HBAR2_OVER_AMU / (mass * wk)) \
                        * math.sqrt(HBAR2_OVER_AMU / (mass * w3))
                    V2 = _V2_three(phi_a, phi_b, phi_c, e1, e2, e3, pref)
                    if V2 > 0.0:
                        c_i = cuda.atomic.add(count, 0, 1)
                        if c_i < idx_out.shape[1]:
                            idx_out[0, c_i] = typ
                            idx_out[1, c_i] = ip
                            idx_out[2, c_i] = k_idx
                            idx_out[3, c_i] = i3
                            idx_out[4, c_i] = nu
                            idx_out[5, c_i] = lam
                            idx_out[6, c_i] = lam2
                            w_out[c_i] = V2 * (0.4179 / sigma) * math.exp(-0.5 * dE * dE / (sigma * sigma))


# ----------------------------------------------------------------------------------------------------- time step / rates
@cuda.jit
def phph_time_step(idx_in, w_in, count, n_phon, dn_phon, N_points):
    i = cuda.grid(1)
    if i >= count[0] or i >= w_in.shape[0]:
        return
    a = idx_in[1, i]
    b = idx_in[2, i]
    c = idx_in[3, i]
    nu = idx_in[4, i]
    lam = idx_in[5, i]
    lam2 = idx_in[6, i]
    pref = 0.5 * (TWO_PI / HBAR) / N_points
    na = n_phon[a, nu]
    nb_ = n_phon[b, lam]
    nc_ = n_phon[c, lam2]
    R = pref * w_in[i] * (na * nb_ + na * nc_ + na - nb_ * nc_)         # = n_a (n_b+1)(n_c+1) - (n_a+1) n_b n_c
    nbr = n_phon.shape[1]
    cuda.atomic.add(dn_phon, a * nbr + nu, -R)
    cuda.atomic.add(dn_phon, b * nbr + lam, R)
    cuda.atomic.add(dn_phon, c * nbr + lam2, R)


@cuda.jit
def phph_gamma_grid(idx_in, w_in, count, n_phon, gamma_phon, N_points):
    """linearised relaxation rates of all grid phonon modes from the very events the dynamics integrates"""
    i = cuda.grid(1)
    if i >= count[0] or i >= w_in.shape[0]:
        return
    a = idx_in[1, i]
    b = idx_in[2, i]
    c = idx_in[3, i]
    nu = idx_in[4, i]
    lam = idx_in[5, i]
    lam2 = idx_in[6, i]
    pref = 0.5 * (TWO_PI / HBAR) / N_points * w_in[i]
    na = n_phon[a, nu]
    nb_ = n_phon[b, lam]
    nc_ = n_phon[c, lam2]
    nbr = n_phon.shape[1]
    cuda.atomic.add(gamma_phon, a * nbr + nu, pref * (1.0 + nb_ + nc_))
    cuda.atomic.add(gamma_phon, b * nbr + lam, pref * (nc_ - na))
    cuda.atomic.add(gamma_phon, c * nbr + lam2, pref * (nb_ - na))


@cuda.jit
def phph_gamma_path(idx_in, w_in, count, n_phon_grid, gamma_path, N_points):
    i = cuda.grid(1)
    if i >= count[0] or i >= w_in.shape[0]:
        return
    typ = idx_in[0, i]
    ip = idx_in[1, i]
    k = idx_in[2, i]
    p = idx_in[3, i]
    nu = idx_in[4, i]
    lam = idx_in[5, i]
    lam2 = idx_in[6, i]
    fgr = (TWO_PI / HBAR) / N_points
    nk = n_phon_grid[k, lam]
    npp = n_phon_grid[p, lam2]
    nbr = n_phon_grid.shape[1]
    if typ == 0:
        r = 0.5 * fgr * w_in[i] * (1.0 + nk + npp)
    else:
        r = fgr * w_in[i] * (nk - npp)
    cuda.atomic.add(gamma_path, ip * nbr + nu, r)


# ----------------------------------------------------------------------------------------------------- host glue
class PhPhGPU:
    """Tables of the lattice Fourier transform on the device and the scan drivers."""

    def __init__(self, fc3_file, crystal_data, min_energy):
        if crystal_data.l_atoms != 1:
            raise SystemExit(f"--phph is implemented for one-atom primitive cells (this mesh has {crystal_data.l_atoms} atoms).")
        ph3_dir = os.path.dirname(os.path.abspath(fc3_file))
        for need in ("fc3.hdf5", "phono3py.yaml"):
            if not os.path.exists(os.path.join(ph3_dir, need)):
                raise SystemExit(f"--phph: {need} not found next to the fc3 file ({ph3_dir}); the supercell geometry is read from phono3py.yaml.")
        A = np.linalg.inv(np.asarray(crystal_data.reciprocal_lattice, float)).T          # rows: primitive vectors of the mesh file
        self.fc = phph.Fc3Bcc(ph3_dir, A, need_fc2=False)
        if abs(self.fc.mass - float(crystal_data.atom_masses[0])) > 1e-3:
            print(f" -> phph: warning, mass in the fc3 data ({self.fc.mass}) differs from the phonon mesh file ({crystal_data.atom_masses[0]})")
        self.min_energy = float(min_energy)
        self.N = crystal_data.N
        mesh = np.asarray(crystal_data.mesh, int)
        qf = crystal_data.q_grid / mesh
        P = self.fc.phase(qf)
        phi_l = np.transpose(self.fc.phi, (1, 0, 2, 3, 4)).reshape(64, 64 * 27)
        T = (P @ phi_l).reshape(self.N, 64, 27)
        neg = np.zeros(self.N, dtype=np.int32)
        for i in range(self.N):
            neg[i] = crystal_data.grid_map[tuple((-crystal_data.q_grid[i]) % mesh)]
        img_atom = np.concatenate([np.full(len(im), j, dtype=np.int32) for j, im in enumerate(self.fc.images)])
        img_n = np.concatenate(self.fc.images).astype(np.float64)
        img_inv_mult = np.concatenate([np.full(len(im), 1.0 / len(im)) for im in self.fc.images])
        self.d_P, self.d_T = cuda.to_device(P), cuda.to_device(T)
        self.d_phi = cuda.to_device(np.ascontiguousarray(self.fc.phi.reshape(64, 64, 27)))
        self.d_neg = cuda.to_device(neg)
        self.d_img_atom, self.d_img_n, self.d_img_inv_mult = cuda.to_device(img_atom), cuda.to_device(img_n), cuda.to_device(img_inv_mult)
        print(f" -> phph: fc3 from {fc3_file}  (a_conv = {self.fc.a_ph3:.4f} A, {int(self.fc.mult.sum())} shortest images of 64 supercell atoms, "
              f"tables {T.nbytes / 1e6:.0f} MB)")

    # -- grid events (dynamics) -------------------------------------------------------------------------------------------------
    def scan_grid(self, gpu_data, smearing, min_sigma, gamma_idx, blocks, threads, pad=1.02):
        """returns (d_idx, d_w, n_events): count pass (energy conservation only) -> exact allocation -> store pass."""
        d_count = cuda.to_device(np.zeros(1, dtype=np.int64))
        dummy_i, dummy_w = cuda.device_array((7, 1), dtype=np.int32), cuda.device_array(1, dtype=np.float64)
        args = (gpu_data["mesh"], gpu_data["q_grid"], gpu_data["grid_map"], self.d_neg, gpu_data["w_phon"], gpu_data["grad_f_phon"],
                gpu_data["eig_phon"], gpu_data["atom_masses"], self.d_P, self.d_T, smearing, min_sigma, self.min_energy, gamma_idx)
        phph_scan_grid[blocks, threads](*args, 0, dummy_i, dummy_w, d_count)
        cuda.synchronize()
        n_res = int(d_count.copy_to_host()[0])
        cap = max(int(n_res * pad) + 16, 16)
        d_idx, d_w = cuda.device_array((7, cap), dtype=np.int32), cuda.device_array(cap, dtype=np.float64)
        d_count = cuda.to_device(np.zeros(1, dtype=np.int64))
        phph_scan_grid[blocks, threads](*args, 1, d_idx, d_w, d_count)
        cuda.synchronize()
        n = int(d_count.copy_to_host()[0])
        if n > cap:
            raise SystemExit(f"phph event buffer overflow ({n:,} > {cap:,}); this is a bug (the count pass is an upper bound).")
        return d_idx[:, :n], d_w[:n], d_count, n, n_res

    # -- path events (lifetimes) -----------------------------------------------------------------------------------------------
    def scan_path(self, gpu_data, d_path_q_frac, d_path_w_phon, d_path_eig_phon, smearing, min_sigma, gamma_idx, blocks, threads, pad=1.02):
        d_count = cuda.to_device(np.zeros(1, dtype=np.int64))
        dummy_i, dummy_w = cuda.device_array((7, 1), dtype=np.int32), cuda.device_array(1, dtype=np.float64)
        args = (gpu_data["mesh"], gpu_data["q_grid"], gpu_data["grid_map"], self.d_neg, gpu_data["w_phon"], gpu_data["grad_f_phon"],
                gpu_data["eig_phon"], gpu_data["atom_masses"], self.d_P, self.d_T, self.d_phi,
                self.d_img_atom, self.d_img_n, self.d_img_inv_mult, d_path_q_frac, d_path_w_phon, d_path_eig_phon,
                smearing, min_sigma, self.min_energy, gamma_idx)
        phph_scan_path[blocks, threads](*args, 0, dummy_i, dummy_w, d_count)
        cuda.synchronize()
        n_res = int(d_count.copy_to_host()[0])
        cap = max(int(n_res * pad) + 16, 16)
        d_idx, d_w = cuda.device_array((7, cap), dtype=np.int32), cuda.device_array(cap, dtype=np.float64)
        d_count = cuda.to_device(np.zeros(1, dtype=np.int64))
        phph_scan_path[blocks, threads](*args, 1, d_idx, d_w, d_count)
        cuda.synchronize()
        n = int(d_count.copy_to_host()[0])
        if n > cap:
            raise SystemExit(f"phph path event buffer overflow ({n:,} > {cap:,}); this is a bug.")
        return d_idx[:, :n], d_w[:n], d_count, n, n_res
