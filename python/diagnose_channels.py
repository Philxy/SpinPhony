"""
Per-channel breakdown of the hybrid path scattering rate.

The lifetime kernel collapses every allowed channel into a single number per
mode, which hides *where* the rate comes from. This reads the same channel
list the kernel consumes and decomposes 1/tau for a chosen mode by

  * process type   (splitting vs coalescence)
  * partner bands  (which branches the two other legs sit in)
  * |q| shell      (how far from Gamma the partner momentum is)

Use in main(), after phase_1_scan_hybrid_path and before the buffers are
reused:

    from diagnose_channels import breakdown
    breakdown(crystal_data,
              d_path_hyb_chan_indices, d_path_hyb_chan_weights,
              path_hyb_num_channels, n_hyb_cpu, N_points,
              path_idx=..., band=...)
"""
import numpy as np

HBAR = 0.6582119569  # meV * ps


def pick_modes(crystal_data, band, n=2, e_min=1.0):
    """
    Returns path indices for the lowest- and highest-energy modes of `band`.

    In --full_bz mode path_idx is only a row index into the mesh, which is NOT
    ordered by energy, so guessing an index gives an arbitrary point. Modes
    below e_min are skipped (near-Gamma numerical noise).
    """
    w = crystal_data.path_w_hyb[:, band]
    ok = np.where(w > e_min)[0]
    if ok.size == 0:
        return []
    order = ok[np.argsort(w[ok])]
    picks = list(order[:n]) + list(order[-n:])
    for p in picks:
        print(f"[pick_modes] band {band}: path_idx {int(p)} -> {w[p]:.3f} meV")
    return [int(p) for p in picks]


def path_observables(crystal_data):
    """
    (phon_char, mag_char, phon_AM, spin_AM) for the PATH modes, shape
    (N_path, num_hyb). Same construction as save_hybrid_path_properties,
    including the per-q Cartesian -> branch rotation of L^z. N_path is small,
    so a plain loop is fine here.
    """
    L_tot = crystal_data.get_nambu_angular_momentum(axis=2)
    n_ph = crystal_data.phon_branches
    blk = n_ph + crystal_data.n_mag_branches
    dim = 2 * blk
    N = crystal_data.N_path

    L_spin = np.zeros((dim, dim), dtype=np.complex128)
    L_spin[n_ph:blk, n_ph:blk] = L_tot[n_ph:blk, n_ph:blk]
    L_spin[blk + n_ph:, blk + n_ph:] = L_tot[blk + n_ph:, blk + n_ph:]

    J_ph = np.zeros((dim, dim))
    np.fill_diagonal(J_ph[:n_ph, :n_ph], 1.0)
    np.fill_diagonal(J_ph[blk:blk + n_ph, blk:blk + n_ph], -1.0)
    J_mg = np.zeros((dim, dim))
    np.fill_diagonal(J_mg[n_ph:blk, n_ph:blk], 1.0)
    np.fill_diagonal(J_mg[blk + n_ph:, blk + n_ph:], -1.0)

    L_cart = L_tot[:n_ph, :n_ph]

    phc = np.zeros((N, blk)); mgc = np.zeros((N, blk))
    pam = np.zeros((N, blk)); sam = np.zeros((N, blk))

    for i in range(N):
        T = crystal_data.path_eig_hyb[i]
        Td = T.conj().T
        E_q = crystal_data.path_eig_phon[i].reshape(n_ph, n_ph).T
        Lb = E_q.conj().T @ L_cart @ E_q
        Lp = np.zeros((dim, dim), dtype=np.complex128)
        Lp[:n_ph, :n_ph] = Lb
        Lp[blk:blk + n_ph, blk:blk + n_ph] = -Lb.conj()

        phc[i] = np.diag(Td @ J_ph @ T).real[:blk]
        mgc[i] = np.diag(Td @ J_mg @ T).real[:blk]
        pam[i] = np.diag(Td @ Lp @ T).real[:blk]
        sam[i] = np.diag(Td @ L_spin @ T).real[:blk]

    return phc, mgc, pam, sam


def channel_table(crystal_data, d_chan_indices, d_chan_weights, num_channels,
                  n_hyb, N_grid_points, path_idx, band, grid_props=None,
                  path_props=None, top=20, am_tol=0.15):
    """
    Ranks the INDIVIDUAL scattering channels of one hybrid mode and annotates
    each partner with where it sits in the BZ and what it carries.

    Answers, for a chosen (path_idx, band):
      * which modes does it scatter into, and where in the BZ are they
      * are those partners hybrid, or nearly pure magnon / pure phonon
      * what phonon angular momentum and spin angular momentum do they carry
      * is total J_z = l_z(phonon) + S_z(magnon) conserved in the dominant
        channels, and if not, by how much

    grid_props: the 4-tuple from crystal_data.extract_full_grid_hybrid_properties().
                Pass it in so it is computed once and reused across modes.

    KINEMATICS AT GAMMA (path_q_frac = 0) - worth keeping in mind when reading
    the table:
        splitting    p = round(0 - k) = -k   -> partners sit at +/- k
        coalescence  s = round(0 + k) =  k   -> partner AND product are at the
                                               same k, i.e. a vertical transition
    """
    idx = np.asarray(d_chan_indices[:, :num_channels].copy_to_host())
    wts = np.asarray(d_chan_weights[:num_channels].copy_to_host())
    c_type, p_i, k_i, o_i, b_q, b_k, b_o = (idx[r] for r in range(7))

    sel = (p_i == path_idx) & (b_q == band)
    if not sel.any():
        print(f"[channel_table] no channels for path_idx={path_idx}, band={band}")
        return None

    c_type, k_i, o_i = c_type[sel], k_i[sel], o_i[sel]
    b_k, b_o, wts = b_k[sel], b_o[sel], wts[sel]

    n_k, n_o = n_hyb[k_i, b_k], n_hyb[o_i, b_o]
    pre_split = (np.pi / HBAR) / N_grid_points
    pre_coal = (2.0 * np.pi / HBAR) / N_grid_points
    rate = np.where(c_type == 0,
                    pre_split * wts * (n_k + n_o + 1.0),
                    pre_coal * wts * (n_k - n_o))
    total = rate.sum()

    if grid_props is None:
        grid_props = crystal_data.extract_full_grid_hybrid_properties()
    g_phc, g_mgc, g_pam, g_sam = grid_props

    if path_props is None:
        path_props = path_observables(crystal_data)
    p_phc, p_mgc, p_pam, p_sam = path_props

    # Parent observables come from the PATH arrays, partners from the GRID.
    E_par = crystal_data.path_w_hyb[path_idx, band]
    Jpar = p_pam[path_idx, band] + p_sam[path_idx, band]

    # Folded fractional coordinates of the partners, so |q| is the physical
    # distance from Gamma rather than a zone-corner alias.
    qf = crystal_data.q_grid.astype(np.float64) / crystal_data.mesh
    qf -= np.round(qf)
    qcart = np.dot(qf, crystal_data.reciprocal_lattice * 2.0 * np.pi)
    qmag = np.linalg.norm(qcart, axis=1)

    print(f"\n[channel_table] path_idx={path_idx} band={band}   E = {E_par:.4f} meV")
    print(f"   1/tau = {total:.6e} 1/ps    tau = {1.0 / total if total > 0 else np.inf:.4e} ps"
          f"    {len(rate):,} channels")

    order = np.argsort(-np.abs(rate))[:top]
    cum = 0.0
    print(f"   top {len(order)} channels by |rate|:")
    print("     proc   rate(1/ps)   cum%  |  partner k: frac / |q|      E_k    b_k  magch   l_z    S_z"
          "  |  other: E_o    b_o  magch   l_z    S_z  | dJz")
    for c in order:
        r = rate[c]
        cum += r
        ki, oi = int(k_i[c]), int(o_i[c])
        bk, bo = int(b_k[c]), int(b_o[c])

        # J_z = phonon orbital + magnon spin, for each of the three legs.
        Jk = g_pam[ki, bk] + g_sam[ki, bk]
        Jo = g_pam[oi, bo] + g_sam[oi, bo]

        if c_type[c] == 0:      # splitting: parent -> k + o
            dJ = Jpar - Jk - Jo
            proc = "split"
        else:                   # coalescence: parent + k -> o
            dJ = Jpar + Jk - Jo
            proc = "coal "

        print(f"     {proc} {r:+.3e} {cum / total * 100:5.1f}  |  "
              f"({qf[ki, 0]:+.3f},{qf[ki, 1]:+.3f},{qf[ki, 2]:+.3f}) {qmag[ki]:.3f}  "
              f"{crystal_data.w_hyb[ki, bk]:7.3f} {bk:3d} {g_mgc[ki, bk]:6.3f} "
              f"{g_pam[ki, bk]:+6.3f} {g_sam[ki, bk]:+6.3f}  |  "
              f"{crystal_data.w_hyb[oi, bo]:7.3f} {bo:3d} {g_mgc[oi, bo]:6.3f} "
              f"{g_pam[oi, bo]:+6.3f} {g_sam[oi, bo]:+6.3f}  | {dJ:+6.3f}")

    # --- aggregate: is the partner hybrid or nearly pure? ----------------
    print("   rate by partner magnon character (k-leg):")
    edges = np.array([0.0, 0.05, 0.25, 0.75, 0.95, 1.0 + 1e-9])
    names = ["pure phonon", "phonon-like", "HYBRID", "magnon-like", "pure magnon"]
    for a, b, nm in zip(edges[:-1], edges[1:], names):
        m = (g_mgc[k_i, b_k] >= a) & (g_mgc[k_i, b_k] < b)
        if m.any():
            print(f"      {nm:>12s} [{a:.2f},{b:.2f})  {rate[m].sum():+.4e}  "
                  f"({rate[m].sum() / total * 100:6.2f}%)  {int(m.sum()):,} ch")

    # --- aggregate: chirality of the partners ----------------------------
    print("   rate by partner phonon angular momentum (k-leg):")
    for lo, hi, nm in ((-np.inf, -am_tol, "l_z < 0"),
                       (-am_tol, am_tol, "l_z ~ 0"),
                       (am_tol, np.inf, "l_z > 0")):
        m = (g_pam[k_i, b_k] > lo) & (g_pam[k_i, b_k] <= hi)
        if m.any():
            print(f"      {nm:>8s}  {rate[m].sum():+.4e}  "
                  f"({rate[m].sum() / total * 100:6.2f}%)  {int(m.sum()):,} ch")

    # --- angular momentum balance ----------------------------------------
    Jk_all = g_pam[k_i, b_k] + g_sam[k_i, b_k]
    Jo_all = g_pam[o_i, b_o] + g_sam[o_i, b_o]
    dJ_all = np.where(c_type == 0, Jpar - Jk_all - Jo_all,
                      Jpar + Jk_all - Jo_all)

    print("   rate-weighted J_z balance  (dJz = J_parent -/+ J_k - J_other):")
    print(f"      mean {np.average(dJ_all, weights=np.abs(rate)):+.4f} hbar,"
          f"  |dJz| < {am_tol}: "
          f"{np.abs(rate[np.abs(dJ_all) < am_tol]).sum() / np.abs(rate).sum() * 100:.1f}% of |rate|")
    for lo, hi in ((-0.5, 0.5), (2.5, 3.5), (-3.5, -2.5)):
        m = (dJ_all > lo) & (dJ_all <= hi)
        if m.any():
            print(f"      dJz in ({lo:+.1f},{hi:+.1f}]  {rate[m].sum():+.4e}  "
                  f"({rate[m].sum() / total * 100:6.2f}%)  {int(m.sum()):,} ch")

    return total


def breakdown(crystal_data, d_chan_indices, d_chan_weights, num_channels,
              n_hyb, N_grid_points, path_idx, band, n_shells=6, top=12):
    """
    Reproduces phase_lifetime_hybrid_path on the host for one (path_idx, band)
    and reports how the total rate is distributed. Rates are in 1/ps and sum to
    the same 1/tau the GPU wrote out, so any disagreement is itself a finding.
    """
    idx = np.asarray(d_chan_indices[:, :num_channels].copy_to_host())
    wts = np.asarray(d_chan_weights[:num_channels].copy_to_host())

    c_type, p_i, k_i, o_i, b_q, b_k, b_o = (idx[r] for r in range(7))

    sel = (p_i == path_idx) & (b_q == band)
    if not sel.any():
        print(f"[breakdown] no channels for path_idx={path_idx}, band={band}")
        return

    c_type, k_i, o_i = c_type[sel], k_i[sel], o_i[sel]
    b_k, b_o, wts = b_k[sel], b_o[sel], wts[sel]

    n_k = n_hyb[k_i, b_k]
    n_o = n_hyb[o_i, b_o]

    pre_split = (np.pi / HBAR) / N_grid_points
    pre_coal = (2.0 * np.pi / HBAR) / N_grid_points

    rate = np.where(c_type == 0,
                    pre_split * wts * (n_k + n_o + 1.0),
                    pre_coal * wts * (n_k - n_o))

    total = rate.sum()
    energy = crystal_data.path_w_hyb[path_idx, band]
    print(f"\n[breakdown] path_idx={path_idx} band={band}  "
          f"E = {energy:.3f} meV   1/tau = {total:.6e} 1/ps   "
          f"tau = {1.0 / total if total > 0 else np.inf:.4e} ps")
    print(f"            {len(rate):,} channels")

    # --- process type ---------------------------------------------------
    for t, name in ((0, "splitting  "), (1, "coalescence")):
        m = c_type == t
        if m.any():
            print(f"   {name}: {rate[m].sum():+.4e} 1/ps  "
                  f"({rate[m].sum() / total * 100:6.2f}%)  {m.sum():,} channels")

    # --- partner band pairs, SPLIT BY PROCESS ---------------------------
    # Splitting and coalescence have different symmetry, so they must not be
    # pooled. For SPLITTING the two children are interchangeable: the scan
    # enumerates each unordered final pair twice, once as (b1,b2) and once as
    # (b2,b1), with identical |Gamma~|^2 and identical (1 + n_k + n_p). So the
    # splitting table MUST be symmetric - any asymmetry is a bug.
    # For COALESCENCE (b_k, b_s) are partner and parent, i.e. genuinely
    # different roles, so no symmetry is expected there.
    for t, name in ((0, "splitting"), (1, "coalescence")):
        m = c_type == t
        if not m.any():
            continue
        pairs = {}
        for bk, bo, r in zip(b_k[m], b_o[m], rate[m]):
            pairs[(int(bk), int(bo))] = pairs.get((int(bk), int(bo)), 0.0) + r
        print(f"   {name} by partner bands (b_k, b_other):")
        for (bk, bo), r in sorted(pairs.items(), key=lambda kv: -abs(kv[1]))[:top]:
            print(f"      ({bk:2d},{bo:2d})  {r:+.4e}  ({r / total * 100:6.2f}%)")

        if t == 0:
            worst, wpair = 0.0, None
            for (bk, bo), r in pairs.items():
                if bk == bo:
                    continue
                mirror = pairs.get((bo, bk), 0.0)
                denom = max(abs(r), abs(mirror))
                if denom > 0:
                    asym = abs(r - mirror) / denom
                    if asym > worst:
                        worst, wpair = asym, (bk, bo)
            print(f"      -> splitting symmetry: worst |A-B|/max = {worst:.3e}"
                  + (f" at {wpair}" if wpair else ""))
            if worst > 1e-6:
                print("         BUG: splitting must be symmetric under swapping "
                      "the two children.")

    # --- |q| shells of the grid partner ---------------------------------
    # Folded to [-0.5, 0.5) so |q| is the physical distance from Gamma.
    qf = crystal_data.q_grid[k_i].astype(np.float64) / crystal_data.mesh
    qf -= np.round(qf)
    qcart = np.dot(qf, crystal_data.reciprocal_lattice * 2.0 * np.pi)
    qmag = np.linalg.norm(qcart, axis=1)

    print("   by |q| of the grid partner:")
    edges = np.linspace(0.0, qmag.max() + 1e-12, n_shells + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (qmag >= a) & (qmag < b)
        if m.any():
            print(f"      {a:5.3f}-{b:5.3f} 1/A  {rate[m].sum():+.4e}  "
                  f"({rate[m].sum() / total * 100:6.2f}%)  {m.sum():,} ch")

    return total
