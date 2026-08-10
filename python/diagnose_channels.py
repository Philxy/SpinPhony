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

    # --- partner band pairs ---------------------------------------------
    print("   by partner bands (b_k, b_other), largest |rate| first:")
    pairs = {}
    for bk, bo, r in zip(b_k, b_o, rate):
        pairs[(int(bk), int(bo))] = pairs.get((int(bk), int(bo)), 0.0) + r
    for (bk, bo), r in sorted(pairs.items(), key=lambda kv: -abs(kv[1]))[:top]:
        print(f"      ({bk:2d},{bo:2d})  {r:+.4e}  ({r / total * 100:6.2f}%)")

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
