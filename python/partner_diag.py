"""
Partner-magnon diagnostic of the bare path lifetimes: which magnons take part in the scattering events that make up each rate, and how
much of G_mp every magnon mode carries.   (enabled by --partner_diag of SpinPhony.py)

This module only READS the channel list (chan_indices / chan_weights) and the occupations that phase_lifetime_path has already used; it
does not modify them and it is not part of the rate calculation. The rate of every event is re-evaluated with exactly the expressions
of phase_lifetime_path (c_type 0, 2, 3, 4), summed per row, and compared with the rates of the lifetime kernel (printed as a check;
the deviation is round-off only).

Events (see phase_1_scan_path / phase_1_scan_path_rel), magnon energies from the BZ grid unless marked 'path':
  c_type 0  magnon row (path,n):   M(path,n) <-> M(k,m) + Ph               magnons: E_path(n) [path], E(k,m)
  c_type 2  phonon row (path,lam): Ph + M(p,m) <-> M(k,n)                  magnons: E(k,n) [initial], E(p,m) [final]
  c_type 3  phonon row (path,lam): Ph <-> M(k,n) + M(p,m)   (relativistic) magnons: E(k,n), E(p,m)
  c_type 4  magnon row (path,n):   M(path,n) + M(k,m) <-> Ph   (relativistic) magnons: E_path(n) [path], E(k,m)
For every event the two magnon energies are sorted into E_hi >= E_lo.

Output (companion files, same row order as path_lifetimes.csv: all magnon rows, then all phonon rows):
  path_partner_magnons.csv   one row per lifetime of path_lifetimes.csv: q_idx, particle, branch, energy_meV, then
      rate_exch_events_ps-1, mag_E_hi_mean_meV, mag_E_lo_mean_meV, mag_E_hi_std_meV, mag_E_lo_std_meV   (number-conserving events, rate-weighted)
      rate_rel_events_ps-1, mag_E_hi_mean_rel_meV, mag_E_lo_mean_rel_meV                                   (relativistic events)
      Gmp_row_meV_ps-1_K-1   (phonon rows: E_ph gamma dn/dT / N_q, the contribution of this phonon mode to G_mp per cell; 0 for magnon rows)
      Means are rate-weighted over the contributing events; empty when the row has no event.
  magnon_Gmp_contrib.csv     one row per magnon mode of the BZ grid (k, branch): grid q (fractional), energy, and the share of
      G_mp [meV ps^-1 K^-1 per cell] carried by that mode in the number-conserving (G_exch) and relativistic (G_rel) channels. Every event
      contributes rate_event * E_ph dn/dT / N_q, shared 50/50 between its two magnons (the net energy flows to the phonon, so
      there is no unique 'owner'; the split does not matter for energy-resolved sums because the two magnons differ by one
      phonon energy). The sum over all modes is the G_mp of the run (phonon rows of path_lifetimes.csv, no energy cut).
      Only meaningful for a full-BZ run (path points = grid points).
SI conversion: G[W m^-3 K^-1] = G[meV ps^-1 K^-1 per cell] * 1.602176634e-10 / (V_cell / A^3 * 1e-30).
"""
import math

import numpy as np
from numba import cuda

HBAR = 0.6582119569          # meV ps, as in phase_lifetime_path
KB = 0.08617333262           # meV / K
NCOL = 8                     # 0 R  1 R E_hi  2 R E_lo  3 R E_hi^2  4 R E_lo^2  | 5 R_rel  6 R_rel E_hi  7 R_rel E_lo


@cuda.jit
def _partner_kernel(chan_indices, chan_weights, num_channels, n_mag_grid, n_phon_grid, w_mag_grid, path_w_mag, N_grid_points,
                    acc_mag, acc_phon, gw_phon, g_mag):
    idx = cuda.grid(1)
    if idx >= num_channels[0] or idx >= chan_weights.shape[0]:
        return
    c_type = chan_indices[0, idx]
    path_idx = chan_indices[1, idx]
    k_idx = chan_indices[2, idx]
    p_idx = chan_indices[3, idx]
    n = chan_indices[4, idx]
    m = chan_indices[5, idx]
    lam = chan_indices[6, idx]
    V_sq = chan_weights[idx]
    pref = (2.0 * math.pi / HBAR) / N_grid_points
    nbm = n_mag_grid.shape[1]
    nbp = n_phon_grid.shape[1]

    if c_type == 0:                      # magnon row: M(path,n) <-> M(k,m) + Ph(p,lam)
        rate = pref * V_sq * (1.0 + n_phon_grid[p_idx, lam] + n_mag_grid[k_idx, m])
        e1 = path_w_mag[path_idx, n]
        e2 = w_mag_grid[k_idx, m]
        row = path_idx * nbm + n
        c0 = 0
    elif c_type == 4:                    # relativistic magnon row
        rate = pref * V_sq * (n_mag_grid[k_idx, m] - n_phon_grid[p_idx, lam])
        e1 = path_w_mag[path_idx, n]
        e2 = w_mag_grid[k_idx, m]
        row = path_idx * nbm + n
        c0 = 5
    elif c_type == 2:                    # phonon row: Ph(path,lam) + M(p,m) <-> M(k,n)
        rate = pref * V_sq * (n_mag_grid[p_idx, m] - n_mag_grid[k_idx, n])
        e1 = w_mag_grid[k_idx, n]
        e2 = w_mag_grid[p_idx, m]
        row = path_idx * nbp + lam
        c0 = 0
    elif c_type == 3:                    # relativistic phonon row: Ph <-> M(k,n) + M(p,m), each unordered pair counted twice -> 1/2
        rate = 0.5 * pref * V_sq * (1.0 + n_mag_grid[k_idx, n] + n_mag_grid[p_idx, m])
        e1 = w_mag_grid[k_idx, n]
        e2 = w_mag_grid[p_idx, m]
        row = path_idx * nbp + lam
        c0 = 5
    else:
        return

    e_hi = e1 if e1 >= e2 else e2
    e_lo = e2 if e1 >= e2 else e1
    if c_type == 0 or c_type == 4:
        acc = acc_mag
    else:
        acc = acc_phon
    cuda.atomic.add(acc, (row, c0), rate)
    cuda.atomic.add(acc, (row, c0 + 1), rate * e_hi)
    cuda.atomic.add(acc, (row, c0 + 2), rate * e_lo)
    if c0 == 0:
        cuda.atomic.add(acc, (row, 3), rate * e_hi * e_hi)
        cuda.atomic.add(acc, (row, 4), rate * e_lo * e_lo)
    if c_type == 2 or c_type == 3:       # share of G_mp: rate * E_ph dn/dT / N_q, half to each magnon of the event
        c = 0.5 * rate * gw_phon[path_idx, lam]
        ch = 0 if c_type == 2 else 1
        cuda.atomic.add(g_mag, (ch, k_idx, n), c)
        cuda.atomic.add(g_mag, (ch, p_idx, m), c)


def dn_dT(E, T):
    x = np.asarray(E, float) / (KB * T)
    out = np.zeros_like(x)
    ok = (x > 0) & (x < 700)
    out[ok] = (x[ok] / T) / (4.0 * np.sinh(0.5 * x[ok]) ** 2)
    return out


def run(chan_indices_active, chan_weights_active, channel_count, d_n_mag, d_n_phon, d_w_mag_grid, d_path_w_mag, path_w_phon, N_points, N_path,
        T_phon, gamma_mag_exch, gamma_mag_rel, gamma_phon_exch, gamma_phon_rel, threads_per_block=256):
    """Evaluate the diagnostic from the (device) channel list. gamma_* are the host arrays (N_path, branches) of the lifetime kernel,
    used only for the consistency check. Returns a dict with the accumulators and the checks."""
    nbm = d_n_mag.shape[1]
    nbp = d_n_phon.shape[1]
    gw = path_w_phon * dn_dT(path_w_phon, T_phon) / float(N_path)          # meV / K  per path phonon, weight of 1/N_q
    d_gw = cuda.to_device(np.ascontiguousarray(gw))
    d_acc_mag = cuda.to_device(np.zeros((N_path * nbm, NCOL)))
    d_acc_phon = cuda.to_device(np.zeros((N_path * nbp, NCOL)))
    d_g_mag = cuda.to_device(np.zeros((2, d_n_mag.shape[0], nbm)))
    n_ch = int(channel_count.copy_to_host()[0]) if hasattr(channel_count, "copy_to_host") else int(channel_count[0])
    n_ch = min(n_ch, chan_weights_active.shape[0])
    blocks = max(1, math.ceil(n_ch / threads_per_block))
    _partner_kernel[blocks, threads_per_block](chan_indices_active, chan_weights_active, channel_count, d_n_mag, d_n_phon, d_w_mag_grid,
                                               d_path_w_mag, N_points, d_acc_mag, d_acc_phon, d_gw, d_g_mag)
    cuda.synchronize()
    acc_mag = d_acc_mag.copy_to_host().reshape(N_path, nbm, NCOL)
    acc_phon = d_acc_phon.copy_to_host().reshape(N_path, nbp, NCOL)
    g_mag = d_g_mag.copy_to_host()

    def rel_dev(a, b):
        scale = np.maximum(np.abs(b), 1e-30)
        m = np.abs(b) > 1e-12
        return float(np.max(np.abs(a - b)[m] / scale[m])) if m.any() else 0.0

    chk = dict(mag_exch=rel_dev(acc_mag[..., 0], gamma_mag_exch), phon_exch=rel_dev(acc_phon[..., 0], gamma_phon_exch),
               mag_rel=rel_dev(acc_mag[..., 5], gamma_mag_rel), phon_rel=rel_dev(acc_phon[..., 5], gamma_phon_rel))
    chk["G_modes_vs_rows_exch"] = float(abs(g_mag[0].sum() - (gw * gamma_phon_exch).sum()) / max((gw * gamma_phon_exch).sum(), 1e-300))
    chk["G_modes_vs_rows_rel"] = float(abs(g_mag[1].sum() - (gw * gamma_phon_rel).sum()) / max((gw * gamma_phon_rel).sum(), 1e-300))
    return dict(acc_mag=acc_mag, acc_phon=acc_phon, g_mag=g_mag, gw=gw, check=chk, n_channels=n_ch)


def _moments(acc, c0, with_std):
    R = acc[..., c0]
    with np.errstate(divide="ignore", invalid="ignore"):
        hi = acc[..., c0 + 1] / R
        lo = acc[..., c0 + 2] / R
        out = [R, hi, lo]
        if with_std:
            out += [np.sqrt(np.maximum(acc[..., 3] / R - hi ** 2, 0.0)), np.sqrt(np.maximum(acc[..., 4] / R - lo ** 2, 0.0))]
    return out


def write_files(res, out_dir, path_q_frac, w_mag_path, w_phon_path, grid_q_frac, w_mag_grid):
    """Write path_partner_magnons.csv (row order of path_lifetimes.csv) and magnon_Gmp_contrib.csv"""
    N_path, nbm, _ = res["acc_mag"].shape
    nbp = res["acc_phon"].shape[1]

    def fmt(x):
        return "" if not np.isfinite(x) else f"{x:.6e}"

    with open(f"{out_dir}/path_partner_magnons.csv", "w") as f:
        f.write("q_idx,particle,branch,energy_meV,rate_exch_events_ps-1,mag_E_hi_mean_meV,mag_E_lo_mean_meV,mag_E_hi_std_meV,mag_E_lo_std_meV,"
                "rate_rel_events_ps-1,mag_E_hi_mean_rel_meV,mag_E_lo_mean_rel_meV,Gmp_row_meV_ps-1_K-1\n")
        for part, acc, nb, wpath in (("magnon", res["acc_mag"], nbm, w_mag_path), ("phonon", res["acc_phon"], nbp, w_phon_path)):
            R, hi, lo, shi, slo = _moments(acc, 0, True)
            Rr, hir, lor = _moments(acc, 5, False)
            for i in range(N_path):
                for b in range(nb):
                    grow = (res["gw"][i, b] * (res["acc_phon"][i, b, 0] + res["acc_phon"][i, b, 5])) if part == "phonon" else 0.0
                    f.write(f"{i},{part},{b},{wpath[i, b]:.6f},{fmt(R[i, b])},{fmt(hi[i, b])},{fmt(lo[i, b])},{fmt(shi[i, b])},{fmt(slo[i, b])},"
                            f"{fmt(Rr[i, b])},{fmt(hir[i, b])},{fmt(lor[i, b])},{grow:.6e}\n")
    with open(f"{out_dir}/magnon_Gmp_contrib.csv", "w") as f:
        f.write("grid_idx,qx_frac,qy_frac,qz_frac,branch,energy_meV,G_exch_meV_ps-1_K-1_per_cell,G_rel_meV_ps-1_K-1_per_cell\n")
        for k in range(w_mag_grid.shape[0]):
            for b in range(w_mag_grid.shape[1]):
                f.write(f"{k},{grid_q_frac[k, 0]:.6f},{grid_q_frac[k, 1]:.6f},{grid_q_frac[k, 2]:.6f},{b},{w_mag_grid[k, b]:.6f},"
                        f"{res['g_mag'][0, k, b]:.6e},{res['g_mag'][1, k, b]:.6e}\n")
