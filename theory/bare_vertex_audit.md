# Audit of the bare magnon-phonon rates against `theory/mag_ph.txt`

Reference: the Hamiltonian, `W^{+-}`, `V^{+-}` of `theory/mag_ph.txt` and the rates 1.2 (phonons) and 2.1 (magnons) of
`theory/bare_scattering_rates_plus_relativistic.tex` (`mag_ph.txt` itself ends after the phonon rates).
Independent implementation: `theory/checks/magph/ref_bare_rates.py` (numpy, no kernel code). `kernel_vs_ref.py` runs the CUDA kernels and the reference
on the same modes. `--legacy_bare` restores the kernels as they were.

## What the vertex is

`H = sum_{K,Q} (a_Q + a^+_{-Q}) V_{K,Q} b^+_K b_{K-Q}`,  `V_{K,Q} = N^{-1/2} sum_{l,mu} sqrt(hbar^2/(2 M_l E_Q)) e^mu_l(Q) W^{+-,mu}(K,Q)`,
`W^{+-}(K,Q) = [J~xx + J~yy - i J~xy + i J~yx](K-Q, Q)/S  -  (2/S) J~zz(0,Q)`.
The first Fourier argument is the momentum of the ANNIHILATED magnon (phase `exp(i p_j . r_ij)`), the second is the phonon momentum `Q` of `a_Q`
(`a^+_{-Q}` carries the same `Q`). For the process  magnon `a` <-> magnon `b` + phonon `p`  (`a = b + p`):

    amplitude  =  V_{K=a, Q=p}   (J~(b, p), e(p), E(p))   ==   V_{K=b, Q=-p}^*  (hermiticity)

This one object serves the magnon emission `a -> b + p`, the absorption `b + p -> a`, and the absorption of the phonon `p` by the magnon `b`.
The kernels' conventions of the CSV (`r_ij = r_i - r_j` or the opposite) drop out: for the inversion-symmetric bcc lattice
`|J~(-p,-q)| = |J~(p,q)|`.

## Discrepancies found

| # | where | what the code did | what the math says | effect (20^3, 600 K / 300 K, smearing 0.3) |
|---|-------|-------------------|--------------------|---------------------------------------------|
| 1 | magnon channel, path kernel `phase_1_scan_path` and dynamics kernel `phase_1_scan` | `V(K=q, Q=k-q)` (called `V_Gmp` in the grid kernel, which also computes the correct `V(K=q, Q=q-k)` and discards it). First Fourier argument `2q-k` instead of `k`. | `V_{q,q-k}` of tex 2.1 | see the combined effect below |
| 2 | magnon channel, path kernel | emission term only. The `c_type 1` branch of `phase_lifetime_path` is inside a string and the kernel never writes type 1 events | tex 2.1 has a second term `[n_ph(k-q) - n_k] delta(E_q - E_k + E_ph(k-q))` | path magnon rates lack the absorption channel (+36 % channels: 18.6 M -> 25.4 M). The grid kernel is complete (its unified events feed all three participants), so `path_lifetimes.csv` and `equilibrium_lifetimes.csv` disagreed |
| 3 | phonon channel, path kernel | `V(K=k, Q=-q)` with `E_q + E_{k-q} - E_k = 0` (copied from tex 1.2) | for the absorption `k-q -> k` the amplitude is `V_{k,+q}` = `V_{k-q,-q}^*`. Tex 1.2 labels it `V_{k,-q}`: with the energy delta written for `k-q -> k` the first label has to be `k-q`. The grid kernel never had this error (its single event weight is `V_{q,q-k}`) | see the combined effect below |
| 4 | adaptive width, all bare kernels | `grad_f_phon` is never filled (`_compute_group_velocities` disabled, `np.zeros`), so `d_g = -grad_mag + grad_phon` ignored the phonon dispersion | the adaptive Gaussian is meant to follow both dispersions | sigma too small / large by a mode-dependent factor, changes the rates by 10-30 % |
| 5 | SLC data | acoustic sum rule `sum_k J_{ij,k} = 0` is violated by 46 % of max |J| (`check_slc_asr`); `enforce_slc_asr()` is commented out | translation invariance: `J~(p, q -> 0) -> 0` | `V ~ 1/sqrt(E_q)` has no soft-phonon suppression; only the `E < 0.1 / 0.2 meV` cut-offs regulate it. Not changed (needs a decision on the remedy) |
| 6 | width floor | path kernel floor `1e-5 meV` hard-coded, grid kernel uses `min_sigma` | one floor for both | lifetimes and dynamics use different broadening |
| 7 | not audited | hybrid kernels (`fill_hybrid_vertex_cache*`) and the relativistic kernels `*_rel`, `*_nc` build their own vertex arguments | | |

Tex notes: `mag_ph.txt` has no magnon section; 1.2 should read `V^{+-}_{k-q,-q,lambda,nm}` (equivalently `|V^{+-}_{k,q,lambda,nm}|^2`).
2.1 is consistent with the Hamiltonian as written.

The remaining difference between the reference with the tex-literal magnon vertex and the one built from the Hamiltonian (`V_{k,k-q}`) is 5-15 %.
It measures how far the CSV tensors are from the exchange symmetry `J^{ab}_{ij,k} = J^{ba}_{ji,k}` that hermiticity needs.

## Combined effect (20^3 mesh, T_mag = 600 K, T_phon = 300 K, smearing 0.3; `Outputs/bare_audit/`, plot `python/plot_bare_audit.py`)

- magnons (7987 modes): gamma_after / gamma_before has median 1.28, 10-90 % band 0.34-1.69; below ~150 meV the rate drops by up to 10x
  (E = 62 meV: 5.2 -> 0.7 ps^-1), above ~250 meV it rises by up to ~2x.
- phonons (23997 modes): median 0.66, 10-90 % band 0.10-1.39; summed rate after / before = 0.11, i.e. the mode-resolved weight moves.
- the patched kernels equal the independent numpy reference to 4 digits (kernel_vs_ref.py).
