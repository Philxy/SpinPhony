"""
Solve the nonlinear two-temperature model (2TM) for CrI3, using an ab initio
G_mp(T_m, T_p) grid (from SpinPhony's --Gmp_grid) and exact, non-linearized
magnon/phonon heat capacities C_m(T), C_p(T) built from the full-BZ
hybrid_path_lifetimes.csv:

    C_m(T_m) dT_m/dt = G_mp(T_m, T_p) * (T_p - T_m)
    C_p(T_p) dT_p/dt = G_mp(T_m, T_p) * (T_m - T_p)

C_m, C_p use the same (w_mag)^2 / (w_ph)^2 weighting as G_mp itself, so the
ODE is internally consistent with how G_mp was derived. G_mp is looked up
from the tabulated grid via bilinear interpolation (clamped at the tabulated
edges); C_m, C_p are evaluated exactly at any T from a fine cubic-spline fit
to the ab initio DOS.

Usage:
    python solve_2TM.py Outputs/CrI3_Full_NonHyrbid_sig_2.0_15K \\
        --Tm0 45 --Tp0 45 --t_max 3000
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator, CubicSpline
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import brentq

K_B = 0.08617333262  # meV/K


def load_path_csv(path):
    """Loads a SpinPhony path CSV, skipping the '# path_labels:' comment line."""
    with open(path) as f:
        first = f.readline().strip()
    return pd.read_csv(path, skiprows=1) if first.startswith("# path_labels:") else pd.read_csv(path)


def bose_einstein(energy_meV, T):
    """Vectorized n_BE(eps;T) over a temperature array; safe at eps<=0."""
    T = np.atleast_1d(np.asarray(T, dtype=np.float64))
    energy_meV = np.asarray(energy_meV, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        x = energy_meV[None, :] / (K_B * T[:, None])
        occ = 1.0 / (np.exp(x) - 1.0)
    occ[~np.isfinite(occ)] = 0.0
    occ[:, energy_meV <= 0] = 0.0
    return occ  # shape (len(T), n_modes)


def build_energy_flux(gmp_csv):
    """
    Builds Q(T_m, T_p), the magnon -> phonon energy flux, from the grid
    written by SpinPhony's --Gmp_grid. Sign convention: Q > 0 means the
    magnon subsystem is LOSING energy, i.e. dE_m/dt = -Q, dE_p/dt = +Q.

    Prefers the exact column 'Q_flux',

        Q = sum (w_mag)^2 eps gamma[T_m,T_p] * [n(eps;T_m) - n(eps;T_p)],

    which is exactly antisymmetric under T_m <-> T_p and vanishes identically
    on the diagonal, so the equilibrium fixed point is exact rather than
    interpolated.

    Falls back to the legacy linear-response column 'G_mp' (older CSVs),
    reconstructing Q ~ G_mp * (T_m - T_p). That reconstruction is only valid
    for |T_m - T_p| << eps/k_B and underestimates the flux by orders of
    magnitude for high-energy modes outside that regime -- see the header of
    the --Gmp_grid block in SpinPhony.py.

    Returns (flux_func, T_range, mode) with mode in {"exact", "linear"}.
    """
    df = pd.read_csv(gmp_csv)
    T_m_vals = np.sort(df["T_m_K"].unique())
    T_p_vals = np.sort(df["T_p_K"].unique())

    if "Q_flux" in df.columns:
        col, mode = "Q_flux", "exact"
    else:
        col, mode = "G_mp", "linear"

    grid = (df.pivot(index="T_m_K", columns="T_p_K", values=col)
              .reindex(index=T_m_vals, columns=T_p_vals).to_numpy())

    interp = RegularGridInterpolator(
        (T_m_vals, T_p_vals), grid, method="linear",
        bounds_error=False, fill_value=None,
    )
    T_range = (T_m_vals.min(), T_m_vals.max(), T_p_vals.min(), T_p_vals.max())

    if mode == "exact":
        def flux_func(T_m, T_p):
            return float(interp(np.array([[T_m, T_p]]))[0])
    else:
        def flux_func(T_m, T_p):
            return float(interp(np.array([[T_m, T_p]]))[0]) * (T_m - T_p)

    return flux_func, T_range, mode


def load_mode_data(hybrid_csv):
    """Loads (energy_meV, w_mag, w_ph) for every valid mode of the full-BZ
    hybrid_path_lifetimes.csv, shared by the heat-capacity and
    magnetization builders below so the (large) CSV is only read once."""
    df = load_path_csv(hybrid_csv)
    energy = df["energy_meV"].to_numpy()
    w_mag = df["mag_character"].to_numpy()
    w_ph = 1.0 - w_mag

    valid = energy > 1e-6
    energy, w_mag, w_ph = energy[valid], w_mag[valid], w_ph[valid]
    print(f"  {len(energy):,} valid modes (of {len(df):,}) from the full-BZ hybrid path file")
    return energy, w_mag, w_ph


def build_heat_capacities(energy, w_mag, w_ph, T_min=1.0, T_max=200.0, n_T=400):
    """
    Exact (non-linearized) magnon/phonon heat capacities from the full-BZ
    mode list:

        C_m(T) = sum_k (w_mag_k)^2 * eps_k * phi_k(T)
        C_p(T) = sum_k (w_ph_k)^2  * eps_k * phi_k(T)
        phi_k(T) = eps_k/(kB T^2) * n0(1+n0)

    using (w_mag)^2 / (w_ph)^2, matching the convention already used for
    G_mp in --Gmp_grid, so this ODE is internally consistent with that data.
    Tabulated on a fine grid and cubic-spline interpolated (with cumulative
    integrals E_m(T), E_p(T) for the energy-conservation check) for cheap
    repeated evaluation inside the ODE solver.
    """
    T_grid = np.linspace(T_min, T_max, n_T)
    n0 = bose_einstein(energy, T_grid)  # (n_T, n_modes)
    phi = energy[None, :] / (K_B * T_grid[:, None] ** 2) * n0 * (1.0 + n0)

    C_m_grid = np.sum((w_mag ** 2)[None, :] * energy[None, :] * phi, axis=1)
    C_p_grid = np.sum((w_ph ** 2)[None, :] * energy[None, :] * phi, axis=1)

    C_m_func = CubicSpline(T_grid, C_m_grid)
    C_p_func = CubicSpline(T_grid, C_p_grid)

    # Cumulative energy content E_m(T) = integral_{T_min}^{T} C_m(T') dT',
    # used only for the energy-conservation diagnostic (an additive constant
    # is irrelevant since only differences E(T_f)-E(T_0) are ever compared).
    E_m_grid = np.concatenate([[0.0], cumulative_trapezoid(C_m_grid, T_grid)])
    E_p_grid = np.concatenate([[0.0], cumulative_trapezoid(C_p_grid, T_grid)])
    E_m_func = CubicSpline(T_grid, E_m_grid)
    E_p_func = CubicSpline(T_grid, E_p_grid)

    return C_m_func, C_p_func, E_m_func, E_p_func


def build_magnetization_func(energy, w_mag, T_eq, T_min=1.0, T_max=200.0, n_T=400):
    """
    Dimensionless magnetization, from the magnon occupation via the BE
    distribution -- the same character-weighted construction as the
    "Magnetization Recovery" section:

        M(T)/M(T_eq) = sum_k w_mag_k * [1 - n_BE(eps_k;T)]
                     / sum_k w_mag_k * [1 - n_BE(eps_k;T_eq)]

    Evaluated purely as a function of T_m (the magnon temperature) -- the
    lattice temperature T_p does not enter here, since only the magnon
    occupation carries the ordered moment. Tabulated and cubic-spline
    interpolated, same pattern as the heat capacities.
    """
    T_grid = np.linspace(T_min, T_max, n_T)
    n0 = bose_einstein(energy, T_grid)  # (n_T, n_modes)
    numerator = np.sum(w_mag[None, :] * (1.0 - n0), axis=1)

    n0_eq = bose_einstein(energy, np.array([T_eq]))[0]
    denom = np.sum(w_mag * (1.0 - n0_eq))

    M_ratio_grid = numerator / denom
    return CubicSpline(T_grid, M_ratio_grid)


def make_rhs(flux_func, T_range, C_m_func, C_p_func):
    """
    2TM right-hand side driven by the energy flux directly:

        C_m dT_m/dt = -Q(T_m,T_p),    C_p dT_p/dt = +Q(T_m,T_p)

    Using Q rather than G_mp*(T_p-T_m) keeps total energy conserved by
    construction (the same Q enters both equations with opposite sign) and,
    with the exact Q_flux column, makes the T_m = T_p fixed point exact.
    """
    Tm_lo, Tm_hi, Tp_lo, Tp_hi = T_range

    def rhs(t, y):
        T_m, T_p = y
        Q = flux_func(np.clip(T_m, Tm_lo, Tm_hi), np.clip(T_p, Tp_lo, Tp_hi))

        C_m = float(C_m_func(T_m))
        C_p = float(C_p_func(T_p))

        return [-Q / C_m, Q / C_p]

    return rhs


def solve_final_temperature(T_m0, T_p0, E_m_func, E_p_func, T_lo=1.0, T_hi=200.0):
    """
    The isolated-system final common temperature T_f, from energy
    conservation alone (E_m(T_f)+E_p(T_f) = E_m(T_m0)+E_p(T_p0)), independent
    of the ODE integration -- a closed-form cross-check on where the
    trajectory *should* end up, and on whether (T_m0,T_p0) is actually
    consistent with the intended equilibrium bath temperature.
    """
    E_target = E_m_func(T_m0) + E_p_func(T_p0)

    def f(T):
        return E_m_func(T) + E_p_func(T) - E_target

    return brentq(f, T_lo, T_hi)


def main():
    p = argparse.ArgumentParser(
        description="Solve the nonlinear 2TM using ab initio G_mp(T_m,T_p) and C_m(T),C_p(T)."
    )
    p.add_argument("data_dir", help="Directory with Gmp_temperature_grid.csv and hybrid_path_lifetimes.csv")
    p.add_argument("--Tm0", type=float, required=True, help="Initial magnon temperature (K)")
    p.add_argument("--Tp0", type=float, required=True, help="Initial phonon temperature (K)")
    p.add_argument("--Teq", type=float, default=15.0, help="Expected bath/equilibrium temperature (K), for reporting only")
    p.add_argument("--t_max", type=float, default=3000.0, help="Integration time (ps)")
    p.add_argument("--n_out", type=int, default=600, help="Number of output time points")
    p.add_argument("--out_csv", default=None, help="Output CSV path (default: <data_dir>/2TM_trajectory.csv)")
    p.add_argument("--out_png", default=None, help="Output plot path (default: <data_dir>/2TM_trajectory.png)")
    args = p.parse_args()

    gmp_csv = f"{args.data_dir}/Gmp_temperature_grid.csv"
    hybrid_csv = f"{args.data_dir}/hybrid_path_lifetimes.csv"

    print(f"Loading magnon-phonon energy flux from {gmp_csv} ...")
    flux_func, T_range, flux_mode = build_energy_flux(gmp_csv)
    Tm_lo, Tm_hi, Tp_lo, Tp_hi = T_range
    print(f"  tabulated over T_m in [{Tm_lo:.1f},{Tm_hi:.1f}] K, T_p in [{Tp_lo:.1f},{Tp_hi:.1f}] K")
    if flux_mode == "exact":
        print("  using EXACT Q_flux column (no linearization)")
    else:
        print("  WARNING: this CSV predates the Q_flux column, so the flux is being "
              "reconstructed as G_mp*(T_m-T_p) -- a first-order expansion about T_p that "
              "underestimates the true flux by orders of magnitude once |T_m-T_p| is not "
              "small compared to eps/k_B. Re-run SpinPhony with --Gmp_grid to regenerate.")
    if not (Tm_lo <= args.Tm0 <= Tm_hi):
        print(f"  WARNING: Tm0={args.Tm0} K is outside the tabulated range -- "
             "the flux will be clamped to the nearest edge whenever T_m leaves it.")
    if not (Tp_lo <= args.Tp0 <= Tp_hi):
        print(f"  WARNING: Tp0={args.Tp0} K is outside the tabulated range -- "
             "the flux will be clamped to the nearest edge whenever T_p leaves it.")

    print(f"Loading mode data from {hybrid_csv} (full-BZ) ...")
    energy, w_mag, w_ph = load_mode_data(hybrid_csv)

    print("Building C_m(T), C_p(T) ...")
    C_m_func, C_p_func, E_m_func, E_p_func = build_heat_capacities(energy, w_mag, w_ph)
    print(f"  C_m({args.Tm0:.1f}K) = {C_m_func(args.Tm0):.6e}   C_p({args.Tp0:.1f}K) = {C_p_func(args.Tp0):.6e}")

    print(f"Building M(T)/M(Teq) referenced to Teq={args.Teq:g} K ...")
    M_func = build_magnetization_func(energy, w_mag, args.Teq)

    T_f_analytic = solve_final_temperature(args.Tm0, args.Tp0, E_m_func, E_p_func)
    print(f"\nEnergy-conservation prediction: (T_m0,T_p0)=({args.Tm0},{args.Tp0}) K "
         f"-> T_f = {T_f_analytic:.3f} K")
    if abs(T_f_analytic - args.Teq) > 0.5:
        print(f"  NOTE: this is {abs(T_f_analytic - args.Teq):.2f} K away from the requested "
             f"--Teq={args.Teq} K -- (T_m0,T_p0) may not be the right initial split "
             "for this bath temperature; adjust them if T_f should match --Teq exactly.")

    rhs = make_rhs(flux_func, T_range, C_m_func, C_p_func)

    t_eval = np.linspace(0.0, args.t_max, args.n_out)
    print(f"\nIntegrating 2TM ODE over t in [0,{args.t_max}] ps ...")
    sol = solve_ivp(rhs, (0.0, args.t_max), [args.Tm0, args.Tp0], t_eval=t_eval,
                    method="Radau", rtol=1e-8, atol=1e-10)
    if not sol.success:
        print(f"  WARNING: integration did not fully succeed: {sol.message}")

    T_m_t, T_p_t = sol.y

    # Energy-conservation diagnostic: E_tot(t) should be flat.
    E_tot_t = E_m_func(T_m_t) + E_p_func(T_p_t)
    E_drift = (E_tot_t - E_tot_t[0])
    rel_drift = np.max(np.abs(E_drift)) / max(abs(E_tot_t[0]), 1e-30)
    print(f"Energy conservation check: max|E_tot(t)-E_tot(0)| / |E_tot(0)| = {rel_drift:.2e}")

    print(f"\nFinal state at t={sol.t[-1]:.1f} ps: T_m={T_m_t[-1]:.3f} K, T_p={T_p_t[-1]:.3f} K "
         f"(analytic T_f={T_f_analytic:.3f} K, target --Teq={args.Teq} K)")
    if abs(T_m_t[-1] - T_f_analytic) > 0.1 or abs(T_p_t[-1] - T_f_analytic) > 0.1:
        print("  NOTE: trajectory has not fully converged to T_f within t_max -- consider increasing --t_max.")

    # Magnetization follows from T_m(t) alone (via M(T)/M(Teq), built from
    # the magnon occupation through the BE distribution) -- T_p never enters,
    # since only the magnon subsystem carries the ordered moment.
    M_t = M_func(T_m_t)
    M_final = M_func(T_f_analytic)
    print(f"M(t=0)/M(Teq) = {M_func(args.Tm0):.4f}   M(t_final)/M(Teq) = {M_t[-1]:.4f}   "
         f"(analytic M(T_f)/M(Teq) = {M_final:.4f})")

    out_csv = args.out_csv or f"{args.data_dir}/2TM_trajectory.csv"
    pd.DataFrame({"t_ps": sol.t, "T_m_K": T_m_t, "T_p_K": T_p_t,
                 "M_over_Meq": M_t, "E_tot": E_tot_t}).to_csv(out_csv, index=False)
    print(f"\n-> Saved trajectory to {out_csv}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 8), sharex=True)

    ax1.plot(sol.t, T_m_t, label=r"$T_m(t)$", color="tab:red")
    ax1.plot(sol.t, T_p_t, label=r"$T_p(t)$", color="tab:blue")
    ax1.axhline(args.Teq, color="gray", ls="--", lw=0.8, label=rf"$T_\mathrm{{eq}}={args.Teq:g}$ K")
    ax1.axhline(T_f_analytic, color="black", ls=":", lw=0.8, label=rf"analytic $T_f={T_f_analytic:.2f}$ K")
    ax1.set_ylabel("Temperature (K)")
    ax1.legend(fontsize=9)
    ax1.grid(True, ls="--", alpha=0.3)

    ax2.plot(sol.t, M_t, color="tab:green", label=r"$M(t)/M(T_\mathrm{eq})$")
    ax2.axhline(1.0, color="gray", ls="--", lw=0.8, label=r"$M(T_\mathrm{eq})$")
    ax2.axhline(M_final, color="black", ls=":", lw=0.8, label=rf"analytic $M(T_f)/M(T_\mathrm{{eq}})={M_final:.3f}$")
    ax2.set_xlabel("time (ps)")
    ax2.set_ylabel(r"$M(t)/M(T_\mathrm{eq})$")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    out_png = args.out_png or f"{args.data_dir}/2TM_trajectory.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"-> Saved plot to {out_png}")


if __name__ == "__main__":
    main()
