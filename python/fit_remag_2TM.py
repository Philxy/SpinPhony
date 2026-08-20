"""
Fit the ab initio nonlinear 2TM model to the REMAGNETIZATION portion of the
digitized CrI3 TRPR curve (Padmanabhan et al., Nat. Commun. 13, 4473, 2022).

The model is initialised at t0 (= --t_cutoff, default 150 ps) and integrated
forward from there; nothing before t0 is simulated. This is deliberate: CrI3
shows two distinct demagnetization stages (a sub-ps Elliott-Yafet spin-flip
stage and a subsequent tens-of-ps electron-phonon equilibration stage), both
electronically driven and outside what a magnon-phonon-only model contains.
The state at t0 is therefore taken as an initial condition to be fitted, not
as something the model must reproduce.

    TRPR_model(t) = A * [ M(t) / M(Teq) - 1 ],    t >= t0

with M(t) obtained by solving the 2TM ODE (reusing solve_2TM.py's machinery),
not a phenomenological exponential -- the shape is fixed by the ab initio
Q_flux(T_m,T_p), C_m(T), C_p(T) and character-weighted M(T). Free parameters:
T_m(t0), optionally T_p(t0) (see --fit_Tp), and a linear amplitude A absorbing
the arbitrary Kerr-rotation calibration. The offset y0 is fixed to 0, following
the same reasoning validated in calc_demagcopy.py (letting it float was found
to be degenerate with tau).

KNOWN LIMITATION: this is a closed two-subsystem model with no external bath,
so it relaxes to a common T_f > Teq and cannot return to M(Teq). The
experimental ns-scale recovery to baseline requires a third channel (substrate
/ cryostat) that is not implemented here.

Usage:
    python fit_remag_2TM.py Outputs/CrI3_Full_NonHyrbid_sig_2.0_15K \\
        --Teq 15 --t_cutoff 150 --Tm0_max 50
"""
import argparse
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

from solve_2TM import (
    build_energy_flux, load_mode_data, build_heat_capacities,
    build_magnetization_func, make_rhs, solve_final_temperature,
)

# ---------------------------------------------------------------------
# Digitized experimental TRPR data (Padmanabhan et al., Nat. Commun. 13,
# 4473 (2022)), copied verbatim from calc_demagcopy.py.
# ---------------------------------------------------------------------
lin_data = """
1.1008974616569698, -0.0003616636528028794
1.0690844551604273, -0.055334538878842765
1.0523407675306817, -0.084267631103074
0.8049527828009091, -0.11175406871609406
1.4692585895117922, -0.16383363471971069
0.9920634920635011, -0.18842676311030737
1.202615364007789, -0.22459312839059686
1.408144129663128, -0.26943942133815546
1.1607561449333792, -0.2969258589511753
1.1507099323555057, -0.3142857142857143
1.140663719777649, -0.33164556962025327
1.8250619516442534, -0.3490054249547919
1.8133413703034083, -0.36925858951175417
2.0305907172995834, -0.39385171790235074
2.4818330989217134, -0.4141048824593129
3.395201259125335, -0.4358047016274864
4.314429709999345, -0.4473779385171789
5.231146607728901, -0.46329113924050624
6.613338021565897, -0.47486437613019883
7.7640479539213745, -0.4864376130198915
8.91392070189541, -0.4994575045207956
10.294437746969413, -0.5139240506329115
11.674954792043419, -0.5283905967450272
12.593346058535937, -0.5414104882459313
13.744893175272935, -0.5515370705244123
15.588373183309903, -0.566003616636528
17.428504453820928, -0.5862567811934899
19.039665795994928, -0.6021699819168174
20.649152769405948, -0.6209764918625678
22.493469961824406, -0.6339963833634719
24.107142857142875, -0.6455696202531644
26.412748643761315, -0.6614828209764919
27.328628357109384, -0.6788426763110308
28.93895251490188, -0.6962025316455696
30.550951041457385, -0.7106690777576853
32.85404527493136, -0.7309222423146474
34.7008740204943, -0.7396021699819169
36.543516844149764, -0.755515370705244
38.62098988681267, -0.7656419529837251
40.92408412028666, -0.7858951175406872
42.99820842542362, -0.8018083182640143
46.224717031679056, -0.8264014466546111
48.530322818297506, -0.8423146473779385
50.142321344852995, -0.8567811934900542
52.215608465608476, -0.8741410488245932
54.76441631504924, -0.8698010849909583
56.37725202598621, -0.8828209764918625
60.07760699216396, -0.8886075949367088
62.1542428504454, -0.9001808318264015
63.757869533185975, -0.9291139240506329
67.22925457102673, -0.9305605786618445
69.76801620788964, -0.9435804701627486
73.92965976826736, -0.9522603978300179
76.47763043332664, -0.9493670886075948
78.09046614426364, -0.962386980108499
82.2562956265488, -0.9638336347197107
84.57194762574511, -0.962386980108499
88.04333266358582, -0.9638336347197106
92.20497622396356, -0.9725135623869802
95.21088674569687, -0.9783001808318263
98.68645770544506, -0.9725135623869802
"""

log_exp = """
102.43040264272612, -0.9754954954954955
104.91987385550993, -0.9855855855855855
112.41927739901041, -0.9812612612612613
107.79292321723639, -0.9855855855855855
114.11923176004694, -0.9971171171171171
117.59664705963829, -0.9827027027027027
121.90969901941544, -0.9927927927927926
126.00215384548649, -0.9971171171171171
129.841662846783, -0.9985585585585587
136.63924023084442, -0.9884684684684685
141.226159258709, -0.9754954954954954
133.79816849393947, -0.9971171171171171
152.68944199429706, -0.9726126126126126
147.73020419788665, -0.9812612612612613
158.76542773701647, -0.9610810810810809
164.58841296973932, -0.9581981981981982
169.60371369135518, -0.9581981981981981
172.68595526863626, -0.9697297297297295
177.94800245923028, -0.9553153153153152
186.70281970862413, -0.9553153153153153
195.88836292296878, -0.9538738738738738
203.68336750703858, -0.9466666666666668
210.52092831679684, -0.9409009009009008
216.9358742091096, -0.927927927927928
225.56842444678847, -0.9221621621621623
235.24957412312185, -0.9207207207207208
245.34622813826277, -0.9135135135135135
255.10931433942753, -0.9120720720720721
262.8829473990812, -0.9005405405405404
272.5246152511977, -0.8890090090090089
278.31140583821644, -0.8774774774774774
292.0039759309533, -0.8688288288288287
304.5364666995754, -0.865945945945946
318.56162234658217, -0.8630630630630629
329.2555955026119, -0.852972972972973
339.2885971940735, -0.8443243243243241
350.67836575380574, -0.8342342342342342
362.45048382046804, -0.8241441441441442
376.8735110014648, -0.8169369369369368
"""


def load_experimental_data(t_cutoff):
    """Merges lin_data + log_exp, sorts by delay, and restricts to t>t_cutoff
    -- the recovery-only window (electronically-driven fast demag excluded)."""
    lin_t, lin_y = np.loadtxt(io.StringIO(lin_data), delimiter=",", unpack=True)
    log_t, log_y = np.loadtxt(io.StringIO(log_exp), delimiter=",", unpack=True)

    t_all = np.concatenate([lin_t, log_t])
    y_all = np.concatenate([lin_y, log_y])
    order = np.argsort(t_all)
    t_all, y_all = t_all[order], y_all[order]

    mask = t_all > t_cutoff
    return t_all, y_all, t_all[mask], y_all[mask]


def make_trpr_model(rhs, M_func, t0, Tp_fixed=None):
    """
    TRPR_model(t) = A * [M(t)/M(Teq) - 1], y0 fixed to 0, with the 2TM
    integrated FORWARD FROM t0 -- not from t=0.

    Starting at t0 (= the fit cutoff) is deliberate: the sub-ps Elliott-Yafet
    demagnetization and the subsequent tens-of-ps electron-phonon stage are
    outside what a magnon-phonon-only model can describe, so the state at t0
    is taken as the initial condition rather than something this model has to
    reproduce. It also removes a severe conditioning problem: integrating from
    t=0 spent ~80% of the magnon temperature excursion before the first fitted
    data point, so T_m(0) was extrapolated backwards through a window no data
    constrains, which pushed the fit hard against its upper bound.

    If Tp_fixed is not None the model has 2 parameters (T_m(t0), A); otherwise
    the lattice temperature at t0 is fitted too, giving 3 (T_m(t0), T_p(t0), A).
    Note T_p(t0) is generally NOT the bath temperature: the lattice has already
    absorbed energy from the fast stages by t0.

    t_array must be pre-sorted and all entries >= t0 (guaranteed by
    load_experimental_data, which masks on t > t_cutoff = t0).
    """
    def _T_m_of_t(t_array, Tm_t0, Tp_t0):
        sol = solve_ivp(rhs, (t0, t_array[-1]), [Tm_t0, Tp_t0], t_eval=t_array,
                        method="Radau", rtol=1e-8, atol=1e-10)
        return sol.y[0]

    if Tp_fixed is not None:
        def trpr_model(t_array, Tm_t0, A):
            return A * (M_func(_T_m_of_t(t_array, Tm_t0, Tp_fixed)) - 1.0)
    else:
        def trpr_model(t_array, Tm_t0, Tp_t0, A):
            return A * (M_func(_T_m_of_t(t_array, Tm_t0, Tp_t0)) - 1.0)

    return trpr_model


def main():
    p = argparse.ArgumentParser(
        description="Fit the ab initio nonlinear 2TM model to the CrI3 TRPR recovery curve."
    )
    p.add_argument("data_dir", help="Directory with Gmp_temperature_grid.csv and hybrid_path_lifetimes.csv")
    p.add_argument("--Teq", type=float, default=15.0,
                   help="Bath/equilibrium temperature (K), used as the reference for M(T)/M(Teq) "
                        "and as the default T_p(t0) when --fit_Tp is not set.")
    p.add_argument("--t_cutoff", type=float, default=150.0,
                   help="t0: the model is initialised here and only t > t0 is fitted. Excludes the "
                        "two electronically-driven demagnetization stages entirely.")
    p.add_argument("--Tm0_guess", type=float, default=30.0, help="Initial guess for T_m(t0) (K)")
    p.add_argument("--Tm0_max", type=float, default=50.0,
                   help="Upper bound on T_m(t0) (K) -- keep comfortably below Tc so linear spin-wave "
                        "theory (the un-renormalized dispersion this model is built on) stays valid.")
    p.add_argument("--fit_Tp", action="store_true",
                   help="Also fit the lattice temperature at t0. By default T_p(t0) is fixed to "
                        "--Tp_t0 (or --Teq). Physically T_p(t0) > Teq, since the lattice has "
                        "already absorbed energy from the fast stages by t0.")
    p.add_argument("--Tp_t0", type=float, default=None,
                   help="Fixed lattice temperature at t0 (K). Default: --Teq. Ignored with --fit_Tp.")
    p.add_argument("--A_guess", type=float, default=1.0, help="Initial guess for the amplitude A")
    p.add_argument("--A_max", type=float, default=10.0, help="Upper bound on the amplitude A")
    p.add_argument("--t_plot_max", type=float, default=2000.0, help="Extend the plotted model prediction to this t (ps)")
    p.add_argument("--out_csv", default=None, help="Output CSV path (default: <data_dir>/remag_2TM_fit.csv)")
    p.add_argument("--out_png", default=None, help="Output plot path (default: <data_dir>/remag_2TM_fit.png)")
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
        print("  WARNING: legacy CSV without Q_flux -- flux reconstructed as G_mp*(T_m-T_p), "
              "which underestimates the true flux badly for T_m >> T_p. Re-run --Gmp_grid.")
    if args.Tm0_max > Tm_hi:
        print(f"  WARNING: --Tm0_max={args.Tm0_max} exceeds the tabulated range "
             f"(max {Tm_hi:.1f} K) -- the flux will be clamped to the edge above that.")

    print(f"Loading mode data from {hybrid_csv} (full-BZ) ...")
    energy, w_mag, w_ph = load_mode_data(hybrid_csv)

    print("Building C_m(T), C_p(T), M(T)/M(Teq) ...")
    C_m_func, C_p_func, E_m_func, E_p_func = build_heat_capacities(energy, w_mag, w_ph)
    M_func = build_magnetization_func(energy, w_mag, args.Teq)

    rhs = make_rhs(flux_func, T_range, C_m_func, C_p_func)

    t0 = args.t_cutoff
    Tp_t0_fixed = None if args.fit_Tp else (args.Tp_t0 if args.Tp_t0 is not None else args.Teq)
    trpr_model = make_trpr_model(rhs, M_func, t0, Tp_fixed=Tp_t0_fixed)

    print(f"\nLoading digitized TRPR data, fitting t > {t0} ps ...")
    t_all, y_all, t_fit, y_fit = load_experimental_data(t0)
    print(f"  {len(t_fit)} of {len(t_all)} digitized points used in the fit "
         f"(t = {t_fit[0]:.1f} to {t_fit[-1]:.1f} ps)")
    print(f"  model initialised AT t0 = {t0} ps (the pre-t0 demagnetization stages are "
          "not simulated at all)")

    if Tp_t0_fixed is not None:
        p0 = [args.Tm0_guess, args.A_guess]
        bounds = ([args.Teq, 0.0], [args.Tm0_max, args.A_max])
        print(f"  T_p(t0) fixed to {Tp_t0_fixed:g} K; fitting T_m(t0), A")
    else:
        p0 = [args.Tm0_guess, args.Teq, args.A_guess]
        bounds = ([args.Teq, args.Teq, 0.0], [args.Tm0_max, args.Tm0_max, args.A_max])
        print("  fitting T_m(t0), T_p(t0), A")

    print(f"\nFitting TRPR(t) = A * [M(t)/M(Teq) - 1]  (y0 fixed to 0) ...")
    popt, pcov = curve_fit(trpr_model, t_fit, y_fit, p0=p0, bounds=bounds, maxfev=2000)
    perr = np.sqrt(np.diag(pcov))

    if Tp_t0_fixed is not None:
        T_m0_fit, A_fit = popt
        T_m0_err, A_err = perr
        T_p0_fit, T_p0_err = Tp_t0_fixed, 0.0
    else:
        T_m0_fit, T_p0_fit, A_fit = popt
        T_m0_err, T_p0_err, A_err = perr

    print(f"\nFit result:")
    print(f"  T_m(t0) = {T_m0_fit:.3f} +/- {T_m0_err:.3f} K")
    if Tp_t0_fixed is None:
        print(f"  T_p(t0) = {T_p0_fit:.3f} +/- {T_p0_err:.3f} K")
    else:
        print(f"  T_p(t0) = {T_p0_fit:.3f} K (fixed)")
    print(f"  A       = {A_fit:.4f} +/- {A_err:.4f}")

    if T_m0_fit > args.Tm0_max - 0.05:
        print(f"  WARNING: T_m(t0) is pinned at the upper bound (--Tm0_max={args.Tm0_max}). "
             "Raise it (keeping well below Tc and within the --Gmp_grid range), or reconsider "
             "the model assumptions -- note the closed 2TM cannot recover to M(Teq) at all, "
             "since it relaxes to T_f > Teq with no external bath.")

    T_f_fit = solve_final_temperature(T_m0_fit, T_p0_fit, E_m_func, E_p_func)
    print(f"  Implied final common temperature: T_f = {T_f_fit:.3f} K")

    # Residual check against the independent absorbed-energy estimate from
    # earlier (~25-55 K): report where the fit landed relative to that range.
    print(f"  (compare to the independent absorbed-energy T_exc estimate of ~25-55 K from earlier)")

    # Extended model prediction, fit params applied over a longer window, to
    # extract an effective single-exponential tau_remag directly comparable
    # to the literature value (Padmanabhan et al.: 1277(33) ps), and for
    # plotting the full predicted recovery beyond the digitized data range.
    t_model = np.linspace(t0, args.t_plot_max, 800)
    sol_full = solve_ivp(rhs, (t0, args.t_plot_max), [T_m0_fit, T_p0_fit], t_eval=t_model,
                         method="Radau", rtol=1e-8, atol=1e-10)
    T_m_model = sol_full.y[0]
    T_p_model = sol_full.y[1]
    y_model = A_fit * (M_func(T_m_model) - 1.0)

    y_model_at_t0 = y_model[0]

    def single_exp(t, tau):
        return y_model_at_t0 * np.exp(-(t - t0) / tau)

    try:
        tau_popt, tau_pcov = curve_fit(single_exp, t_model, y_model, p0=[500.0])
        tau_eff = tau_popt[0]
        print(f"  Effective single-exponential tau_remag (fit to the model's own recovery, "
             f"t>{t0} ps) = {tau_eff:.1f} ps")
        print(f"  (compare to Padmanabhan et al.'s fitted tau_remag = 1277(33) ps)")
    except RuntimeError:
        tau_eff = np.nan
        print("  Could not extract an effective single-exponential tau_remag from the model curve.")

    out_csv = args.out_csv or f"{args.data_dir}/remag_2TM_fit.csv"
    pd.DataFrame({"t_ps": t_model, "T_m_K": T_m_model, "T_p_K": T_p_model, "TRPR_model": y_model}).to_csv(out_csv, index=False)
    print(f"\n-> Saved fitted model trajectory to {out_csv}")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(t_all[t_all <= t0], y_all[t_all <= t0],
              s=12, color="lightgray", label="digitized data (not modelled)")
    ax.scatter(t_fit, y_fit, s=14, color="black", label=f"digitized data (fit, t>{t0:g} ps)")
    # Model curve starts at t0 - there is deliberately no prediction before it.
    ax.plot(t_model, y_model, color="tab:red", lw=1.8,
           label=rf"2TM from $t_0$: $T_m(t_0)={T_m0_fit:.1f}\pm{T_m0_err:.1f}$ K, $A={A_fit:.2f}$")
    ax.axvline(t0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Delay (ps)")
    ax.set_ylabel("TRPR (a.u.)")
    ax.set_xlim(-20,1000)
    ax.set_ylim(-1.5,0.2)
    ax.legend(fontsize=9)
    ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    out_png = args.out_png or f"{args.data_dir}/remag_2TM_fit.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"-> Saved plot to {out_png}")


if __name__ == "__main__":
    main()
