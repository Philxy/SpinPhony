"""
Type-II demagnetization + ab initio magnon-phonon remagnetization for CrI3,
following the physical picture in Padmanabhan et al., Nat. Commun. 13, 4473
(2022): a fast sub-ps Elliott-Yafet spin-flip demag, a further reduction
over tens of ps as electron-phonon equilibration continues, and an eventual
nanosecond-timescale recovery. 

The recovery is modeled phenomenologically by a single macroscopic relaxation 
time tau_mag, which is computed rigorously from the ab initio data by 
weighting the mode-resolved scattering rates (gamma_{k,mu}) with the mode's 
spin angular momentum (S^z ~ mag_char) and the initial non-equilibrium 
occupation deviation:

    1 / tau_mag = sum_{k,mu} [ gamma_{k,mu} * S^z_{k,mu} * Delta n_{k,mu}(0) ]
                  ------------------------------------------------------------
                  sum_{k,mu} [ S^z_{k,mu} * Delta n_{k,mu}(0) ]

where Delta n_{k,mu}(0) = n_BE(eps_{k,mu}; T_exc) - n_BE(eps_{k,mu}; T0).

T0 = 15 K is fixed (ab initio equilibrium reference temperature).
T_exc is the initial excitation temperature fitted to the recovery shape.
"""

import io
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scienceplots

plt.style.use("science")

K_B_MEV = 0.08617333262
EPS_FLOOR_MEV = 1e-4

# ----------------------------------------------------------------------
# Experimental data (unchanged)
# ----------------------------------------------------------------------
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

def load_hybrid_modes(filepath):
    df = pd.read_csv(filepath, comment="#")
    try:
        eps = df["energy_meV"].values.astype(float)
        gamma = df["gamma_ps-1"].values.astype(float)
        mag_char = df["mag_character"].values.astype(float)
    except KeyError as e:
        raise KeyError(f"Missing expected column in {filepath}: {e}")

    if "phon_character" in df.columns:
        residual = np.abs(mag_char + df["phon_character"].values - 1.0)
        if residual.max() > 1e-3:
            print(f"[check] WARNING: mag_character+phon_character deviates from 1 "
                  f"by up to {residual.max():.3e} -- check the input file.")

    eps = np.maximum(eps, EPS_FLOOR_MEV)
    return eps, gamma, mag_char

def bose(eps_mev, T, kB=K_B_MEV):
    x = eps_mev / (kB * T)
    return 1.0 / np.expm1(x)

def compute_tau_mag(T_exc, T0, eps, gamma, mag_char):
    """
    Computes the effective macroscopic relaxation time tau_mag evaluated
    using the initial non-equilibrium occupation deviation Delta n(0).
    
    Because S^z_{k,mu} = hbar * mag_char_{k,mu} and the hbar cancels out in
    the sum ratio, we directly weight by mag_char.
    """
    n0 = bose(eps, T0)
    nexc = bose(eps, T_exc)
    
    # Delta n_k,mu(0)
    delta_n = nexc - n0
    
    # Weight = S^z_{k,mu} * Delta n_{k,mu}(0)
    weight = mag_char * delta_n 
    
    # 1 / tau_mag = sum(gamma * weight) / sum(weight)
    inv_tau_mag = np.sum(gamma * weight) / np.sum(weight)
    
    return 1.0 / inv_tau_mag

def main():
    parser = argparse.ArgumentParser(
        description="Type-II demag (phenomenological) x ab initio single-tau "
                    "recovery, for CrI3 TRPR data."
    )
    parser.add_argument("filepath", help="Full-BZ hybrid_path_lifetimes.csv")
    parser.add_argument("--T0", type=float, default=15.0)
    parser.add_argument("--T_exc_max", type=float, default=100.0)
    parser.add_argument("--tau1_bounds", type=float, nargs=2, default=[0.05, 2.0],
                        help="Bounds (ps) for the sub-ps Elliott-Yafet demag stage.")
    parser.add_argument("--tau2_bounds", type=float, nargs=2, default=[2.0, 150.0],
                        help="Bounds (ps) for the 'tens of ps' secondary demag stage.")
    args = parser.parse_args()

    print(f"Loading ab initio hybrid-mode data from {args.filepath}...")
    eps, gamma, mag_char = load_hybrid_modes(args.filepath)
    print(f"  {len(eps)} modes loaded, T0 = {args.T0} K (fixed)")

    i_min = np.argmin(gamma)
    print(f"Slowest mode: eps = {eps[i_min]:.3f} meV, mag_char = {mag_char[i_min]:.3f}, "
        f"gamma = {gamma[i_min]:.1f} 1/ps")

    print(f"gamma range: [{gamma.min():.6e}, {gamma.max():.6e}]")
    neg = gamma < 0
    print(f"negative gamma: {neg.sum()} modes, of which {int((neg & (mag_char>0.01)).sum())} have mag_char > 0.01")
    
    lin_x, lin_y = np.loadtxt(io.StringIO(lin_data), delimiter=",", unpack=True)
    log_x, log_y = np.loadtxt(io.StringIO(log_exp), delimiter=",", unpack=True)
    idx = np.argsort(np.concatenate([lin_x, log_x]))
    total_x = np.concatenate([lin_x, log_x])[idx]
    total_y = np.concatenate([lin_y, log_y])[idx]

    def model(t, A1, tau1, A2, tau2, T_exc):
        # 1. Compute single macroscopic tau_mag from the initial condition T_exc
        tau_mag = compute_tau_mag(T_exc, args.T0, eps, gamma, mag_char)
        
        # 2. Phenomenological excitation
        rise = A1 * (1.0 - np.exp(-t / tau1)) + A2 * (1.0 - np.exp(-t / tau2))
        
        # 3. Apply the single-exponential macroscopic decay
        recovery = np.exp(-t / tau_mag)
        
        return -rise * recovery

    # Bounds and initial parameters
    p0 = [0.5, 0.3, 0.4, 20.0, 80.0]
    bounds = (
        [0.0, args.tau1_bounds[0], 0.0, args.tau2_bounds[0], args.T0 + 1e-3],
        [1.5, args.tau1_bounds[1], 1.5, args.tau2_bounds[1], args.T_exc_max],
    )
    popt, pcov = curve_fit(model, total_x, total_y, p0=p0, bounds=bounds, maxfev=40000)
    perr = np.sqrt(np.diag(pcov))
    A1_fit, tau1_fit, A2_fit, tau2_fit, T_exc_fit = popt
    A1_err, tau1_err, A2_err, tau2_err, T_exc_err = perr

    # Calculate final tau_mag for reporting
    tau_mag_fit = compute_tau_mag(T_exc_fit, args.T0, eps, gamma, mag_char)

    print()
    print("[Type-II demag onset, phenomenological -- electron-driven physics]")
    print(f"A1        = {A1_fit:.4f} +/- {A1_err:.4f}   (sub-ps, Elliott-Yafet spin-flip)")
    print(f"tau1      = {tau1_fit:.4f} +/- {tau1_err:.4f} ps")
    print(f"A2        = {A2_fit:.4f} +/- {A2_err:.4f}   (tens of ps, continued e-ph equilibration)")
    print(f"tau2      = {tau2_fit:.4f} +/- {tau2_err:.4f} ps")
    print()
    print("[ab initio macroscopic recovery]")
    print(f"T_exc     = {T_exc_fit:.2f} +/- {T_exc_err:.2f} K")
    print(f"tau_mag   = {tau_mag_fit:.2f} ps")

    if T_exc_fit > 0.9 * args.T_exc_max:
        print(f"[check] WARNING: T_exc near its upper bound -- likely in the "
              f"classical limit where tau_mag saturates.")
              
    t_continuous = np.geomspace(max(total_x.min(), 1e-2), total_x.max(), 400)
    M_continuous = model(t_continuous, *popt)

    fig, ax = plt.subplots(figsize=(8 / 2.52, 6 / 2.52))
    ax.scatter(total_x, total_y, label="Experiment", color="black", s=10)
    ax.plot(t_continuous, M_continuous, color="tab:blue", linewidth=2.0,
            label=r"Onset $\times$ exp(-$t/\tau_\mathrm{mag}$)")

    fit_text = (
        rf"$\tau_1={tau1_fit:.2f}$ ps, $\tau_2={tau2_fit:.1f}$ ps" "\n"
        rf"$T_\mathrm{{exc}}={T_exc_fit:.0f}$ K $\rightarrow \tau_\mathrm{{mag}}={tau_mag_fit:.0f}$ ps"
    )
    ax.text(0.97, 0.55, fit_text, transform=ax.transAxes, fontsize=7.5,
            ha="right", va="center",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"))

    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlabel("Delay (ps)", fontsize=12)
    ax.set_ylabel(r"TRPR (a.u.)", fontsize=12)
    fig.tight_layout()
    plt.savefig("remag_typeII_single_tau_fit.pdf", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()