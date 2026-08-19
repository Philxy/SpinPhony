"""
Calculate macroscopic demagnetization timescale at T from microscopic hybrid mode lifetimes.
Compares weightings by magnonic character, phonon angular momentum, and spin angular momentum.

Usage:
    python calc_demag.py Outputs/CrI3_Path_Hyrbid_Full/hybrid_path_lifetimes.csv
"""
import argparse
import numpy as np
import pandas as pd

# Boltzmann constant in meV/K
K_B = 0.086173
T_KELVIN = 15.0

def load_path_csv(path):
    """Loads a SpinPhony path CSV, skipping the '# path_labels:' comment line."""
    with open(path) as f:
        first = f.readline().strip()
    return pd.read_csv(path, skiprows=1) if first.startswith("# path_labels:") else pd.read_csv(path)

def calc_dn_dT(energy_meV, T):
    """Calculates the thermal weight (dn_B/dT) for a given energy and temperature."""
    # Clip energy slightly above 0 to avoid division by zero at the Gamma point
    E = np.clip(energy_meV, 1e-4, None)
    x = E / (K_B * T)
    
    weights = np.zeros_like(x)
    # Avoid numerical overflow for modes with energy >> k_B * T (where weight ~ 0)
    valid = x < 100 
    
    weights[valid] = (E[valid] / (K_B * T**2)) * np.exp(x[valid]) / (np.exp(x[valid]) - 1)**2
    return weights

def calculate_timescale_naive(thermal_weights, mode_weights, gamma_array):
    """
    Initial-slope estimate of the decay of O_tot = sum_k O_k n_k(t), under the
    ad hoc closure Delta n_k(0) ~ O_k * phi_k: each mode's initial excess
    population is assumed proportional to its own "capacity" O_k (mag
    character, |L^z|, |S^z|, ...) weighted by its thermal responsiveness
    phi_k = dn0/dT, with no further assumption about *why* that excess is
    there. Reasonable as a rough estimator, but not tied to any specific
    physical perturbation mechanism.
    """
    numerator = np.sum(mode_weights * gamma_array * thermal_weights)
    denominator = np.sum(mode_weights * thermal_weights)

    if denominator == 0:
        return float('nan'), float('nan')

    gamma_macro = numerator / denominator
    tau_macro = 1.0 / gamma_macro
    return gamma_macro, tau_macro


def calculate_timescale_rigorous(thermal_weights, mag_character, mode_weights, gamma_array):
    """
    2TM-closure-consistent macroscopic rate for the SIGNED observable
    O_tot = sum_k O_k n_k, derived from

        dO_tot/dt = sum_k O_k * dn_k/dt = -sum_k O_k * gamma_k * Delta n_k

    using the linear-response closure Delta n_k = phi_k * omega_mag_k * (T_m - T_p)
    (the same closure used to derive G_mp): the excess population in mode k is
    proportional not just to its thermal responsiveness phi_k, but specifically
    to how strongly mode k couples to the MAGNON subsystem temperature T_m,
    i.e. to its own magnon character omega_mag_k -- regardless of which
    observable O we are tracking the decay of. This gives

        Gamma_O = sum_k( O_k * omega_mag_k * gamma_k * phi_k )
                / sum_k( O_k * omega_mag_k * phi_k )

    which reduces EXACTLY to Gamma_S = sum (omega_mag)^2 gamma phi
                                      / sum (omega_mag)^2 phi
    when mode_weights = S^z_k = -omega_mag_k (the two minus signs cancel).

    mode_weights should be the SIGNED per-mode quantity (S^z_k or L^z_k, not
    its absolute value) -- O_tot is a real, signed conserved quantity, and
    using |O_k| here would not correspond to any physical dO_tot/dt.
    """
    weight = mode_weights * mag_character
    numerator = np.sum(weight * gamma_array * thermal_weights)
    denominator = np.sum(weight * thermal_weights)

    if denominator == 0:
        return float('nan'), float('nan')

    gamma_macro = numerator / denominator
    tau_macro = 1.0 / gamma_macro
    return gamma_macro, tau_macro

def main():
    p = argparse.ArgumentParser(description=f"Calculate tau_demag at {T_KELVIN}K.")
    p.add_argument("csv_file", nargs='?', default="Outputs/CrI3_Path_Hyrbid_Full_sig_0.2_15K/hybrid_path_lifetimes.csv", 
                   help="Path to hybrid_path_lifetimes.csv")
    args = p.parse_args()

    # 1. Load data
    df = load_path_csv(args.csv_file)
    
    # 2. Extract arrays
    E = df["energy_meV"].values
    Gamma = df["gamma_ps-1"].values
    
    # Extract the three different weighting properties. mag_character is
    # already non-negative by construction; L^z and S^z are kept SIGNED here
    # (needed for the rigorous closure below) with |.| versions derived
    # separately for the naive estimator, matching the original convention.
    omega_mag = df["mag_character"].values
    Lz_signed = df["phon_AM_z_hbar"].values
    Sz_signed = df["spin_AM_z_hbar"].values

    # Filter valid scattering channels
    valid_modes = (np.isfinite(Gamma) & (Gamma > 0) & np.isfinite(omega_mag)
                   & np.isfinite(Lz_signed) & np.isfinite(Sz_signed))
    E = E[valid_modes]
    Gamma = Gamma[valid_modes]
    omega_mag = omega_mag[valid_modes]
    Lz_signed = Lz_signed[valid_modes]
    Sz_signed = Sz_signed[valid_modes]
    Lz_abs = np.abs(Lz_signed)
    Sz_abs = np.abs(Sz_signed)

    # 3. Calculate thermal weights (dn_k/dT)
    thermal_weights = calc_dn_dT(E, T_KELVIN)

    # 4a. Naive (initial-slope) estimates -- original script's formula
    g_mag_n, tau_mag_n = calculate_timescale_naive(thermal_weights, omega_mag, Gamma)
    g_Lz_n,  tau_Lz_n  = calculate_timescale_naive(thermal_weights, Lz_abs, Gamma)
    g_Sz_n,  tau_Sz_n  = calculate_timescale_naive(thermal_weights, Sz_abs, Gamma)

    # 4b. Rigorous (2TM-closure) estimates -- signed O_k, extra omega_mag factor
    g_mag_r, tau_mag_r = calculate_timescale_rigorous(thermal_weights, omega_mag, omega_mag, Gamma)
    g_Lz_r,  tau_Lz_r  = calculate_timescale_rigorous(thermal_weights, omega_mag, Lz_signed, Gamma)
    g_Sz_r,  tau_Sz_r  = calculate_timescale_rigorous(thermal_weights, omega_mag, Sz_signed, Gamma)

    # Cancellation diagnostic: how much the signed L^z/S^z denominator shrank
    # relative to the |L^z|/|S^z| version -- a value near 0 means positive-
    # and negative-PAM contributions are nearly canceling, and Gamma_L is
    # numerically fragile (small denominator, sensitive to noise).
    def _cancellation_fraction(signed, absval):
        w = omega_mag * thermal_weights
        denom_signed = np.sum(signed * w)
        denom_abs = np.sum(absval * w)
        return abs(denom_signed) / denom_abs if denom_abs != 0 else float('nan')

    frac_Lz = _cancellation_fraction(Lz_signed, Lz_abs)
    frac_Sz = _cancellation_fraction(Sz_signed, Sz_abs)

    # 5. Output
    print(f"Loaded {len(E):,} valid modes from {args.csv_file}")
    print(f"Temperature         : {T_KELVIN} K\n")

    print("Naive (initial-slope, |O_k|-weighted):")
    print(f"{'Weighting Property':<25} | {'Gamma (ps^-1)':<15} | {'tau (ps)':<15}")
    print("-" * 61)
    print(f"{'Magnonic Character (ω)':<25} | {g_mag_n:<15.6f} | {tau_mag_n:<15.4f}")
    print(f"{'Phonon AM (|L^z|)':<25} | {g_Lz_n:<15.6f} | {tau_Lz_n:<15.4f}")
    print(f"{'Spin AM (|S^z|)':<25} | {g_Sz_n:<15.6f} | {tau_Sz_n:<15.4f}")

    print("\nRigorous (2TM closure, signed O_k * omega_mag weighting):")
    print(f"{'Weighting Property':<25} | {'Gamma (ps^-1)':<15} | {'tau (ps)':<15}")
    print("-" * 61)
    print(f"{'Magnonic Character (ω)':<25} | {g_mag_r:<15.6f} | {tau_mag_r:<15.4f}")
    print(f"{'Phonon AM (L^z, signed)':<25} | {g_Lz_r:<15.6f} | {tau_Lz_r:<15.4f}   "
         f"[|denom| retained: {frac_Lz:.1%}]")
    print(f"{'Spin AM (S^z, signed)':<25} | {g_Sz_r:<15.6f} | {tau_Sz_r:<15.4f}   "
         f"[|denom| retained: {frac_Sz:.1%}]")

if __name__ == "__main__":
    main()