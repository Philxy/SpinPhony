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

def calculate_timescale(thermal_weights, mode_weights, gamma_array):
    """Helper to calculate the macroscopic timescale given a specific mode property."""
    numerator = np.sum(mode_weights * gamma_array * thermal_weights)
    denominator = np.sum(mode_weights * thermal_weights)
    
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
    
    # Extract the three different weighting properties.
    # We use absolute values for angular momenta to represent the absolute 
    # capacity/magnitude of the mode to carry or transfer angular momentum, 
    # preventing oppositely polarized modes from canceling out in the denominator.
    omega_mag = df["mag_character"].values
    Lz = np.abs(df["phon_AM_z_hbar"].values)
    Sz = np.abs(df["spin_AM_z_hbar"].values)
    
    # Filter valid scattering channels
    valid_modes = np.isfinite(Gamma) & (Gamma > 0) & np.isfinite(omega_mag) & np.isfinite(Lz) & np.isfinite(Sz)
    E = E[valid_modes]
    Gamma = Gamma[valid_modes]
    omega_mag = omega_mag[valid_modes]
    Lz = Lz[valid_modes]
    Sz = Sz[valid_modes]

    # 3. Calculate thermal weights (dn_k/dT)
    thermal_weights = calc_dn_dT(E, T_KELVIN)
    
    # 4. Apply the exact derived equation for all three properties
    g_mag, tau_mag = calculate_timescale(thermal_weights, omega_mag, Gamma)
    g_Lz, tau_Lz   = calculate_timescale(thermal_weights, Lz, Gamma)
    g_Sz, tau_Sz   = calculate_timescale(thermal_weights, Sz, Gamma)

    # 5. Output
    print(f"Loaded {len(E):,} valid modes from {args.csv_file}")
    print(f"Temperature         : {T_KELVIN} K\n")
    
    # Print a formatted comparison table
    print(f"{'Weighting Property':<25} | {'Gamma (ps^-1)':<15} | {'tau (ps)':<15}")
    print("-" * 61)
    print(f"{'Magnonic Character (ω)':<25} | {g_mag:<15.6f} | {tau_mag:<15.4f}")
    print(f"{'Phonon AM (|L^z|)':<25} | {g_Lz:<15.6f} | {tau_Lz:<15.4f}")
    print(f"{'Spin AM (|S^z|)':<25} | {g_Sz:<15.6f} | {tau_Sz:<15.4f}")

if __name__ == "__main__":
    main()