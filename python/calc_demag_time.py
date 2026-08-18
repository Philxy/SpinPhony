"""
Calculate macroscopic demagnetization timescale at T  from microscopic hybrid mode lifetimes.

Usage:
    python calc_demag_20K.py Outputs/CrI3_Path_Hyrbid_Full/hybrid_path_lifetimes.csv
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

def main():
    p = argparse.ArgumentParser(description="Calculate tau_demag at 20K.")
    p.add_argument("csv_file", nargs='?', default="Outputs/CrI3_Path_Hyrbid_Full/hybrid_path_lifetimes.csv", 
                   help="Path to hybrid_path_lifetimes.csv")
    args = p.parse_args()

    # 1. Load data
    df = load_path_csv(args.csv_file)
    
    # 2. Extract arrays
    E = df["energy_meV"].values
    Gamma = df["gamma_ps-1"].values
    # Absolute capacity of the mode to absorb/transfer Angular Momentum
    Lz = np.abs(df["phon_AM_z_hbar"].values)  
    
    # Filter valid scattering channels
    valid_modes = np.isfinite(Gamma) & (Gamma > 0) & np.isfinite(Lz)
    E = E[valid_modes]
    Gamma = Gamma[valid_modes]
    Lz = Lz[valid_modes]

    # 3. Calculate weights (dn_k/dT at 20 K)
    weights = calc_dn_dT(E, T_KELVIN)
    
    # 4. Apply the equation
    numerator = np.sum(Lz * Gamma * weights)
    denominator = np.sum(Lz * weights)
    
    if denominator == 0:
        print(f"Error: Denominator is 0. No modes have both PAM and thermal population at {T_KELVIN}K.")
        return

    gamma_demag = numerator / denominator
    tau_demag = 1.0 / gamma_demag

    # 5. Output
    print(f"Loaded {len(E):,} valid modes from {args.csv_file}")
    print(f"Temperature         : {T_KELVIN} K")
    print(f"Gamma_demag         : {gamma_demag:.4f} ps^-1")
    print(f"tau_demag           : {tau_demag:.4f} ps")

if __name__ == "__main__":
    main()