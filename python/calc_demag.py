import numpy as np
import pandas as pd
import sys

def calculate_tau_mag(filepath, T_K=20.0):
    # Boltzmann constant in meV/K
    kB = 0.08617333262145
    
    print(f"Loading data from {filepath}...")
    try:
        # Read the CSV, skipping the comment line
        df = pd.read_csv(filepath, comment='#')
    except FileNotFoundError:
        print(f"Error: Could not find file at {filepath}")
        sys.exit(1)
        
    # Verify columns exist
    required_cols = ['energy_meV', 'gamma_ps-1', 'mag_character']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: The CSV must contain the following columns: {required_cols}")
        sys.exit(1)

    # Extract arrays
    E_meV = df['energy_meV'].values
    gamma = df['gamma_ps-1'].values
    w_mag = df['mag_character'].values
    
    # Filter out unphysical zero-energy modes
    valid_idx = E_meV > 1e-6
    E_meV = E_meV[valid_idx]
    gamma = gamma[valid_idx]
    w_mag = w_mag[valid_idx]
    
    # Calculate Bose-Einstein derivative at 20 K
    x = E_meV / (kB * T_K)
    dn0_dT = np.zeros_like(x)
    
    # Prevent overflow for high-energy modes at 20 K
    overflow_mask = x < 500  
    x_safe = x[overflow_mask]
    exp_x = np.exp(x_safe)
    dn0_dT[overflow_mask] = (E_meV[overflow_mask] / (kB * T_K**2)) * (exp_x / (exp_x - 1.0)**2)
    
    # Calculate rate-weighted mean
    numerator = np.sum(gamma * w_mag * dn0_dT)
    denominator = np.sum(w_mag * dn0_dT)
    
    if denominator == 0:
        print("Error: Total magnonic heat capacity is zero. Cannot calculate tau_mag.")
        sys.exit(1)
        
    gamma_mag = numerator / denominator
    tau_mag = 1.0 / gamma_mag
    
    print("-" * 40)
    print(f"Dataset Temperature : {T_K} K")
    print(f"Demagnetization Rate: {gamma_mag:.5f} ps^-1")
    print(f"Demagnetization Time: {tau_mag:.5f} ps")
    print("-" * 40)

if __name__ == "__main__":
    # Point to your exact file path
    target_file = 'Outputs/CrI3_Path_Hyrbid_Full_sig_0.2_15K/hybrid_path_lifetimes.csv'
    calculate_tau_mag(target_file, T_K=15.0)