import numpy as np
import pandas as pd
import argparse

def compute_Gmp_and_tau(filepath, T0=15.0):
    # Base Constants
    k_B_meV = 0.08617333262             # meV / K
    
    # SI Conversion Factors
    meV_to_J = 1.602176634e-22          # 1 meV in Joules
    ps_to_s  = 1.0e-12                  # 1 picosecond in seconds
    
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, comment='#')
    
    # Extract arrays
    try:
        energy_meV = df['energy_meV'].values
        gamma      = df['gamma_ps-1'].values
        mag_char   = df['mag_character'].values
        phon_char  = df['phon_character'].values
    except KeyError as e:
        raise KeyError(f"Missing expected column in the file: {e}")

    # Compute eps * phi (Heat Capacity of a single mode)
    # Using the numerically stable specific-heat formulation to avoid div-by-zero
    x = energy_meV / (k_B_meV * T0)
    eps_phi = np.zeros_like(x)

    # Assert that wmag + wph = 1
    # np.testing.assert_allclose(mag_char + phon_char, 1.0, atol=1e-5)
    
    small_mask = x < 1e-6
    large_mask = x > 100.0
    mid_mask = ~(small_mask | large_mask)
    
    eps_phi[small_mask] = k_B_meV
    eps_phi[large_mask] = 0.0
    
    x_mid = x[mid_mask]
    eps_phi[mid_mask] = k_B_meV * ( (x_mid / 2.0) / np.sinh(x_mid / 2.0) )**2
    
    # ---------------------------------------------------------
    # Calculate G_mp, G_pm, C_m, C_p
    # ---------------------------------------------------------
    G_mp = np.sum( (mag_char**2) * eps_phi * gamma )      # meV / (K * ps)
    G_pm = np.sum( (phon_char**2) * eps_phi * gamma )     # meV / (K * ps)
    
    C_m  = np.sum( (mag_char**2) * eps_phi )              # meV / K
    C_p  = np.sum( (phon_char**2) * eps_phi )             # meV / K
    
    # Calculate relative deviation delta_mp
    if (G_mp + G_pm) == 0:
        delta_mp = 0.0
    else:
        delta_mp = np.abs(G_mp - G_pm) / (0.5 * (G_mp + G_pm))
    
    # Calculate characteristic lifetime tau_mp
    if G_mp <= 0:
        tau_mp = np.inf
    else:
        tau_mp = 1.0 / (G_mp * (1.0 / C_m + 1.0 / C_p))   # ps

    # Number of q-points to compute intensive (per q-point) variables
    n_qpoints = len(np.unique(df['q_idx']))
    
    results = {
        'T0': T0,
        'G_mp': G_mp,
        'G_mp_per_q': G_mp / n_qpoints,
        'G_pm': G_pm,
        'G_pm_per_q': G_pm / n_qpoints,
        'delta_mp': delta_mp,
        'C_m': C_m,
        'C_m_per_q': C_m / n_qpoints,
        'C_p': C_p,
        'C_p_per_q': C_p / n_qpoints,
        'tau_mp': tau_mp,
        
        # SI Conversions
        # G_mp, G_pm: meV/(K*ps) -> (meV_to_J / ps_to_s) -> Watts/K (Joules/(K*s))
        'G_mp_SI': G_mp * (meV_to_J / ps_to_s), 
        'G_mp_per_q_SI': (G_mp / n_qpoints) * (meV_to_J / ps_to_s), 
        'G_pm_SI': G_pm * (meV_to_J / ps_to_s), 
        'G_pm_per_q_SI': (G_pm / n_qpoints) * (meV_to_J / ps_to_s), 
        
        # C_m, C_p: meV/K -> Joules/K
        'C_m_SI': C_m * meV_to_J,
        'C_p_SI': C_p * meV_to_J,
        
        # tau_mp: ps -> s
        'tau_mp_SI': tau_mp * ps_to_s if tau_mp != np.inf else np.inf
    }
    
    return results

if __name__ == "__main__":
    import sys
    
    filename = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
    temperature = 15.0  # K
    
    try:
        res = compute_Gmp_and_tau(filename, T0=temperature)
        
        print(f"\n--- Results at T = {res['T0']} K ---")
        
        print("\n[ standard units ]")
        print(f"G_mp (total)  = {res['G_mp']:.6e}  meV / (K * ps)")
        print(f"G_pm (total)  = {res['G_pm']:.6e}  meV / (K * ps)")
        print(f"delta_mp      = {res['delta_mp']:.6e}")
        print(f"C_m  (total)  = {res['C_m']:.6e}  meV / K")
        print(f"C_p  (total)  = {res['C_p']:.6e}  meV / K")
        print(f"tau_mp        = {res['tau_mp']:.6e}  ps")
        
        print("\n[ SI units ]")
        print(f"G_mp (total)  = {res['G_mp_SI']:.6e}  W / K")
        print(f"G_pm (total)  = {res['G_pm_SI']:.6e}  W / K")
        print(f"C_m  (total)  = {res['C_m_SI']:.6e}  J / K")
        print(f"C_p  (total)  = {res['C_p_SI']:.6e}  J / K")
        print(f"tau_mp        = {res['tau_mp_SI']:.6e}  s")
        
        print("\n[ per q-point ] (Often physically relevant for cell-scaling)")
        print(f"G_mp/q  = {res['G_mp_per_q']:.6e} meV / (K*ps)  |  {res['G_mp_per_q_SI']:.6e} W/K")
        print(f"G_pm/q  = {res['G_pm_per_q']:.6e} meV / (K*ps)  |  {res['G_pm_per_q_SI']:.6e} W/K")
        print(f"C_m/q   = {res['C_m_per_q']:.6e} meV / K       |  {res['C_m_SI']/len(np.unique(pd.read_csv(filename, comment='#')['q_idx'])):.6e} J/K")
        
        print("-" * 40)
        
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")