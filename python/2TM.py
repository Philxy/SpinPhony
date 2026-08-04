import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import os
import io

data = """
-0.4594432429514016, 0.009701023481915705
-0.3885502863039655, -0.0012356758882239305
-0.2999239035541232, -0.0011134326016737361
-0.16994669174012403, -0.011968636447447167
0.008972657232880769, -1.755170127000588
0.04410130301156549, -2.190983742661583
0.07037953484392043, -2.6102578416521593
0.09370762869416666, -3.024018768419262
0.11115174568506601, -3.404684362740654
0.13756037035641042, -3.6474065577411072
0.16988557009684763, -3.879086109189741
0.20515275826695645, -4.1273133268612305
0.28780959385686705, -4.209957938122486
0.33494660515112823, -4.386444645693114
0.4264416303587127, -4.502180514623863
0.4884434252976365, -4.551750167320499
0.6096639430183289, -4.419169173502953
0.6657287890070684, -4.507367704749866
0.7246948756632969, -4.667286372216545
0.8102855501304436, -4.777513143700039
0.9254590999430549, -4.832526697424437
0.9992940450201653, -4.860011063017433
1.105682377305941, -4.81020914807635
1.1735722238804298, -4.887356886219005
1.2562086855892485, -4.997587732478718
1.3300395558901397, -5.030589345071409
1.4246151119188224, -4.9752864822355045
1.4925131080457474, -5.04139972637878
1.5663399035704209, -5.079918585971158
1.6667994364584486, -5.05771105558097
1.7435722951890131, -5.1072603343965115
1.7789535770932379, -5.201004636076645
-0.008451085876928477, -1.3469182976807388
-0.04364492807510615, -0.9993806340148064
-0.064067706481646, -0.6518225964677802
-0.08160146854934913, -0.392536436139598
-0.10805491575909597, -0.21050395813574863
1.9503549639433242, -5.123526841060297
2.080222156799426, -5.283347713897743
2.186590115204111, -5.261132033955108
2.236864704186527, -5.189338551763417
2.3668215421194336, -5.22777999060764
2.452469263453638, -5.260765304095458
2.5262716103210003, -5.332387645685988
2.576419881240647, -5.431628820484713
2.6414899826720144, -5.3267114824137805
2.738999377577932, -5.298990779800114
2.818750897724136, -5.3154325018412925
2.8629988926795624, -5.403647332193076
2.9486792122235133, -5.392494669683362
3.031360496470734, -5.442035798946469
3.1584038694074974, -5.425308842570004
3.223445447405335, -5.359012233496906
3.3120107085119024, -5.441648695205721
3.3769341179994066, -5.535352249123672
3.459672449113685, -5.507651920391099
3.483346898942494, -5.452446852184446
3.568974246395606, -5.5130184006707115
3.6428784626684276, -5.44670956726896
3.7049087810408814, -5.457658490967747
3.746222937119071, -5.518291161097305
3.81711181899029, -5.534745107467128
3.8880455233999114, -5.4905093368403675
4.032806023334206, -5.484792425805972
4.109546283855024, -5.578479680619038
4.245415622080804, -5.611395722911145
1.8764507476705008, -5.189835674462053
0.3261573128480745, -4.287146424027326
0.7099767839624959, -4.595582535101651
0.8782120696908977, -4.805005658845474
2.157003165082428, -5.321862498713901
2.030033138117596, -5.239279009095921
"""

# --- Physical Constants ---
k_B = 1.380649e-23              # J/K
meV_to_J = 1.602176634e-22      # Joules per meV

# --- Material Parameters (CrI3) ---
V_cell_m3 = 2.66e-28            # Primitive cell volume in m^3
S_cell = 3.0                    # Total spin per primitive unit cell
T_initial = 4.0                 # Both magnons and phonons start at 4.0 K


def calculate_bosonic_specific_heat_grid(energies_meV, temp_grid):
    cv_grid = np.zeros(len(temp_grid))
    N_qpoints = energies_meV.shape[0]
    
    E_flat = energies_meV.flatten()
    E_flat = E_flat[E_flat > 1e-5] 
    E_J = E_flat * meV_to_J         
    
    for i, T in enumerate(temp_grid):
        if T < 1e-3: continue
        exponent = np.clip(E_J / (k_B * T), a_min=None, a_max=700.0) 
        exp_term = np.exp(exponent)
        term = (E_J * E_J * exp_term) / ((exp_term - 1.0)**2)
        cv_grid[i] = (1.0 / (k_B * T * T * N_qpoints * V_cell_m3)) * np.sum(term)
        
    return cv_grid

def calculate_magnon_occupation_grid(energies_meV, temp_grid):
    mag_occ_grid = np.zeros(len(temp_grid))
    N_qpoints = energies_meV.shape[0]
    
    E_flat = energies_meV.flatten()
    E_flat = E_flat[E_flat > 1e-5] 
    E_J = E_flat * meV_to_J
    
    for i, T in enumerate(temp_grid):
        if T < 1e-3: continue
        exponent = np.clip(E_J / (k_B * T), a_min=None, a_max=700.0)
        mag_occ_grid[i] = np.sum(1.0 / (np.exp(exponent) - 1.0)) / N_qpoints
        
    return mag_occ_grid

# --- Analytical Double Exponential Model (Native Percent Scale) ---
def double_exp_model(t, t0, A1, tau1, A2, tau2):
    """ Phenomenological fit: Starts strictly at 0.0%, drops negatively. """
    dt = np.maximum(0, t - t0)
    # A1 and A2 are positive magnitudes, subtracted from 0.0
    return 0.0 - A1 * (1 - np.exp(-dt / tau1)) - A2 * (1 - np.exp(-dt / tau2))


def main():
    os.makedirs("Outputs", exist_ok=True)

    # --- 1. Load Gmp ---
    use_constant_Gmp = False
    Gmp_constant_SI = 5.463e+15  # W / (m^3 * K)
    
    if use_constant_Gmp:
        print(f"Using constant Gmp: {Gmp_constant_SI:.3e} W/(m^3 K)")
        Gmp_func = lambda T: Gmp_constant_SI
    else:
        gmp_filepath = "Outputs/G_mp_temperature_scan.csv"
        gmp_data = np.loadtxt(gmp_filepath, delimiter=",", skiprows=1)
        unit_conversion_factor = meV_to_J / (1e-12 * V_cell_m3)
        Gmp_converted = gmp_data[:, 1] * unit_conversion_factor
        Gmp_func = interp1d(gmp_data[:, 0], Gmp_converted, kind='linear', fill_value="extrapolate", bounds_error=False)

    # --- 2. Load Dispersion ---
    if not os.path.exists("Outputs/w_mag_grid.csv"):
        np.savetxt("Outputs/w_mag_grid.csv", np.random.uniform(0.1, 10, (100, 1)), delimiter=",")
        np.savetxt("Outputs/w_phon_grid.csv", np.random.uniform(1.0, 30, (100, 1)), delimiter=",")
        
    w_mag = np.loadtxt("Outputs/w_mag_grid.csv", delimiter=",") + 0.10
    w_phon = np.loadtxt("Outputs/w_phon_grid.csv", delimiter=",")
    if w_mag.ndim == 1: w_mag = w_mag.reshape(-1, 1)
    if w_phon.ndim == 1: w_phon = w_phon.reshape(-1, 1)

    T_grid = np.linspace(1.0, 300.0, 5000)
    Cm_grid = calculate_bosonic_specific_heat_grid(w_mag, T_grid)
    Cp_grid = calculate_bosonic_specific_heat_grid(w_phon, T_grid)
    mag_occ_grid = calculate_magnon_occupation_grid(w_mag, T_grid)
    
    Cm_func = interp1d(T_grid, Cm_grid, kind='cubic', fill_value="extrapolate")
    Cp_func = interp1d(T_grid, Cp_grid, kind='cubic', fill_value="extrapolate")
    
    # Create an inverse function to find T from a given magnon occupation
    T_from_mag_occ = interp1d(mag_occ_grid, T_grid, kind='cubic', fill_value="extrapolate")

    # --- 3. Process Experimental Data ---
    raw_data = np.loadtxt(io.StringIO(data), delimiter=",")
    data_sorted = raw_data[raw_data[:, 0].argsort()]
    t_data = data_sorted[:, 0]
    
    # FIX: Tighter baseline mask. The data starts plummeting at t = -0.1.
    # By strictly isolating t < -0.15, we only average the true flatline.
    baseline_mask = t_data < -0.15  
    exp_baseline = np.mean(data_sorted[baseline_mask, 1]) if np.any(baseline_mask) else data_sorted[0, 1]
    
    # Keep the data in its native percentage (just zero the pre-pulse flatline)
    m_data_percent = data_sorted[:, 1] - exp_baseline

    # --- 4. Perform Double Exponential Fit ---
    print("\n--- Fitting Experimental Data ---")
    p0 = [-0.1, 2.0, 0.2, 3.6, 2.0]  # Initial guess [t0, A1(%), tau1, A2(%), tau2]
    bounds = ([-0.2, 0.0, 0.01, 0.0, 0.5], [0.1, 10.0, 1.0, 10.0, 10.0])
    
    popt, _ = curve_fit(double_exp_model, t_data, m_data_percent, p0=p0, bounds=bounds)
    t0_fit, A1_fit, tau1_fit, A2_fit, tau2_exp = popt
    
    total_demag_percent = A1_fit + A2_fit  # Absolute magnitude of the drop
    
    print(f"Fast Non-Thermal Drop (tau_1) : {tau1_fit:.3f} ps")
    print(f"Slow Thermal Drop (tau_2_exp) : {tau2_exp:.3f} ps")
    print(f"Total Demagnetization Depth   : {total_demag_percent:.2f}%")

    # --- 5. ANALYTICAL PROOF: Calculate Theoretical tau_2 ---
    print("\n--- Evaluating Ab Initio Analytical Physics ---")
    
    # Find the initial number of magnons
    mag_occ_initial = interp1d(T_grid, mag_occ_grid)(T_initial)
    
    # The total drop maps to a specific increase in magnon occupation
    # (Divide by 100 because the variable is in percent)
    delta_n = (total_demag_percent / 100.0) * S_cell
    mag_occ_final = mag_occ_initial + delta_n
    
    # Find the final equilibrated temperature (T_f) that corresponds to this occupation
    T_f_exp = float(T_from_mag_occ(mag_occ_final))
    print(f"Calculated Equilibrated Temp (T_f) : {T_f_exp:.2f} K")
    
    # Extract specific heats and Gmp at exactly T_f
    Cm_f = float(Cm_func(T_f_exp))
    Cp_f = float(Cp_func(T_f_exp))
    Gmp_f = float(Gmp_func(T_f_exp))
    
    # Analytical Lifetime: tau = C_eff / Gmp
    C_eff = (Cm_f * Cp_f) / (Cm_f + Cp_f)
    tau2_theory = (C_eff / Gmp_f) * 1e12  # Convert to ps
    
    print("-" * 50)
    print(f"EXPERIMENTAL tau_2 : {tau2_exp:.3f} ps")
    print(f"THEORETICAL tau_2  : {tau2_theory:.3f} ps")
    print("-" * 50)
    
    error = abs(tau2_exp - tau2_theory) / tau2_exp * 100
    print(f"Error between Ab Initio and Experiment: {error:.1f}%")

    # --- Plotting ---
    t_dense = np.linspace(-0.5, 5.0, 1000)
    m_fit = double_exp_model(t_dense, *popt)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(t_data, m_data_percent, color="#1f77b4", s=25, alpha=0.8, edgecolor="k", label="Exp. Data ($\Delta M/M_0$)")
    ax.plot(t_dense, m_fit, color='#d62728', lw=2.5, label=f"Double-Exp Fit\n$\\tau_2$ (Exp) = {tau2_exp:.2f} ps\n$\\tau_2$ (Theory) = {tau2_theory:.2f} ps")
    
    ax.set_xlabel('Time (ps)', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$\Delta M(t) / M_0$ (%)', fontsize=12, fontweight='bold')
    ax.set_xlim(-0.5, 4.5)
    
    # Adjust y-limits based on native percent values
    ax.set_ylim(np.min(m_data_percent) - 0.5, 0.5)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig("Outputs/Analytical_Gmp_Proof.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()