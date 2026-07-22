import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp, cumulative_trapezoid
import os
import io

data = """
-14.232209737827773, -0.0008683068017365791
-10.674157303370825, -0.0008683068017365791
-7.6779026217229, 0.0002894356005788967
-5.243445692883961, -0.0008683068017365791
-0.18726591760303127, 0.00028943560057886897
-0.18726591760303155, -0.058755426917510845
-0.1872659176030421, -0.11201157742402312
-0.1872659176030428, -0.16526772793053549
-1.8117433919970204e-14, -0.18958031837916062
-0.18726591760305325, -0.22315484804630972
-0.1872659176030346, -0.27062228654124465
0.18726591760297637, -0.29956584659913177
59.737827715355785, -0.8888567293777137
61.985018726591754, -0.8992764109985532
64.2322097378277, -0.9293777134587557
66.29213483146066, -0.9316931982633865
69.1011235955056, -0.9351664254703331
71.53558052434455, -0.9467438494934879
74.3445692883895, -0.9479015918958034
76.21722846441946, -0.9467438494934881
77.90262172284643, -0.9617945007235893
81.27340823970036, -0.9617945007235893
84.4569288389513, -0.9652677279305355
87.82771535580524, -0.9698986975397976
92.88389513108613, -0.9745296671490593
0.936329588014951, -0.38755426917510866
3.1835205992509277, -0.4639652677279306
5.805243445692873, -0.4975397973950797
9.550561797752787, -0.5345875542691751
12.921348314606716, -0.5646888567293777
15.917602996254674, -0.5982633863965269
19.85018726591759, -0.6376266280752534
23.408239700374516, -0.6619392185238785
26.217228464419456, -0.6920405209840812
29.5880149812734, -0.710564399421129
31.64794007490637, -0.7290882778581766
33.707865168539314, -0.7395079594790162
36.70411985018725, -0.7591895803183792
38.014981273408225, -0.7696092619392187
42.32209737827716, -0.8008683068017368
40.07490636704118, -0.7846599131693202
49.625468164794, -0.8541244573082492
45.318352059925076, -0.8298118668596237
54.11985018726591, -0.8738060781476125
56.55430711610485, -0.8888567293777139
1.3108614232209557, -0.40955137481910275
2.2471910112359312, -0.43502170767004344
0.3745318352059765, -0.31461649782923307
0.3745318352059555, -0.3342981186685964
0.7490636704119515, -0.3539797395079596
1.6853932584269298, -0.37134587554269183
27.340823970037437, -0.6642547033285096
21.34831460674155, -0.6109985528219973
29.5880149812734, -0.6885672937771348
16.292134831460665, -0.5704775687409551
22.659176029962534, -0.6410998552821998
10.674157303370771, -0.5102749638205502
14.981273408239694, -0.5496382054992764
12.172284644194738, -0.5299565846599132
"""

# --- Physical Constants ---
k_B = 1.380649e-23              # J/K
meV_to_J = 1.602176634e-22      # Joules per meV

# --- Material Parameters (CrI3) ---
V_cell_m3 = 2.66e-28            # Primitive cell volume in m^3
S_cell = 3.0                    # Total spin per primitive unit cell (2 Cr atoms * 3/2)

# --- Laser / Pulse Parameters ---
F_inc_mJ_cm2 = 0.15             # Incident fluence in mJ/cm^2 (From Padmanabhan et al.)
F_inc = F_inc_mJ_cm2 * 10.0     # Convert to J/m^2
R = 0.25                        # Optical reflectivity at 800 nm
delta_opt = 40e-9               # Optical penetration depth in meters (~40 nm)


def calculate_bosonic_specific_heat_grid(energies_meV, temp_grid):
    print(f"Pre-computing specific heat for {energies_meV.shape[1]} branches...")
    cv_grid = np.zeros(len(temp_grid))
    N_qpoints = energies_meV.shape[0]
    
    E_flat = energies_meV.flatten()
    E_flat = E_flat[E_flat > 1e-5]  # Skip Gamma point zero-energy modes
    E_J = E_flat * meV_to_J         # Energy in Joules
    
    for i, T in enumerate(temp_grid):
        if T < 1e-3:
            cv_grid[i] = 0.0
            continue
            
        exponent = E_J / (k_B * T)
        exponent = np.clip(exponent, a_min=None, a_max=700.0) 
        
        exp_term = np.exp(exponent)
        denominator = exp_term - 1.0
        term = (E_J * E_J * exp_term) / (denominator * denominator)
        sum_term = np.sum(term)
        
        pre_factor = 1.0 / (k_B * T * T * N_qpoints * V_cell_m3)
        cv_grid[i] = pre_factor * sum_term
        
    return cv_grid

def calculate_magnon_occupation_grid(energies_meV, temp_grid):
    print("Pre-computing magnon occupation distribution...")
    mag_occ_grid = np.zeros(len(temp_grid))
    N_qpoints = energies_meV.shape[0]
    
    E_flat = energies_meV.flatten()
    E_flat = E_flat[E_flat > 1e-5] 
    E_J = E_flat * meV_to_J
    
    for i, T in enumerate(temp_grid):
        if T < 1e-3:
            mag_occ_grid[i] = 0.0
            continue
            
        exponent = E_J / (k_B * T)
        exponent = np.clip(exponent, a_min=None, a_max=700.0)
        
        occupation = 1.0 / (np.exp(exponent) - 1.0)
        total_magnon_occupation = np.sum(occupation)
        
        mag_occ_grid[i] = total_magnon_occupation / N_qpoints
        
    return mag_occ_grid

def laser_power(t):
    """Energy is injected instantaneously as a T jump at t=0; continuous source is 0."""
    return 0.0

def main():
    os.makedirs("Outputs", exist_ok=True)

    # --- Load Magnon-Phonon Coupling (Gmp) Temperature Dependence ---
    gmp_filepath = "Outputs/G_mp_temperature_scan.csv"
    if not os.path.exists(gmp_filepath):
        # Create a fallback/dummy file using the example provided if it doesn't exist
        with open(gmp_filepath, "w") as f:
            f.write("Temperature_K,G_mp_meV_per_K_ps_per_cell\n")
            f.write("10.00,2.000000e-04\n")
            f.write("100.00,2.886391e-04\n")
            f.write("108.08,3.327675e-04\n")
            f.write("1000.00,3.500000e-04\n")
            
    print(f"Loading Gmp(T) from {gmp_filepath}...")
    gmp_data = np.loadtxt(gmp_filepath, delimiter=",", skiprows=1)
    T_gmp = gmp_data[:, 0]
    Gmp_raw = gmp_data[:, 1]

    
    
    # Convert from [meV / (K * ps * cell)] to [W / (m^3 * K)]
    # Conversion factor = (J / meV) / (s / ps * m^3 / cell)
    unit_conversion_factor = meV_to_J / (1e-12 * V_cell_m3)
    Gmp_converted = Gmp_raw * unit_conversion_factor
    
    # Create interpolation function for Gmp(T)
    Gmp_func = interp1d(T_gmp, Gmp_converted, kind='linear', fill_value="extrapolate", bounds_error=False)


    if not os.path.exists("Outputs/w_mag_grid.csv") or not os.path.exists("Outputs/w_phon_grid.csv"):
        raise FileNotFoundError("Could not find dispersion CSVs. Export them from SpinPhony first.")
        
    w_mag = np.loadtxt("Outputs/w_mag_grid.csv", delimiter=",")
    w_phon = np.loadtxt("Outputs/w_phon_grid.csv", delimiter=",")
    
    if w_mag.ndim == 1: w_mag = w_mag.reshape(-1, 1)
    if w_phon.ndim == 1: w_phon = w_phon.reshape(-1, 1)

    # Shift the zero-energy magnon mode by the experimental anisotropy gap (~0.4 meV)
    w_mag = w_mag + 0.10

    # Pre-compute and interpolate grids
    T_grid = np.linspace(1.0, 1000.0, 2000)
    
    Cm_grid = calculate_bosonic_specific_heat_grid(w_mag, T_grid)
    Cp_grid = calculate_bosonic_specific_heat_grid(w_phon, T_grid)
    mag_occ_grid = calculate_magnon_occupation_grid(w_mag, T_grid)
    
    Cm_func = interp1d(T_grid, Cm_grid, kind='cubic', fill_value="extrapolate", bounds_error=False)
    Cp_func = interp1d(T_grid, Cp_grid, kind='cubic', fill_value="extrapolate", bounds_error=False)
    MagOcc_func = interp1d(T_grid, mag_occ_grid, kind='cubic', fill_value="extrapolate", bounds_error=False)

    # --- Calculate Initial Phonon Temperature Jump ---
    T_initial_mag = 15.0  # Base experimental temperature

    # 1. Volumetric absorbed energy (Joules / m^3)
    E_abs = (F_inc * (1.0 - R)) / delta_opt
    
    # 2. Build Internal Energy Grid U_p(T) = int C_p dT
    Up_grid = cumulative_trapezoid(Cp_grid, T_grid, initial=0.0)
    Up_func = interp1d(T_grid, Up_grid, kind='cubic', fill_value="extrapolate", bounds_error=False)
    
    # 3. Inverse function: T_p as a function of U_p
    T_from_Up_func = interp1d(Up_grid, T_grid, kind='cubic', fill_value="extrapolate", bounds_error=False)

    # 4. Calculate new T_p(0^+)
    U_initial = Up_func(T_initial_mag)
    U_final = U_initial + E_abs
    T_initial_ph = float(T_from_Up_func(U_final))

    print(f"Base Temperature (Magnons): {T_initial_mag} K")
    print(f"Absorbed Laser Energy Density: {E_abs:.2e} J/m^3")
    print(f"Calculated Initial Phonon Temperature: {T_initial_ph:.2f} K")
    print("-" * 50)

    # Quick code snippet to verify phonon saturation
    C_p_highT = Cp_func(1000.0)  # Evaluate at 1000 K
    C_p_DP = (3 * 8 * k_B) / V_cell_m3
    print(f"Calculated Cp(1000K): {C_p_highT:.3e} | Dulong-Petit Limit: {C_p_DP:.3e}")
    

    # ODE System (2-Temperature Model)
    def derivatives(t, y):
        T_m, T_p = y[0], y[1]
        
        T_m = max(T_m, 1.0)
        T_p = max(T_p, 1.0)
        
        Cm = Cm_func(T_m)
        Cp = Cp_func(T_p)
        
        # Evaluate Temperature Dependent Gmp 
        # (Using T_p here since lattice phonons govern the coupling bath strength)
        Gmp_current = Gmp_func(T_p)
        
        dTm_dt = (1.0 / Cm) * ( Gmp_current * (T_p - T_m) )
        dTp_dt = (1.0 / Cp) * (-Gmp_current * (T_p - T_m) + laser_power(t))
        
        return [dTm_dt, dTp_dt]

    # Integrate
    y0 = [T_initial_mag, T_initial_ph]
    
    # Extended integration out to 95 ps to cover the full experimental data window
    t_span = (0.0, 95e-12) 
    t_eval = np.linspace(t_span[0], t_span[1], 5000)
    
    print("Integrating 2TM ODEs...")
    sol = solve_ivp(derivatives, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10)
    
    t_ps = sol.t * 1e12
    Tm_sol = sol.y[0]
    Tp_sol = sol.y[1]

    mag_occ_t = MagOcc_func(Tm_sol)
    mag_occ_initial = MagOcc_func(T_initial_mag)
    delta_n = mag_occ_t - mag_occ_initial
    
    M_over_M0 = 1.0 - (delta_n / S_cell)
    M_over_M0 = np.maximum(0.0, M_over_M0)

    # --- Experimental Data Processing & Scaling ---
    raw_data = np.loadtxt(io.StringIO(data), delimiter=",")
    data_sorted = raw_data[raw_data[:, 0].argsort()]
    t_data = data_sorted[:, 0]
    m_data = data_sorted[:, 1]

    # 1. Establish the pre-pulse baseline (t < 0) for the experiment
    baseline_mask = t_data < 0
    exp_baseline = np.mean(m_data[baseline_mask]) if np.any(baseline_mask) else m_data[0]
    
    # Shift experimental data so baseline = 0
    m_data_shifted = m_data - exp_baseline
    
    # 2. Extract maximum depths (minimum values) of both curves
    exp_min = np.min(m_data_shifted)
    sim_min = np.min(M_over_M0) - 1.0  # Simulation minimum drop relative to 1.0
    
    # 3. Apply the scaling factor to match the experimental depth to the simulation depth
    scaling_factor = sim_min / exp_min
    m_data_scaled = 1.0 + m_data_shifted * scaling_factor
    # ---------------------------------------------

    # Output Data
    output_file = "Outputs/2TM_CrI3_Dynamics.csv"
    np.savetxt(output_file, np.column_stack((t_ps, Tm_sol, Tp_sol, M_over_M0)), 
               delimiter=",", header="Time(ps),T_mag(K),T_phon(K),M_over_M0", comments="")
    print(f"Dynamics saved to {output_file}")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1.5]})
    
    ax1.plot(t_ps, Tp_sol, color='#1f77b4', lw=2.5, label='Phonons ($T_p$)')
    ax1.plot(t_ps, Tm_sol, color='#d62728', lw=2.5, linestyle='--', label='Magnons ($T_m$)')
    
    ax1.set_ylabel('Temperature (K)', fontsize=12, fontweight='bold')
    ax1.set_title(f'CrI$_3$ 2-Temperature Dynamics ($G_{{mp}}(T)$)', fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=11)

    # Plot scaled Experimental Data vs Simulation
    ax2.scatter(
        t_data,
        m_data_scaled,
        color="#1f77b4",
        s=25,
        alpha=0.8,
        edgecolor="k",
        linewidth=0.5,
        label="Exp. TRPR (Scaled to Sim)",
        zorder=3
    )

    ax2.plot(t_ps, M_over_M0, color='#2ca02c', lw=2.5, label=r'Sim. $M(t)/M(t_0)$', zorder=2)
    
    ax2.set_xlabel('Time (ps)', fontsize=12, fontweight='bold')
    ax2.set_ylabel(r'$M(t) / M(t_0)$', fontsize=12, fontweight='bold')
    ax2.set_xlim(-5, 95) # Constrain view to show pre-pulse and long-tail
    
    # Dynamically set y-limits based on the depth of the curve
    min_y = min(np.min(M_over_M0), np.min(m_data_scaled))
    ax2.set_ylim(min_y - 0.05, 1.05)
    
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig("Outputs/2TM_CrI3_Plot_with_Mag.png", dpi=300)
    print("Plot saved to Outputs/2TM_CrI3_Plot_with_Mag.png")
    plt.show()

if __name__ == "__main__":
    main()