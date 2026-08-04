import os
import numpy as np
import matplotlib.pyplot as plt

def plot_spinphony_dashboard(csv_filename="Outputs/G_mp_temperature_scan.csv", 
                             unit_cell_volume_A3=None,
                             fit_max_temp=50.0,
                             save_plot="Outputs/SpinPhony_Dashboard.png"):
    """
    Reads SpinPhony output and plots G_mp, C_s, C_l, and tau across temperature.
    Includes a scaling check for G_mp in the low-temperature limit.
    """
    if not os.path.exists(csv_filename):
        raise FileNotFoundError(f"Could not find '{csv_filename}'.")

    # Load data: Temp, G_mp, C_s, C_l, tau
    data = np.loadtxt(csv_filename, delimiter=',', skiprows=1)
    temp = data[:, 0]
    g_mp_raw = data[:, 1]
    c_s  = data[:, 2]
    c_l  = data[:, 3]
    tau  = data[:, 4]

    # --- Unit Conversion for G_mp ---
    if unit_cell_volume_A3 is not None:
        # Conversion factors:
        # 1 meV/ps = 1.602176634e-10 Watts
        # 1 Å^3 = 1e-30 m^3
        meV_ps_to_watts = 1.602176634e-10
        volume_m3 = unit_cell_volume_A3 * 1e-30
        
        g_plot = (g_mp_raw * meV_ps_to_watts) / volume_m3
        g_label = r'Coupling Constant $G_{\mathrm{mp}}$ ($\mathrm{W} / \mathrm{m}^3 \cdot \mathrm{K}$)'
    else:
        g_plot = g_mp_raw
        g_label = r'$G_{\mathrm{mp}}$ ($\mathrm{meV} / (\mathrm{K} \cdot \mathrm{ps} \cdot \mathrm{cell})$)'

    # Create 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    
    # ---------------------------------------------------------
    # 1. Spin-Lattice Coupling (Top Left)
    # ---------------------------------------------------------
    ax = axs[0, 0]
    ax.plot(temp, g_plot, marker='o', color='#d62728', lw=2, label=r'Calc $G_{\mathrm{mp}}$')
    
    # Power-law fit for low-temperature regime
    valid = (temp > 0) & (temp <= fit_max_temp) & (g_plot > 0)
    temp_v = temp[valid]
    g_v = g_plot[valid]

    if len(temp_v) > 1:
        slope, intercept = np.polyfit(np.log(temp_v), np.log(g_v), 1)
        print(f"--- Low-T Scaling Check (T <= {fit_max_temp} K) ---")
        print(f"Extracted exponent: G_mp ~ T^{slope:.3f}")
        
        # Plot reference line anchored to the first valid point
        A_ref = g_v[0] / (temp_v[0] ** 1.5)
        g_ref = A_ref * (temp_v ** 1.5)
        ax.plot(temp_v, g_ref, '--', color='black', lw=1.5, label=r'Ref $\propto T^{1.5}$')

    ax.set_ylabel(g_label, fontsize=12)
    ax.set_title('3TM Spin-Lattice Coupling Constant', fontsize=13, fontweight='bold')
    ax.legend()

    # ---------------------------------------------------------
    # 2. Relaxation Time (Top Right)
    # ---------------------------------------------------------
    ax = axs[0, 1]
    ax.plot(temp, tau, marker='s', color='#2ca02c', lw=2)
    ax.set_ylabel(r'Relaxation Time $\tau$ (ps)', fontsize=12)
    ax.set_title('Macroscopic Lifetimes', fontsize=13, fontweight='bold')

    # ---------------------------------------------------------
    # 3. Spin Heat Capacity (Bottom Left)
    # ---------------------------------------------------------
    ax = axs[1, 0]
    ax.plot(temp, c_s, marker='^', color='#ff7f0e', lw=2)
    ax.set_ylabel(r'$C_s$ (meV / K$\cdot$cell)', fontsize=12)
    ax.set_title('Magnon Heat Capacity', fontsize=13, fontweight='bold')

    # ---------------------------------------------------------
    # 4. Lattice Heat Capacity (Bottom Right)
    # ---------------------------------------------------------
    ax = axs[1, 1]
    ax.plot(temp, c_l, marker='d', color='#1f77b4', lw=2)
    ax.set_ylabel(r'$C_l$ (meV / K$\cdot$cell)', fontsize=12)
    ax.set_title('Phonon Heat Capacity', fontsize=13, fontweight='bold')

    # ---------------------------------------------------------
    # Global Formatting
    # ---------------------------------------------------------
    for ax in axs.flat:
        ax.set_xlabel('Temperature (K)', fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, linestyle='--', alpha=0.5, which='both')

    plt.tight_layout()
    
    if save_plot:
        os.makedirs(os.path.dirname(save_plot), exist_ok=True)
        plt.savefig(save_plot, dpi=300)
        print(f"\nDashboard saved to '{save_plot}'")

    plt.show()

if __name__ == "__main__":
    # Example usage: Pass unit_cell_volume_A3 if you want SI unit conversion
    plot_spinphony_dashboard(
        csv_filename="Outputs/CrI3_minsig0.01/G_mp_temperature_scan.csv",
        unit_cell_volume_A3=269.0,  # Replace with e.g. 23.5 for a system volume in Å^3
        save_plot="Outputs/G_mp_vs_temperature.png"
    )