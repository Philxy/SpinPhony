import os
import numpy as np
import matplotlib.pyplot as plt

def add_scaling_check(ax, temp_array, y_array, fit_max_temp, variable_name):
    """
    Helper function to fit the low-T data, print the exponent, 
    and plot a T^(3/2) reference line.
    """
    valid = (temp_array > 0) & (temp_array <= fit_max_temp) & (y_array > 0)
    temp_v = temp_array[valid]
    y_v = y_array[valid]

    if len(temp_v) > 1:
        # Fit log(y) = log(A) + n * log(T) to find exponent 'n'
        slope, intercept = np.polyfit(np.log(temp_v), np.log(y_v), 1)
        print(f"--- {variable_name} Low-T Scaling (T <= {fit_max_temp} K) ---")
        print(f"Extracted exponent: {variable_name} ~ T^{slope:.3f}")
        
        # Plot reference line T^(3/2) anchored to the first valid point
        A_ref = y_v[0] / (temp_v[0] ** 1.5)
        y_ref = A_ref * (temp_v ** 1.5)
        ax.plot(temp_v, y_ref, '--', color='black', lw=1.5, label=r'Ref $\propto T^{1.5}$')
        ax.legend(fontsize=10)


def plot_spinphony_dashboard(csv_filename="Outputs/G_mp_temperature_scan.csv", 
                             unit_cell_volume_A3=269.0,
                             fit_max_temp=50.0,
                             save_plot="Outputs/SpinPhony_Dashboard.png"):
    """
    Reads SpinPhony output and plots G_mp, C_s, C_l, and tau across temperature.
    G_mp is explicitly converted and plotted in SI units: W / (m^3 * K).
    Includes a T^(3/2) scaling check for G_mp, C_s, and C_l.
    """
    if not os.path.exists(csv_filename):
        raise FileNotFoundError(f"Could not find '{csv_filename}'. Ensure the SpinPhony simulation ran first.")

    # Load data: Temp, G_mp, C_s, C_l, tau
    data = np.loadtxt(csv_filename, delimiter=',', skiprows=1)
    temp = data[:, 0]
    g_mp_raw = data[:, 1]  # Units: meV / (K * ps) per unit cell
    c_s  = data[:, 2]
    c_l  = data[:, 3]
    tau  = data[:, 4]

    # --- Unit Conversion for G_mp to Watts ---
    if unit_cell_volume_A3 is None:
        raise ValueError("unit_cell_volume_A3 must be provided to convert G_mp to Watts.")
        
    # 1 meV/ps = 1.602176634e-10 Watts
    # 1 Å^3 = 1e-30 m^3
    meV_ps_to_watts = 1.602176634e-10
    volume_m3 = unit_cell_volume_A3 * 1e-30
    
    g_mp_si = (g_mp_raw * meV_ps_to_watts) / volume_m3

    # Create 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    
    # ---------------------------------------------------------
    # 1. Spin-Lattice Coupling (Top Left)
    # ---------------------------------------------------------
    ax = axs[0, 0]
    ax.plot(temp, g_mp_si, marker='o', color='#d62728', lw=2, label=r'Calc $G_{\mathrm{mp}}$')
    add_scaling_check(ax, temp, g_mp_si, fit_max_temp, "G_mp")
    
    ax.set_ylabel(r'Coupling Constant $G_{\mathrm{mp}}$ ($\mathrm{W} / \mathrm{m}^3 \cdot \mathrm{K}$)', fontsize=12)
    ax.set_title('3TM Spin-Lattice Coupling Constant', fontsize=13, fontweight='bold')
    
    # Disable scientific notation on log scale to prevent overlapping labels
    if ax.get_yscale() != 'log':
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

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
    ax.plot(temp, c_s, marker='^', color='#ff7f0e', lw=2, label=r'Calc $C_s$')
    add_scaling_check(ax, temp, c_s, fit_max_temp, "C_s")
    
    ax.set_ylabel(r'$C_s$ ($\mathrm{meV} / \mathrm{K} \cdot \mathrm{cell}$)', fontsize=12)
    ax.set_title('Magnon Heat Capacity', fontsize=13, fontweight='bold')

    # ---------------------------------------------------------
    # 4. Lattice Heat Capacity (Bottom Right)
    # ---------------------------------------------------------
    ax = axs[1, 1]
    ax.plot(temp, c_l, marker='d', color='#1f77b4', lw=2, label=r'Calc $C_l$')
    add_scaling_check(ax, temp, c_l, fit_max_temp, "C_l")
    
    ax.set_ylabel(r'$C_l$ ($\mathrm{meV} / \mathrm{K} \cdot \mathrm{cell}$)', fontsize=12)
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
    # Now simply pass the file and the max temperature for the fit
    plot_spinphony_dashboard(
        csv_filename="Outputs/CrI3_minsig0.02/G_mp_temperature_scan.csv", # Update this to your local filename
        fit_max_temp=50.0,
        save_plot="Outputs/SpinPhony_Dashboard.png"
    )
