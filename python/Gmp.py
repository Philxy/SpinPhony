import os
import numpy as np
import matplotlib.pyplot as plt

def plot_spin_lattice_coupling(csv_filename="Outputs/G_mp_temperature_scan.csv", 
                                unit_cell_volume_A3=None, 
                                save_plot="Outputs/G_mp_vs_temperature.png"):
    """
    Reads G_mp(T) from a CSV file and plots it across temperature.
    
    Parameters:
    -----------
    csv_filename : str
        Path to the calculated CSV file.
    unit_cell_volume_A3 : float, optional
        Volume of the primitive unit cell in cubic Angstroms (Å^3).
        If provided, converts G_mp to SI units [W / (m^3 * K)].
    save_plot : str, optional
        Path to output PNG image.
    """
    if not os.path.exists(csv_filename):
        raise FileNotFoundError(f"Could not find '{csv_filename}'. Ensure the SpinPhony simulation ran first.")

    # Load data from CSV
    data = np.loadtxt(csv_filename, delimiter=',', skiprows=1)
    temp = data[:, 0]
    g_mp_raw = data[:, 1]  # Units: meV / (K * ps) per unit cell

    fig, ax = plt.subplots(figsize=(7, 5))

    if unit_cell_volume_A3 is not None:
        # Conversion factors:
        # 1 meV/ps = 1.602176634e-10 Watts
        # 1 Å^3 = 1e-30 m^3
        # G_SI = (g_mp_raw * 1.602176634e-10 W) / (unit_cell_volume_A3 * 1e-30 m^3 * K)
        meV_ps_to_watts = 1.602176634e-10
        volume_m3 = unit_cell_volume_A3 * 1e-30
        
        g_mp_si = (g_mp_raw * meV_ps_to_watts) / volume_m3
        
        ax.plot(temp, g_mp_si, color='#d62728', lw=2.5, label=r'$G_{\mathrm{mp}}(T)$')
        ax.set_ylabel(r'Coupling Constant $G_{\mathrm{mp}}$ ($\mathrm{W} / \mathrm{m}^3 \cdot \mathrm{K}$)', fontsize=12)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    else:
        ax.plot(temp, g_mp_raw, color='#d62728', lw=2.5, label=r'$G_{\mathrm{mp}}(T)$')
        ax.set_ylabel(r'$G_{\mathrm{mp}}$ ($\mathrm{meV} / (\mathrm{K} \cdot \mathrm{ps} \cdot \mathrm{cell})$)', fontsize=12)

    ax.set_xlabel('Temperature (K)', fontsize=12)
    ax.set_title('3TM Spin-Lattice Coupling Constant', fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_yscale('log')

    plt.tight_layout()
    
    if save_plot:
        os.makedirs(os.path.dirname(save_plot), exist_ok=True)
        plt.savefig(save_plot, dpi=300)
        print(f"Plot saved to '{save_plot}'")

    plt.show()

if __name__ == "__main__":
    # Example usage: Pass unit_cell_volume_A3 if you want SI unit conversion
    plot_spin_lattice_coupling(
        csv_filename="Outputs/G_mp_temperature_scan.csv",
        unit_cell_volume_A3=269.0,  # Replace with e.g. 23.5 for a system volume in Å^3
        save_plot="Outputs/G_mp_vs_temperature.png"
    )