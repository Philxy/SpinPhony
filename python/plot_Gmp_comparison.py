import os
import numpy as np
import matplotlib.pyplot as plt

def plot_spin_lattice_coupling_comparison(min_sigmas=(0.01, 0.05, 0.1, 0.2, 0.5),
                                           out_dir_template="Outputs/CrI3_minsig{}",
                                           csv_name="G_mp_temperature_scan.csv",
                                           unit_cell_volume_A3=None,
                                           save_plot="Outputs/G_mp_vs_temperature_comparison.png"):
    """
    Overlays G_mp(T) curves from multiple min_sigma runs for direct comparison.

    Parameters:
    -----------
    min_sigmas : iterable of values used to build each run's output directory name
    out_dir_template : str
        Format string with a single '{}' placeholder for the min_sigma value.
    csv_name : str
        Filename of the G_mp CSV inside each run's output directory.
    unit_cell_volume_A3 : float, optional
        Volume of the primitive unit cell in cubic Angstroms (Å^3).
        If provided, converts G_mp to SI units [W / (m^3 * K)].
    save_plot : str, optional
        Path to output PNG image.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.0, 0.85, len(min_sigmas)))

    meV_ps_to_watts = 1.602176634e-10

    for color, sigma in zip(colors, min_sigmas):
        csv_filename = os.path.join(out_dir_template.format(sigma), csv_name)

        if not os.path.exists(csv_filename):
            print(f"Warning: '{csv_filename}' not found, skipping min_sigma={sigma}.")
            continue

        data = np.loadtxt(csv_filename, delimiter=',', skiprows=1)
        temp = data[:, 0]
        g_mp_raw = data[:, 1]  # Units: meV / (K * ps) per unit cell

        if unit_cell_volume_A3 is not None:
            volume_m3 = unit_cell_volume_A3 * 1e-30
            g_mp = (g_mp_raw * meV_ps_to_watts) / volume_m3
        else:
            g_mp = g_mp_raw

        ax.plot(temp, g_mp, color=color, lw=2.5, label=rf'$\sigma_{{\min}}$ = {sigma} meV')

    if unit_cell_volume_A3 is not None:
        ax.set_ylabel(r'Coupling Constant $G_{\mathrm{mp}}$ ($\mathrm{W} / \mathrm{m}^3 \cdot \mathrm{K}$)', fontsize=12)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    else:
        ax.set_ylabel(r'$G_{\mathrm{mp}}$ ($\mathrm{meV} / (\mathrm{K} \cdot \mathrm{ps} \cdot \mathrm{cell})$)', fontsize=12)

    ax.set_xlabel('Temperature (K)', fontsize=12)
    ax.set_title(r'3TM Spin-Lattice Coupling Constant vs. $\sigma_{\min}$', fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_yscale('log')
    ax.legend(fontsize=10)

    plt.tight_layout()

    if save_plot:
        os.makedirs(os.path.dirname(save_plot), exist_ok=True)
        plt.savefig(save_plot, dpi=300)
        print(f"Plot saved to '{save_plot}'")

    plt.show()

if __name__ == "__main__":
    plot_spin_lattice_coupling_comparison(
        min_sigmas=(0.01, 0.05, 0.1, 0.2, 0.5),
        out_dir_template="Outputs/CrI3_minsig{}",
        unit_cell_volume_A3=269.0,  # Replace with your CrI3 unit cell volume in Å^3
        save_plot="Outputs/G_mp_vs_temperature_comparison.png",
    )
