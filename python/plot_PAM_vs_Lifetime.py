import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

# Apply SciencePlots style for publication-ready figures
plt.style.use("science")

def load_spinphony_csv(path):
    """
    Loads the updated SpinPhony.py output CSV.
    Safely skips the high-symmetry labels comment line.
    """
    with open(path) as f:
        first_line = f.readline().strip()

    if first_line.startswith("# path_labels:"):
        df = pd.read_csv(path, skiprows=1)
    else:
        df = pd.read_csv(path)

    # Clean whitespace from column names
    df.columns = df.columns.str.strip()
    return df

def plot_pam_vs_scattering_rate(
    csv_path, 
    output_dir,
    x_col="phon_AM_z_hbar",
    y_col="gamma_ps-1",
    c_col="energy_meV",
    save_name="scatter_pam_vs_gamma.png"
):
    """
    Generates a scatter plot of Scattering Rate vs Phonon Angular Momentum.
    Colored by energy to isolate the optical gap modes.
    """
    df = load_spinphony_csv(csv_path)

    # Verify required columns exist
    for col in [x_col, y_col, c_col]:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found. Available: {list(df.columns)}")

    x_pam = df[x_col].to_numpy()
    y_gamma = df[y_col].to_numpy()
    c_energy = df[c_col].to_numpy()

    # Filter out absolute zero/negative scattering rates to allow log scaling
    valid_mask = y_gamma > 0
    x_pam = x_pam[valid_mask]
    y_gamma = y_gamma[valid_mask]
    c_energy = c_energy[valid_mask]

    # Setup Plot
    fig, ax = plt.subplots(figsize=(12 / 2.52, 10 / 2.52))

    # Scatter Plot
    # Using 'viridis' to clearly separate low-energy (dark purple) from high-energy (yellow)
    scatter = ax.scatter(
        x_pam, 
        y_gamma, 
        c=c_energy, 
        cmap='viridis', 
        s=15, 
        alpha=0.8, 
        edgecolors='none'
    )

    # Scale and Labels
    ax.set_yscale('log')
    ax.set_xlim(-1.1, 1.1)
    ax.set_xlabel(r"Phonon Angular Momentum $L_z$ ($\hbar$)", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"Scattering Rate $\Gamma$ (ps$^{-1}$)", fontsize=12, fontweight="bold")

    # Add reference zero-line for PAM
    ax.axvline(0, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(r"Energy (meV)", fontsize=12, fontweight="bold")

    fig.tight_layout()
    plt.show()

    # Save
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, save_name)
        fig.savefig(save_path, dpi=300)
        print(f"Plot saved to '{save_path}'")

    return fig, ax

if __name__ == "__main__":
    
    target_csv = "Outputs/CrI3_Path_Hyrbid_sparse/hybrid_path_lifetimes.csv"
    out_dir = "Outputs/CrI3_Path_Hyrbid_sparse/properties"
    
    if os.path.exists(target_csv):
        print(f"Plotting PAM vs Scattering Rate for {target_csv}...")
        plot_pam_vs_scattering_rate(
            csv_path=target_csv,
            output_dir=out_dir
        )
    else:
        print(f"File not found: {target_csv}. Please verify the path.")