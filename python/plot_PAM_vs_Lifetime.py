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

def plot_pam_vs_lifetime(
    csv_path, 
    output_dir,
    x_col="phon_AM_z_hbar",
    y_col="tau_ps",
    c_col="energy_meV",
    save_name="scatter_pam_vs_tau.png"
):
    """
    Generates a scatter plot of Lifetime vs Phonon Angular Momentum.
    Colored by energy to isolate the optical gap modes.
    """
    df = load_spinphony_csv(csv_path)

    # Verify required columns exist
    for col in [x_col, y_col, c_col]:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found. Available: {list(df.columns)}")

    x_pam = df[x_col].to_numpy()
    y_tau = df[y_col].to_numpy()
    c_energy = df[c_col].to_numpy()

    # Filter out absolute zero/negative lifetimes or infinities to allow log scaling
    valid_mask = (y_tau > 0) & np.isfinite(y_tau)
    x_pam = x_pam[valid_mask]
    y_tau = y_tau[valid_mask]
    c_energy = c_energy[valid_mask]

    # Setup Plot
    fig, ax = plt.subplots(figsize=(12 / 2.52, 10 / 2.52))

    # Scatter Plot
    # Note: Plotted as x=Lifetime, y=PAM based on your modifications
    scatter = ax.scatter(
        y_tau, x_pam, 
        c=c_energy, 
        cmap='jet', 
        s=15, 
        alpha=1.0, 
        edgecolors='none'
    )

    # Scale and Labels
    ax.set_xscale('log')
    ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel(r"Phonon Angular Momentum $L_z$ ($\hbar$)", fontsize=12, fontweight="bold")
    ax.set_xlabel(r"Lifetime $\tau$ (ps)", fontsize=12, fontweight="bold")

    # Add reference zero-line for PAM (changed to axhline since PAM is on the y-axis)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(r"Energy (meV)", fontsize=12, fontweight="bold")
    
    ax.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    plt.show(block=False)

    # Save
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, save_name)
        fig.savefig(save_path, dpi=300)
        print(f"Plot saved to '{save_path}'")

    return fig, ax

def plot_energy_vs_lifetime(
    csv_path, 
    output_dir,
    x_col="energy_meV",
    y_col="tau_ps",
    c_col="phon_AM_z_hbar",
    save_name="scatter_energy_vs_tau.png"
):
    """
    Generates a scatter plot of Energy vs Lifetime.
    Colored by Phonon Angular Momentum (PAM) using a diverging colormap.
    """
    df = load_spinphony_csv(csv_path)

    # Verify required columns exist
    for col in [x_col, y_col, c_col]:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found. Available: {list(df.columns)}")

    x_energy = df[x_col].to_numpy()
    y_tau = df[y_col].to_numpy()
    c_pam = df[c_col].to_numpy()

    # Filter out absolute zero/negative lifetimes or infinities to allow log scaling
    valid_mask = (y_tau > 0) & np.isfinite(y_tau)
    x_energy = x_energy[valid_mask]
    y_tau = y_tau[valid_mask]
    c_pam = c_pam[valid_mask]

    # Setup Plot
    fig, ax = plt.subplots(figsize=(12 / 2.52, 10 / 2.52))

    # Scatter Plot
    # Using 'coolwarm' for a diverging representation of PAM (-1 to +1)
    scatter = ax.scatter(
        x_energy, 
        y_tau, 
        c=c_pam, 
        cmap='coolwarm', 
        vmin=-1.0,   # Fix bounds to theoretical min PAM
        vmax=1.0,    # Fix bounds to theoretical max PAM
        s=15, 
        alpha=1.0, 
        edgecolors='none'
    )

    # Scale and Labels
    ax.set_yscale('log')
    ax.set_xlabel(r"Energy (meV)", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"Lifetime $\tau$ (ps)", fontsize=12, fontweight="bold")

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(r"Phonon Angular Momentum $L_z$ ($\hbar$)", fontsize=12, fontweight="bold")
    
    ax.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    plt.show(block=False)

    # Save
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, save_name)
        fig.savefig(save_path, dpi=300)
        print(f"Plot saved to '{save_path}'")

    return fig, ax

if __name__ == "__main__":
    
    target_csv = "Outputs/CrI3_Path_Hyrbid_Full/hybrid_path_lifetimes.csv"
    out_dir = "Outputs/CrI3_Path_Hyrbid_Full/properties"
    
    if os.path.exists(target_csv):
        print(f"Plotting PAM vs Lifetime for {target_csv}...")
        plot_pam_vs_lifetime(
            csv_path=target_csv,
            output_dir=out_dir
        )
        
        print(f"Plotting Energy vs Lifetime for {target_csv}...")
        plot_energy_vs_lifetime(
            csv_path=target_csv,
            output_dir=out_dir
        )

        df_stats = load_spinphony_csv(target_csv)
        valid_tau = df_stats[df_stats['tau_ps'] > 0] # Filter out invalid lifetimes
        
        avg_pos = valid_tau.loc[valid_tau['phon_AM_z_hbar'] > 0.01, 'tau_ps'].mean()
        avg_neg = valid_tau.loc[valid_tau['phon_AM_z_hbar'] < -0.01, 'tau_ps'].mean()

        # PAM weighted with lifetime
        weighted_avg_pos = np.average(
            valid_tau.loc[valid_tau['phon_AM_z_hbar'] > 0.01, 'tau_ps'], 
            weights=valid_tau.loc[valid_tau['phon_AM_z_hbar'] > 0.01, 'phon_AM_z_hbar']
        )
        weighted_avg_neg = np.average(
            valid_tau.loc[valid_tau['phon_AM_z_hbar'] < -0.01, 'tau_ps'], 
            weights=valid_tau.loc[valid_tau['phon_AM_z_hbar'] < -0.01, 'phon_AM_z_hbar']
        )

        # Print statistics
        print(f"Weighted average lifetime (PAM > +0.01): {weighted_avg_pos:.4f} ps")
        print(f"Weighted average lifetime (PAM < -0.01): {weighted_avg_neg:.4f} ps")
        
        print(f"Average lifetime (PAM > +0.01): {avg_pos:.4f} ps")
        print(f"Average lifetime (PAM < -0.01): {avg_neg:.4f} ps")
        
        # Keeps windows open if executing from command line
        plt.show()
    else:
        print(f"File not found: {target_csv}. Please verify the path.")