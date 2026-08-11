import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scienceplots
from matplotlib.colors import LinearSegmentedColormap

# Apply SciencePlots style matching your existing figure setup
plt.style.use("science")

def get_contrast_coolwarm():
    """
    Creates a diverging Blue-Grey-Red colormap where the neutral midpoint 
    is a medium charcoal grey (#A0A0A0FF) instead of white, ensuring high 
    contrast on white plot backgrounds.
    """
    colors = [
        "#0571b0",  # -1.0 : Deep Blue
        "#92c5de",  # -0.5 : Soft Blue
        "#A0A0A0FF",  #  0.0 : Medium Charcoal Grey (Visible on white!)
        "#f4a582",  # +0.5 : Soft Red/Orange
        "#ca0020",  # +1.0 : Deep Red
    ]
    return LinearSegmentedColormap.from_list("coolwarm_dark_center", colors)

def _detect_column(df, candidate_names):
    """Utility to find matching column names regardless of exact casing or formatting."""
    df_cols_lower = {col.lower(): col for col in df.columns}
    for cand in candidate_names:
        if cand.lower() in df_cols_lower:
            return df_cols_lower[cand.lower()]
    for cand in candidate_names:
        cand_lower = cand.lower()
        for col_lower, orig_col in df_cols_lower.items():
            if cand_lower in col_lower:
                return orig_col
    return None

def load_sparse_csv(path):
    """Loads a SpinPhony.py sparse BZ output CSV."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def plot_lifetime_vs_pam(
    sparse_csv,
    output_dir,
    pam_candidates=["phon_AM_z_hbar", "phonon_Lz", "Lz_ph", "L_z"],
    rate_candidates=["scattering_rate", "inverse_lifetime", "tau_inv", "gamma"],
    lifetime_candidates=["lifetime", "tau", "tau_ps"],
    energy_col="energy_meV",
    use_log_y=True,
    save_name="scatter_lifetime_vs_pam.png"
):
    """
    Generates a scatter plot of Scattering Rate (tau^-1) vs Phonon Angular Momentum (L_z).
    Markers are colored by mode energy to reveal where avoided crossings occur.
    """
    df = load_sparse_csv(sparse_csv)

    # 1. Detect PAM column
    col_pam = _detect_column(df, pam_candidates)
    if col_pam is None:
        raise KeyError(f"Could not find PAM column in {sparse_csv}")

    # 2. Detect or Calculate Scattering Rate (tau^-1)
    col_rate = _detect_column(df, rate_candidates)
    if col_rate is None:
        col_tau = _detect_column(df, lifetime_candidates)
        if col_tau is None:
            raise KeyError("Could not find scattering rate or lifetime column.")
        print(f"Scattering rate not found. Inverting lifetime column: '{col_tau}'")
        # Avoid division by zero for infinite lifetimes
        valid_mask = df[col_tau] > 0
        df = df[valid_mask].copy()
        df['calculated_tau_inv'] = 1.0 / df[col_tau]
        col_rate = 'calculated_tau_inv'

    # 3. Detect Energy column for coloring
    col_energy = _detect_column(df, [energy_col, "energy", "omega"])
    if col_energy is None:
        raise KeyError(f"Could not find energy column in {sparse_csv}")

    # Extract arrays
    x_pam = df[col_pam].to_numpy()
    y_rate = df[col_rate].to_numpy()
    c_energy = df[col_energy].to_numpy()

    # Setup Plot
    fig, ax = plt.subplots(figsize=(12 / 2.52, 10 / 2.52))

    # Scatter plot
    scatter = ax.scatter(
        x_pam, 
        y_rate, 
        c=c_energy, 
        cmap='viridis', 
        s=10, 
        alpha=0.7, 
        edgecolors='none'
    )

    # Aesthetics and Scaling
    if use_log_y:
        ax.set_yscale('log')
        
    ax.set_xlim(-1.1, 1.1)
    ax.set_xlabel(r"Phonon Angular Momentum $L_z$ ($\hbar$)", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"Scattering Rate $\tau^{-1}$", fontsize=12, fontweight="bold")
    
    # Add zero-line for PAM
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

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

    # Note: Update these paths to match where your sparse data is stored.
    # Hybrid CrI3 Sparse Mesh
    cri3_sparse_csv = "Outputs/CrI3_Path_Hyrbid_sparse/hybrid_path_properties.csv"
    
    if os.path.exists(cri3_sparse_csv):
        print("Plotting Lz vs Lifetime for CrI3...")
        plot_lifetime_vs_pam(
            sparse_csv=cri3_sparse_csv,
            output_dir="Outputs/CrI3_Path_Hyrbid_sparse/",
            save_name="cri3_lz_vs_scattering_rate.png"
        )
    else:
        print(f"File not found: {cri3_sparse_csv}. Please verify the sparse mesh output path.")