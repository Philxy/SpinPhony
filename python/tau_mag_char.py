import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scienceplots

plt.style.use("science")


def load_path_csv(path):
    """Loads a SpinPhony path CSV, skipping the '# path_labels:' comment line."""
    with open(path) as f:
        first = f.readline().strip()
    return pd.read_csv(path, skiprows=1) if first.startswith("# path_labels:") else pd.read_csv(path)


def plot_mag_vs_tau_pam(df, E_min=22.0, out_png=None):
    """Plots Magnon character vs tau_SLC."""
    # Using raw CSV column names 'energy_meV' and 'tau_ps'
    mask_energy = df["energy_meV"] > E_min if E_min > 0 else np.ones(len(df), dtype=bool)
    df_sub = df[mask_energy]

    fig, ax = plt.subplots(figsize=(6/2.52, 14/5.52))


    if df_sub.empty:
        ax.text(0.5, 0.5, f"No modes found with E > {E_min} meV.",
                ha="center", va="center", transform=ax.transAxes)
        plt.show()
        return fig, ax

    if "mag_character" not in df_sub.columns or "tau_ps" not in df_sub.columns:
        ax.text(0.5, 0.5, "Required columns ('mag_character' or 'tau_ps') missing.",
                ha="center", va="center", transform=ax.transAxes)
        plt.show()
        return fig, ax

    valid = (
        np.isfinite(df_sub["tau_ps"]) & (df_sub["tau_ps"] > 0) &
        np.isfinite(df_sub["mag_character"]) & (df_sub["mag_character"] > 0)
    )
    invalid = ~valid

    if invalid.any():
        print(f"Note: {invalid.sum():,} modes omitted from Mag vs Tau plot due to non-finite values.")

    if valid.any():
        x_data = df_sub.loc[valid, "mag_character"]
        y_data = df_sub.loc[valid, "tau_ps"]

        sc = ax.scatter(
            x_data,
            y_data,
            s=15,
            linewidths=0,
            marker="D",
            alpha=0.8, 
            color='indianred'
        )

        # --- Add log-log linear trendline ---
        log_x = np.log10(x_data)
        log_y = np.log10(y_data)
        
        # Fit polynomial of degree 1 (log(y) = m * log(x) + c)
        m, c = np.polyfit(log_x, log_y, 1)
        
        # Generate points for the trendline spanning the x-axis limits
        x_fit = np.logspace(np.log10(x_data.min()), np.log10(x_data.max()), 100)
        
        # Enforce exactly -1 slope and add the visual offset you specified
        y_fit = 0.3 * (10**c) * (x_fit**-1)
        
        ax.plot(
            x_fit, y_fit, 
            color="maroon", 
            linewidth=1.5, 
            zorder=3,
            label=r"$\propto (w^{\text{mag}})^{-1}$"
        )
        ax.legend(loc="best", frameon=False, fontsize=9)
        # ------------------------------------

    ax.set_xlabel(r"Magnon character $w^{\text{mag}}_{\boldsymbol{k}\mu}$", fontsize=11)
    ax.set_ylabel(r"Lifetime $\tau_{\boldsymbol{k}\mu}$ (ps)", fontsize=11)
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.tight_layout()
    
    os.makedirs("Outputs", exist_ok=True)
    if out_png:
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        print(f"Plot saved to '{out_png}'")
    
    # Save before plt.show() to prevent blank PDFs
    plt.savefig("Outputs/mag_lifetime.pdf", dpi=400)
    plt.show()
    return fig, ax


def plot_mag_vs_gamma(df, E_min=22.0, out_png=None):
    """Plots Magnon character vs scattering rate Gamma (1/tau)."""
    mask_energy = df["energy_meV"] > E_min if E_min > 0 else np.ones(len(df), dtype=bool)
    df_sub = df[mask_energy]

    fig, ax = plt.subplots(figsize=(6/2.52, 14/5.52))

    if df_sub.empty:
        ax.text(0.5, 0.5, f"No modes found with E > {E_min} meV.",
                ha="center", va="center", transform=ax.transAxes)
        plt.show()
        return fig, ax

    valid = (
        np.isfinite(df_sub["tau_ps"]) & (df_sub["tau_ps"] > 0) &
        np.isfinite(df_sub["mag_character"]) & (df_sub["mag_character"] > 0)
    )

    if valid.any():
        x_data = df_sub.loc[valid, "mag_character"]
        # Gamma in ps^-1
        y_data = 1.0 / df_sub.loc[valid, "tau_ps"]

        sc = ax.scatter(
            x_data,
            y_data,
            s=15,
            linewidths=0,
            marker="D",
            alpha=0.8,
            color='indianred' 
        )

        # --- Add log-log linear trendline ---
        log_x = np.log10(x_data)
        log_y = np.log10(y_data)
        
        m, c = np.polyfit(log_x, log_y, 1)
        
        x_fit = np.logspace(np.log10(x_data.min()), np.log10(x_data.max()), 100)
        
        # Enforce exactly +1 slope, offset slightly above the data
        y_fit = 3.0 * (10**c) * (x_fit**1)
        
        ax.plot(
            x_fit, y_fit, 
            color="maroon", 
            linewidth=1.5, 
            zorder=3,
            label=r"$\propto w^{\text{mag}}$"
        )
        ax.legend(loc="best", frameon=False, fontsize=9)
        # ------------------------------------

    ax.set_xlabel(r"Magnon character $w^{\text{mag}}_{\boldsymbol{k}\mu}$", fontsize=11)
    ax.set_ylabel(r"Scattering rate $\gamma_{\boldsymbol{k}\mu}$ (ps$^{-1}$)", fontsize=11)
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.tight_layout()

    os.makedirs("Outputs", exist_ok=True)
    if out_png:
        # If an explicit arg was passed for out_png, we'll append a suffix for the gamma plot
        out_png_gamma = out_png.replace(".png", "_gamma.png")
        fig.savefig(out_png_gamma, dpi=300, bbox_inches="tight")
        print(f"Plot saved to '{out_png_gamma}'")
        
    plt.savefig("Outputs/mag_gamma.pdf", dpi=400)
    plt.show()
    return fig, ax


def main():
    p = argparse.ArgumentParser(description="Plot Magnon character vs Lifetime and Scattering Rate.")
    p.add_argument("--slc", required=True, help="hybrid_path_lifetimes.csv from the SLC-enabled run")
    p.add_argument("--out", default=None, help="Output PNG file path")
    p.add_argument("--E_min", type=float, default=22.0, help="Energy threshold for modes (meV)")
    args = p.parse_args()

    df = load_path_csv(args.slc)
    
    # Generate both plots
    plot_mag_vs_tau_pam(df, E_min=args.E_min, out_png=args.out)
    plot_mag_vs_gamma(df, E_min=args.E_min, out_png=args.out)


if __name__ == "__main__":
    main()