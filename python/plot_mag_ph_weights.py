"""
Plot Magnon character vs Phonon character from a hybrid_path_lifetimes.csv file,
with points colored by their lifetime.

Usage:
    python plot_characters.py Outputs/CrI3/hybrid_path_lifetimes.csv --out Outputs/CrI3/mag_vs_phon.png
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def load_path_csv(path):
    """Loads a SpinPhony path CSV, skipping the '# path_labels:' comment line."""
    with open(path) as f:
        first = f.readline().strip()
    return pd.read_csv(path, skiprows=1) if first.startswith("# path_labels:") else pd.read_csv(path)


def plot_characters(df, out_png=None, cmap="jet"):
    """Simple scatter plot of Magnon character vs Phonon character, colored by lifetime."""
    need = ["mag_character", "phon_character", "tau_ps"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing column(s): {missing}")

    # Lifetimes can be infinite if no scattering channel is found. 
    # We filter them out so they don't break the colorbar.
    ok = np.isfinite(df["tau_ps"]) & (df["tau_ps"] > 0)
    n_drop = int((~ok).sum())
    plot_df = df[ok]
    
    if plot_df.empty:
        raise ValueError("Nothing to plot - all lifetimes are non-finite or missing.")
    if n_drop:
        print(f"Note: dropped {n_drop:,} modes with non-finite or non-positive lifetimes.")

    fig, ax = plt.subplots(figsize=(6, 5))

    # Scatter plot color-coded by Lifetime.
    # Using LogNorm because lifetimes usually span multiple orders of magnitude.
    # (Change norm=None if you prefer a linear color scale).
    sc = ax.scatter(plot_df["mag_character"], plot_df["phon_character"],
                    c=plot_df["tau_ps"], cmap=cmap, s=15, linewidths=0,
                    norm=mcolors.LogNorm())

    ax.set_xlabel("Magnon character", fontsize=11)
    ax.set_ylabel("Phonon character", fontsize=11)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"Lifetime $\tau$ (ps)", fontsize=10)

    if out_png:
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        print(f"Plot saved to '{out_png}'")

    plt.show()
    return fig, ax


def main():
    p = argparse.ArgumentParser(description="Plot Magnon vs Phonon character colored by lifetime.")
    p.add_argument("csv_file", help="Path to hybrid_path_lifetimes.csv")
    p.add_argument("--out", default=None, help="Optional path to save the output PNG")
    args = p.parse_args()

    df = load_path_csv(args.csv_file)
    print(f"Loaded {len(df):,} modes from {args.csv_file}")
    
    plot_characters(df, out_png=args.out)


if __name__ == "__main__":
    main()