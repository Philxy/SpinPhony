import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
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


def load_path_csv(path):
    """
    Loads a SpinPhony.py path-output CSV (hybrid_path_properties.csv).
    Extracts high-symmetry labels from comments if present.
    """
    with open(path) as f:
        first_line = f.readline().strip()

    labels = []
    if first_line.startswith("# path_labels:"):
        payload = first_line[len("# path_labels:"):].strip()
        if payload and payload != "unavailable":
            for pair in payload.split(","):
                name, val = pair.split("=")
                labels.append((name, float(val)))
        df = pd.read_csv(path, skiprows=1)
    else:
        df = pd.read_csv(path)

    # Clean whitespace from column names
    df.columns = df.columns.str.strip()
    return df, labels


def set_path_ticks(ax, labels):
    """Applies high-symmetry point tick labels + vertical gridlines to ax."""
    if not labels:
        return
    tick_locs = [pos for _, pos in labels]
    tick_labels = [name for name, _ in labels]
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labels, fontsize=13)
    ax.grid(True, axis="x", linestyle="--", color="gray", alpha=0.5)


def _detect_column(df, candidate_names):
    """Utility to find matching column names regardless of exact casing or formatting."""
    df_cols_lower = {col.lower(): col for col in df.columns}
    # Pass 1: Exact match priority
    for cand in candidate_names:
        cand_lower = cand.lower()
        if cand_lower in df_cols_lower:
            return df_cols_lower[cand_lower]
    # Pass 2: Partial match fallback
    for cand in candidate_names:
        cand_lower = cand.lower()
        for col_lower, orig_col in df_cols_lower.items():
            if cand_lower in col_lower:
                return orig_col
    return None


def plot_dense_property(
    dense_csv,
    prop_column_candidates,
    cbar_label,
    cmap="RdBu_r",
    is_diverging=True,
    use_log=False,
    linthresh=1e-2,
    vlim=None,
    linewidth=2.0,
    margin=0.0,
    save_plot=None,
):
    """
    Plots the dense band structure colored by a specific property using LineCollection.
    Supports linear (with exact min/max bounds) and logarithmic colormap scaling.
    """
    df_disp, labels = load_path_csv(dense_csv)

    # Resolve actual column name from candidates
    target_col = _detect_column(df_disp, prop_column_candidates)
    if target_col is None:
        raise KeyError(
            f"Could not find any of {prop_column_candidates} in '{dense_csv}'. "
            f"Available columns: {list(df_disp.columns)}"
        )

    vals = df_disp[target_col].dropna().to_numpy()

    # Determine vmin and vmax bounds
    if vlim is not None:
        if isinstance(vlim, (tuple, list)):
            vmin, vmax = vlim
        else:
            vmin, vmax = -abs(vlim), abs(vlim)
    else:
        if is_diverging:
            max_abs = max(abs(vals.min()), abs(vals.max()))
            vmin, vmax = -max_abs, max_abs
        else:
            vmin, vmax = vals.min(), vals.max()

    # Determine Normalization (Logarithmic vs Linear)
    if use_log:
        if is_diverging:
            # Symmetric Log Scale for diverging properties with zero/negative values (e.g. PAM)
            norm = mcolors.SymLogNorm(
                linthresh=linthresh, vmin=vmin, vmax=vmax, base=10
            )
        else:
            # Standard Log Scale for strictly positive properties
            pos_vals = vals[vals > 0]
            min_pos = pos_vals.min() if len(pos_vals) > 0 else 1e-4
            vmin_log = max(vmin, min_pos) if vmin > 0 else min_pos
            vmax_log = max(vmax, vmin_log * 10)
            norm = mcolors.LogNorm(vmin=vmin_log, vmax=vmax_log)
    else:
        # Linear Normalization
        if is_diverging and vmin < 0.0 < vmax:
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        else:
            if vmin == vmax:
                vmin, vmax = vmin - 0.1, vmax + 0.1
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Setup Plot
    fig, ax = plt.subplots(figsize=(16 / 2.52, 16 / 2.52))

    band_col = _detect_column(df_disp, ["band", "branch"]) or "band"
    path_col = _detect_column(df_disp, ["path_dist", "q_path", "q_idx"]) or "path_dist"
    lc = None

    for band_id in sorted(df_disp[band_col].unique()):
        subset = df_disp[df_disp[band_col] == band_id].sort_values(path_col)
        if len(subset) < 2:
            continue

        x = subset[path_col].to_numpy()
        y = subset["energy_meV"].to_numpy()
        c = subset[target_col].to_numpy()

        # Segment midpoint coloring
        seg_vals = 0.5 * (c[:-1] + c[1:])

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(seg_vals)
        lc.set_linewidth(linewidth)
        ax.add_collection(lc)

    if lc is not None:
        cbar = fig.colorbar(lc, ax=ax)
        cbar.set_label(cbar_label, fontsize=12, fontweight="bold")

    # Set axis limits
    x_min, x_max = df_disp[path_col].min(), df_disp[path_col].max()
    x_dist = x_max - x_min
    ax.set_xlim(x_min - x_dist * margin, x_max + x_dist * margin)

    ax.set_ylim(df_disp["energy_meV"].min(), df_disp["energy_meV"].max() * 1.05)
    ax.set_ylabel("Energy (meV)", fontsize=12, fontweight="bold")
    set_path_ticks(ax, labels)

    fig.tight_layout()
    plt.show()

    if save_plot:
        os.makedirs(os.path.dirname(save_plot), exist_ok=True)
        fig.savefig(save_plot, dpi=300)
        print(f"Plot saved to '{save_plot}'")

    return fig, ax


def plot_all_cri3_properties(
    dense_csv,
    output_dir="Outputs/CrI3_Properties",
):
    cmap_contrast = get_contrast_coolwarm()

    """
    Plots the three key dense properties for CrI3 matching exact header columns:
    - phon_AM_z_hbar
    - mag_character
    - spin_AM_z_hbar
    """

    # 1. Phonon Angular Momentum (Linear scaling with min/max vlim bounds; set use_log=True if logarithmic scaling is desired)
    plot_dense_property(
        dense_csv=dense_csv,
        prop_column_candidates=["phon_AM_z_hbar", "phonon_Lz", "Lz_ph", "L_z"],
        cbar_label=r"Phonon Angular Momentum $L_z$ ($\hbar$)",
        cmap=cmap_contrast,
        is_diverging=True,
        use_log=False,       # Set to True to enable symmetric logarithmic colormap
        vlim=(-1.0, 1.0),    # Explicit min and max for linear/log scaling
        save_plot=os.path.join(output_dir, "cri3_phonon_angular_momentum.png"),
    )

    # 2. Magnon / Phonon Character
    plot_dense_property(
        dense_csv=dense_csv,
        prop_column_candidates=["mag_character", "magnon_character", "character"],
        cbar_label=r"Magnon Character",
        cmap="coolwarm",
        is_diverging=False,
        use_log=False,
        vlim=(0.0, 1.0),
        save_plot=os.path.join(output_dir, "cri3_magnon_character.png"),
    )

    # 3. Spin Angular Momentum
    plot_dense_property(
        dense_csv=dense_csv,
        prop_column_candidates=["spin_AM_z_hbar", "spin_Sz", "Sz_spin", "S_z"],
        cbar_label=r"Spin Angular Momentum $S_z$ ($\hbar$)",
        cmap="copper",
        is_diverging=False,
        use_log=False,
        vlim=(-1.0, 0.0),
        save_plot=os.path.join(output_dir, "cri3_spin_angular_momentum.png"),
    )


if __name__ == "__main__":


    # Hybrid CrI3
    hybrid_dense_csv = "Outputs/Hybrid_GK_dense//hybrid_path_properties.csv"
    if os.path.exists(hybrid_dense_csv):
        print("Plotting Hybrid CrI3 Properties...")
        plot_all_cri3_properties(
            dense_csv=hybrid_dense_csv,
            output_dir="Outputs/Hybrid_GK_32_dense/properties",
        )

    # Hybrid bccFe
    hybrid_dense_csv = "Outputs/bccFePath/bccFe_whole_BZ_20/hybrid_path_properties.csv"

    if os.path.exists(hybrid_dense_csv):
        print("Plotting Hybrid bccFe Properties...")
        plot_all_cri3_properties(
            dense_csv=hybrid_dense_csv,
            output_dir="Outputs/bccFePath/bccFe_whole_BZ_20/properties",
        )




    # Non-Hybrid CrI3
    nonhybrid_dense_csv = "Outputs/NonHybrid_GK_32_dense/hybrid_path_properties.csv"
    if os.path.exists(nonhybrid_dense_csv):
        print("Plotting Non-Hybrid CrI3 Properties...")
        plot_all_cri3_properties(
            dense_csv=nonhybrid_dense_csv,
            output_dir="Outputs/NonHybrid_GK_32_dense/properties",
        )