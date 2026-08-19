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


def set_path_ticks(ax, labels, theme="light"):
    """Applies high-symmetry point tick labels + vertical gridlines to ax."""
    if not labels:
        return
    tick_locs = [pos for _, pos in labels]
    
    # Replace 'G' with LaTeX \Gamma
    tick_labels = [r"$\Gamma$" if name == "G" else name for name, _ in labels]
    
    ax.set_xticks(tick_locs)
    
    # Tick labels are always on the white outer margin, so they stay black
    ax.set_xticklabels(tick_labels, fontsize=13, color="black")
    
    # The grid is inside the axes; use a lighter gray for visibility on black axes
    grid_color = "#AAAAAA" if theme == "dark" else "gray"
    ax.grid(True, axis="x", linestyle="--", color=grid_color, alpha=0.5)


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
    linewidth=1.0,
    margin=0.0,
    save_plot=None,
    theme="light",
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
            norm = mcolors.SymLogNorm(
                linthresh=linthresh, vmin=vmin, vmax=vmax, base=10
            )
        else:
            # --- MODIFIED: Use the user vmin directly if > 0, allowing clip=True in LogNorm ---
            vmin_log = vmin if vmin > 0 else 1e-4
            norm = mcolors.LogNorm(vmin=vmin_log, vmax=vmax, clip=True)
    else:
        if is_diverging and vmin < 0.0 < vmax:
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        else:
            if vmin == vmax:
                vmin, vmax = vmin - 0.1, vmax + 0.1
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Setup Plot
    fig, ax = plt.subplots(figsize=(8 / 2.52, 12 / 2.52))

    # Apply Dark Theme Overrides ONLY to the inside of the plot (the axes)
    if theme == "dark":
        ax.set_facecolor('black')

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
        
        # --- MODIFIED: Clip values so anything below the log limit (like 0) is pinned to the minimum.
        # This guarantees Matplotlib won't "mask" and hide segments with a value of 0. ---
        if use_log and not is_diverging:
            seg_vals = np.clip(seg_vals, norm.vmin, None)

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Added capstyle="round" and joinstyle="round" to remove white gaps
        lc = LineCollection(segments, cmap=cmap, norm=norm, capstyle="round", joinstyle="round")
        lc.set_array(seg_vals)
        lc.set_linewidth(linewidth)
        ax.add_collection(lc)

    if lc is not None:
        cbar = fig.colorbar(lc, ax=ax, location="top", pad=0.01)
        # Label remains black because it's in the white outer margin
        cbar.set_label(cbar_label, fontsize=12, fontweight="bold", labelpad=15, color="black")

    # Set axis limits
    x_min, x_max = df_disp[path_col].min(), df_disp[path_col].max()
    x_dist = x_max - x_min
    ax.set_xlim(x_min - x_dist * margin, x_max + x_dist * margin)

    ax.set_ylim(df_disp["energy_meV"].min(), df_disp["energy_meV"].max() * 1.05)
    ax.set_ylabel(r"Energy $\varepsilon_{\boldsymbol{k}\mu}$ (meV)", fontsize=12, fontweight="bold")
    set_path_ticks(ax, labels, theme=theme)

    fig.tight_layout()
    #plt.show()

    if save_plot:
        os.makedirs(os.path.dirname(save_plot), exist_ok=True)
        # Don't pass a facecolor to savefig, letting the outer figure background stay standard
        fig.savefig(save_plot, dpi=400)
        print(f"Plot saved to '{save_plot}'")

    plt.close(fig) 
    return fig, ax


def plot_all_cri3_properties(
    dense_csv,
    output_dir="Outputs/CrI3_Properties",
):
    cmap_contrast = get_contrast_coolwarm()

    # Define our two plotting configurations
    themes = [
        # Standard Version
        {"name": "light", "cmap_div": cmap_contrast, "cmap_seq": "copper", "cmap_spin": "copper"},
        # Version with Black Axes and Standard RdBu_r colormap
        {"name": "dark", "cmap_div": "bwr", "cmap_seq": "RdBu_r", "cmap_spin": "copper"}
    ]

    for t in themes:
        theme = t["name"]
        suffix = f"_{theme}.png"

        # 1. Phonon Angular Momentum
        plot_dense_property(
            dense_csv=dense_csv,
            prop_column_candidates=["phon_AM_z_hbar", "phonon_Lz", "Lz_ph", "L_z"],
            cbar_label=r"Phonon angular momentum $L^z_{\boldsymbol{k}\mu}$ ($\hbar$)",
            cmap=t["cmap_div"],
            is_diverging=True,
            use_log=True,
            vlim=(-1.0, 1.0),
            save_plot=os.path.join(output_dir, f"cri3_phonon_angular_momentum{suffix}"),
            theme=theme
        )

        # 2. Magnon / Phonon Character
        plot_dense_property(
            dense_csv=dense_csv,
            prop_column_candidates=["mag_character", "magnon_character", "character"],
            cbar_label=r"Magnon character $w_{\boldsymbol{k}\mu}^{\text{mag}}$",
            cmap=t["cmap_seq"],
            is_diverging=False,
            use_log=True,         
            vlim=(1e-4, 1.0),     # Any values < 1e-3 will be clipped to 1e-3 and plotted continuously
            save_plot=os.path.join(output_dir, f"cri3_magnon_character{suffix}"),
            theme=theme
        )

        # 3. Spin Angular Momentum
        plot_dense_property(
            dense_csv=dense_csv,
            prop_column_candidates=["spin_AM_z_hbar", "spin_Sz", "Sz_spin", "S_z"],
            cbar_label=r"Spin Angular Momentum $S_z$ ($\hbar$)",
            cmap=t["cmap_spin"],
            is_diverging=False,
            use_log=False,
            vlim=(-1.0, 0.0),
            save_plot=os.path.join(output_dir, f"cri3_spin_angular_momentum{suffix}"),
            theme=theme
        )


if __name__ == "__main__":

    # Hybrid CrI3
    hybrid_dense_csv = "Outputs/CrI3_Path_Hyrbid_dense/hybrid_path_properties.csv"
    if os.path.exists(hybrid_dense_csv):
        print("Plotting Hybrid CrI3 Properties...")
        plot_all_cri3_properties(
            dense_csv=hybrid_dense_csv,
            output_dir="Outputs/CrI3_Path_Hyrbid_dense/",
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