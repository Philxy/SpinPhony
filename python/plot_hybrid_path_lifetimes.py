import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from scipy.interpolate import interp1d
import scienceplots

plt.style.use("science")


def load_path_csv(path):
    """
    Loads a SpinPhony.py path-output CSV (hybrid_path_properties.csv,
    hybrid_path_lifetimes.csv, path_lifetimes.csv, ...). These all start
    with a "# path_labels: G=0.000000,K=1.234567,..." comment line followed
    by the normal header, and every row carries a 'path_dist' column (1/A,
    cumulative distance along the high-symmetry path).

    Returns (df, labels) where labels is an ORDERED list of (name, path_dist)
    pairs - NOT a dict, since a closed path revisits the same label (e.g.
    Gamma at both the start and end), which a dict would silently collapse -
    or an empty list if the source run had no path_labels available.
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
        # Older file without the label/comment line - read as-is.
        df = pd.read_csv(path)

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


def _prepare_tau(df, tau_col="tau_ps"):
    """Replaces inf with the max finite value and clips to strictly positive,
    matching the convention needed for a log color scale."""
    df = df.copy()
    finite = df.loc[np.isfinite(df[tau_col]), tau_col]
    if finite.empty:
        raise ValueError(f"No finite values found in column '{tau_col}'.")
    max_finite = finite.max()
    df[tau_col] = df[tau_col].replace([np.inf, -np.inf], max_finite)
    min_pos = df.loc[df[tau_col] > 0, tau_col].min()
    df[tau_col] = df[tau_col].clip(lower=min_pos)
    return df


def plot_dense_with_scatter(
    dense_csv="Outputs/HybridDense/hybrid_path_properties.csv",
    lifetime_csv="Outputs/Hybrid_band32/hybrid_path_lifetimes.csv",
    tau_col="tau_ps",
    cmap="rainbow_r",
    vmin=None,
    vmax=None,
    margin=0.02,  # Relative margin added to left and right (0.02 = 2%)
    save_plot="Outputs/Hybrid/hybrid_bands_lifetime_scatter.png",
):
    """
    Figure 1: dense hybrid band structure with lifetime scatter points overlaid.
    """
    df_disp, labels = load_path_csv(dense_csv)
    df_life, _ = load_path_csv(lifetime_csv)
    df_life = _prepare_tau(df_life, tau_col)

    if vmin is None:
        vmin = df_life[tau_col].min()

    if vmax is None:
        vmax = df_life[tau_col].max()

    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(16/2.52, 16/2.52))  

    for band in sorted(df_disp["band"].unique()):
        subset = df_disp[df_disp["band"] == band].sort_values("path_dist")
        ax.plot(subset["path_dist"], subset["energy_meV"], color="lightgray", lw=1.0, zorder=1, alpha=1)

    sc = ax.scatter(
        df_life["path_dist"], df_life["energy_meV"],
        c=df_life[tau_col], cmap=cmap, norm=norm,
        s=10, zorder=2, linewidth=0.5,
    )

    cbar = fig.colorbar(sc, ax=ax, location="top")
    cbar.set_label(r"Lifetime $\tau$ (ps)", fontsize=12, fontweight="bold")

    # Expand x-limits dynamically based on total path distance
    x_min = df_disp["path_dist"].min()
    x_max = df_disp["path_dist"].max()
    x_dist = x_max - x_min
    ax.set_xlim(x_min - x_dist * margin, x_max + x_dist * margin)

    ax.set_ylim(bottom=0)
    ax.set_ylabel("Energy (meV)", fontsize=12, fontweight="bold")
    set_path_ticks(ax, labels)

    fig.tight_layout()
    plt.show()

    if save_plot:
        os.makedirs(os.path.dirname(save_plot), exist_ok=True)
        fig.savefig(save_plot, dpi=300)
        print(f"Plot saved to '{save_plot}'")

    return fig, ax


def plot_dense_interpolated_line(
    dense_csv="Outputs/HybridDense/hybrid_path_properties.csv",
    lifetime_csv="Outputs/Hybrid_band32/hybrid_path_lifetimes.csv",
    tau_col="tau_ps",
    cmap="rainbow_r",
    vmin=1.0,
    vmax=1e5,
    linewidth=2.5,
    save_plot="Outputs/Hybrid/hybrid_bands_lifetime_interpolated.png",
):
    """
    Figure 2: dense hybrid band structure, drawn as a continuous line per
    band, colored by the sparser lifetime data log-linearly interpolated
    onto the dense path's path_dist grid - using each file's own stored
    path_dist directly, no remapping. Matching between the dense 'band'
    column and the sparse 'branch' column is by shared branch index.
    """
    df_disp, labels = load_path_csv(dense_csv)
    df_life, _ = load_path_csv(lifetime_csv)
    df_life = _prepare_tau(df_life, tau_col)


    if vmin is None:
        vmin =  df_life[tau_col].min()

    if vmax is None:
        vmax =  df_life[tau_col].max()

    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(16/2.52, 16/2.52))  


    lc = None
    for branch in sorted(df_life["branch"].unique()):
        subset_life = df_life[df_life["branch"] == branch].sort_values("path_dist")
        if len(subset_life) < 2:
            continue

        subset_disp = df_disp[df_disp["band"] == branch].sort_values("path_dist")
        if subset_disp.empty:
            continue

        # Linear interpolation in log10(tau) space, so the color varies
        # smoothly across orders of magnitude rather than linearly in tau.
        log_tau_sparse = np.log10(subset_life[tau_col].to_numpy())
        f_interp = interp1d(
            subset_life["path_dist"], log_tau_sparse,
            kind="linear", bounds_error=False,
            fill_value=(log_tau_sparse[0], log_tau_sparse[-1]),
        )
        tau_dense = 10.0 ** f_interp(subset_disp["path_dist"])
        tau_dense = np.clip(tau_dense, norm.vmin, norm.vmax)

        x = subset_disp["path_dist"].to_numpy()
        y = subset_disp["energy_meV"].to_numpy()
        seg_vals = np.sqrt(tau_dense[:-1] * tau_dense[1:])  # log-space (geometric) segment color

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(seg_vals)
        lc.set_linewidth(linewidth)
        ax.add_collection(lc)

    if lc is not None:
        cbar = fig.colorbar(lc, ax=ax, location="top")
        cbar.set_label(r"Interpolated lifetime $\tau$ (ps)", fontsize=12, fontweight="bold")

    ax.set_xlim(df_disp["path_dist"].min(), df_disp["path_dist"].max())
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


if __name__ == "__main__":


    # CrI3 
    
    # hybrid
    df_dense, labels_dense = load_path_csv("Outputs/CrI3_Path_Hyrbid_dense/hybrid_path_properties.csv")
    df_sparse, labels_sparse = load_path_csv("Outputs/CrI3_Path_Hyrbid_Anal/hybrid_path_lifetimes.csv")

    print("Dense labels: ", labels_dense)
    print("Sparse labels:", labels_sparse)

    plot_dense_with_scatter(
        dense_csv="Outputs/CrI3_Path_Hyrbid_dense/hybrid_path_properties.csv",
        lifetime_csv="Outputs/CrI3_Path_Hyrbid_Anal/hybrid_path_lifetimes.csv",
        save_plot="Outputs/CrI3_Path_Hyrbid_Anal/hybrid_bands_lifetime_scatter.png",
        vmin=1E0,
        vmax=1E7
    )
    plot_dense_interpolated_line(
        dense_csv="Outputs/CrI3_Path_Hyrbid_dense/hybrid_path_properties.csv",
        lifetime_csv="Outputs/CrI3_Path_Hyrbid_Anal/hybrid_path_lifetimes.csv",
        save_plot="Outputs/CrI3_Path_Hyrbid_Anal/hybrid_bands_lifetime_interpolated.png",
        vmin=1E0,
        vmax=1E7
    )


    # CrI3 
        
    # non hybrid
    df_dense, labels_dense = load_path_csv("Outputs/CrI3_Path_NonHyrbid_dense/hybrid_path_properties.csv")
    df_sparse, labels_sparse = load_path_csv("Outputs/CrI3_Path_NonHyrbid_Anal/hybrid_path_lifetimes.csv")

    print("Dense labels: ", labels_dense)
    print("Sparse labels:", labels_sparse)

    plot_dense_with_scatter(
        dense_csv="Outputs/CrI3_Path_NonHyrbid_dense/hybrid_path_properties.csv",
        lifetime_csv="Outputs/CrI3_Path_NonHyrbid_Anal/hybrid_path_lifetimes.csv",
        save_plot="Outputs/CrI3_Path_NonHyrbid_Anal/hybrid_bands_lifetime_scatter.png",
        vmin=1E0,
        vmax=1E7
    )
    plot_dense_interpolated_line(
        dense_csv="Outputs/CrI3_Path_NonHyrbid_dense/hybrid_path_properties.csv",
        lifetime_csv="Outputs/CrI3_Path_NonHyrbid_Anal/hybrid_path_lifetimes.csv",
        save_plot="Outputs/CrI3_Path_NonHyrbid_Anal/hybrid_bands_lifetime_interpolated.png",
        vmin=1E0,
        vmax=1E7
    )


    #  hybrid
    df_dense, labels_dense = load_path_csv("Outputs/CrI3_Path_Hybrid/hybrid_path_properties.csv")
    df_sparse, labels_sparse = load_path_csv("Outputs/CrI3_Path_Hybrid/hybrid_path_lifetimes.csv")

    print("Dense labels: ", labels_dense)
    print("Sparse labels:", labels_sparse)

    plot_dense_with_scatter(
        dense_csv="Outputs/CrI3_Path_Hybrid/hybrid_path_properties.csv",
        lifetime_csv="Outputs/CrI3_Path_Hybrid/hybrid_path_lifetimes.csv",
        save_plot="Outputs/CrI3_Path_Hybrid/hybrid_bands_lifetime_scatter.png",
    )
    plot_dense_interpolated_line(
        dense_csv="Outputs/CrI3_Path_Hybrid/hybrid_path_properties.csv",
        lifetime_csv="Outputs/CrI3_Path_Hybrid/hybrid_path_lifetimes.csv",
        save_plot="Outputs/CrI3_Path_Hybrid/hybrid_bands_lifetime_interpolated.png",
    )


    # CrI3 

    # non hybrid
    df_dense, labels_dense = load_path_csv("Outputs/NonHybrid_GK_32_dense/hybrid_path_properties.csv")
    df_sparse, labels_sparse = load_path_csv("Outputs/NonHybrid_GK_32_sigma_0.15/hybrid_path_lifetimes.csv")

    print("Dense labels: ", labels_dense)
    print("Sparse labels:", labels_sparse)

    plot_dense_with_scatter(
        dense_csv="Outputs/NonHybrid_GK_32_dense/hybrid_path_properties.csv",
        lifetime_csv="Outputs/NonHybrid_GK_32_sigma_0.15/hybrid_path_lifetimes.csv",
        save_plot="Outputs/NonHybrid_GK_32_sigma_0.15/hybrid_bands_lifetime_scatter.png",
        vmin=1E0,
        vmax=1E5
    )
    plot_dense_interpolated_line(
        dense_csv="Outputs/NonHybrid_GK_32_dense/hybrid_path_properties.csv",
        lifetime_csv="Outputs/NonHybrid_GK_32_sigma_0.15/hybrid_path_lifetimes.csv",
        save_plot="Outputs/NonHybrid_GK_32_sigma_0.15/hybrid_bands_lifetime_interpolated.png",
        vmin=1E0,
        vmax=1E5
    )

    # hybrid
    df_dense, labels_dense = load_path_csv("Outputs/Hybrid_GK_32_dense/hybrid_path_properties.csv")
    df_sparse, labels_sparse = load_path_csv("Outputs/Hybrid_GK_32_sigma_0.15/hybrid_path_lifetimes.csv")

    print("Dense labels: ", labels_dense)
    print("Sparse labels:", labels_sparse)

    plot_dense_with_scatter(
        dense_csv="Outputs/Hybrid_GK_32_dense/hybrid_path_properties.csv",
        lifetime_csv="Outputs/Hybrid_GK_32_sigma_0.15/hybrid_path_lifetimes.csv",
        save_plot="Outputs/Hybrid_GK_32_sigma_0.15/hybrid_bands_lifetime_scatter.png",
        vmin=1E0,
        vmax=1E5
    )
    plot_dense_interpolated_line(
        dense_csv="Outputs/Hybrid_GK_32_dense/hybrid_path_properties.csv",
        lifetime_csv="Outputs/Hybrid_GK_32_sigma_0.15/hybrid_path_lifetimes.csv",
        save_plot="Outputs/Hybrid_GK_32_sigma_0.15/hybrid_bands_lifetime_interpolated.png",
        vmin=1E0,
        vmax=1E5
    )




    # bcc Fe

    df_dense, labels_dense = load_path_csv("Outputs/bccFeHybrid/hybrid_path_properties.csv")
    df_sparse, labels_sparse = load_path_csv("Outputs/bccFeHybrid/hybrid_path_lifetimes.csv")

    print("Dense labels: ", labels_dense)
    print("Sparse labels:", labels_sparse)

    plot_dense_with_scatter(
        dense_csv="Outputs/bccFeHybrid/hybrid_path_properties.csv",
        lifetime_csv="Outputs/bccFeHybrid/hybrid_path_lifetimes.csv",
        save_plot="Outputs/bccFeHybrid/hybrid_bands_lifetime_scatter.png",
    )
    plot_dense_interpolated_line(
        dense_csv="Outputs/bccFeHybrid/hybrid_path_properties.csv",
        lifetime_csv="Outputs/bccFeHybrid/hybrid_path_lifetimes.csv",
        save_plot="Outputs/bccFeHybrid/hybrid_bands_lifetime_interpolated.png",
    )