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
        payload = first_line[len("# path_labels:") :].strip()
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
    """Applies high-symmetry point tick labels + vertical gridlines to ax,
    replacing 'G' and 'Gamma' with the LaTeX symbol \Gamma."""
    if not labels:
        return
    tick_locs = [pos for _, pos in labels]
    tick_labels = [
        name.replace("GAMMA", r"$\Gamma$")
        .replace("Gamma", r"$\Gamma$")
        .replace("G", r"$\Gamma$")
        for name, _ in labels
    ]
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


def _get_log_ticks(vmin, vmax):
    """Generates decade tick positions in [0, 1] normalized space and formatted strings."""
    log_min = np.floor(np.log10(vmin))
    log_max = np.ceil(np.log10(vmax))
    decades = np.arange(log_min, log_max + 1)

    tick_vals = 10.0**decades
    positions = (np.log10(tick_vals) - np.log10(vmin)) / (
        np.log10(vmax) - np.log10(vmin)
    )

    valid = (positions >= -1e-6) & (positions <= 1.0 + 1e-6)
    positions = positions[valid]
    decades = decades[valid]

    labels = [f"$10^{{{int(d)}}}$" if d != 0 else "$1$" for d in decades]
    return positions, labels


def plot_dense_with_scatter(
    dense_csv="Outputs/HybridDense/hybrid_path_properties.csv",
    lifetime_csv="Outputs/Hybrid_band32/hybrid_path_lifetimes.csv",
    tau_col="tau_ps",
    cmap="rainbow_r",
    vmin=None,
    vmax=None,
    margin=0.02,
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

    fig, ax = plt.subplots(figsize=(16 / 2.52, 16 / 2.52))

    for band in sorted(df_disp["band"].unique()):
        subset = df_disp[df_disp["band"] == band].sort_values("path_dist")
        ax.plot(
            subset["path_dist"],
            subset["energy_meV"],
            color="lightgray",
            lw=1.0,
            zorder=1,
            alpha=1,
        )

    sc = ax.scatter(
        df_life["path_dist"],
        df_life["energy_meV"],
        c=df_life[tau_col],
        cmap=cmap,
        norm=norm,
        s=10,
        zorder=2,
        linewidth=0.5,
    )

    cbar = fig.colorbar(sc, ax=ax, location="top", pad=0.05)
    cbar.set_label(r"Lifetime $\tau_{\boldsymbol{k}\mu}$ (ps)", fontsize=12, fontweight="bold")

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
    band, colored by the sparser lifetime data log-linearly interpolated.
    """
    df_disp, labels = load_path_csv(dense_csv)
    df_life, _ = load_path_csv(lifetime_csv)
    df_life = _prepare_tau(df_life, tau_col)

    if vmin is None:
        vmin = df_life[tau_col].min()

    if vmax is None:
        vmax = df_life[tau_col].max()

    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(16 / 2.52, 16 / 2.52))

    lc = None
    for branch in sorted(df_life["branch"].unique()):
        subset_life = df_life[df_life["branch"] == branch].sort_values("path_dist")
        if len(subset_life) < 2:
            continue

        subset_disp = df_disp[df_disp["band"] == branch].sort_values("path_dist")
        if subset_disp.empty:
            continue

        log_tau_sparse = np.log10(subset_life[tau_col].to_numpy())
        f_interp = interp1d(
            subset_life["path_dist"],
            log_tau_sparse,
            kind="linear",
            bounds_error=False,
            fill_value=(log_tau_sparse[0], log_tau_sparse[-1]),
        )
        tau_dense = 10.0 ** f_interp(subset_disp["path_dist"])
        tau_dense = np.clip(tau_dense, norm.vmin, norm.vmax)

        x = subset_disp["path_dist"].to_numpy()
        y = subset_disp["energy_meV"].to_numpy()
        seg_vals = np.sqrt(tau_dense[:-1] * tau_dense[1:])

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(seg_vals)
        lc.set_linewidth(linewidth)
        ax.add_collection(lc)

    if lc is not None:
        cbar = fig.colorbar(lc, ax=ax, location="top", pad=0.05)
        cbar.set_label(
            r"Interpolated lifetime $\tau$ (ps)", fontsize=12, fontweight="bold"
        )

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


def plot_dense_mag_character(
    dense_csv="Outputs/HybridDense/hybrid_path_properties.csv",
    mag_col="mag_character",
    cmap="rainbow",
    vmin=1e-4,
    vmax=1.0,
    linewidth=1.5,
    save_plot="Outputs/Hybrid/hybrid_bands_mag_character.png",
):
    """
    Figure 3: dense hybrid band structure drawn directly from the dense dataset,
    colored by magnon character (mag_character) on a logarithmic color scale.
    """
    df_disp, labels = load_path_csv(dense_csv)

    if mag_col not in df_disp.columns:
        raise ValueError(f"Column '{mag_col}' not found in '{dense_csv}'.")

    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(16 / 2.52, 16 / 2.52))

    lc = None
    for band in sorted(df_disp["band"].unique()):
        subset = df_disp[df_disp["band"] == band].sort_values("path_dist")
        if len(subset) < 2:
            continue

        x = subset["path_dist"].to_numpy()
        y = subset["energy_meV"].to_numpy()

        mag_dense = np.clip(subset[mag_col].to_numpy(), vmin, vmax)
        seg_vals = np.sqrt(mag_dense[:-1] * mag_dense[1:])

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(seg_vals)
        lc.set_linewidth(linewidth)
        ax.add_collection(lc)

    if lc is not None:
        cbar = fig.colorbar(lc, ax=ax, location="top", pad=0.05)
        cbar.set_label(
            r"Magnon character ($w_{\rm mag}$)", fontsize=12, fontweight="bold"
        )

    ax.set_xlim(df_disp["path_dist"].min(), df_disp["path_dist"].max())
    ax.set_ylim(bottom=0, top=df_disp["energy_meV"].max() * 1.05)
    ax.set_ylabel("Energy (meV)", fontsize=12, fontweight="bold")
    set_path_ticks(ax, labels)

    fig.tight_layout()
    plt.show()

    if save_plot:
        os.makedirs(os.path.dirname(save_plot), exist_ok=True)
        fig.savefig(save_plot, dpi=300)
        print(f"Magnon character plot saved to '{save_plot}'")

    return fig, ax


def plot_dense_mag_with_inv_tau_scatter(
    dense_csv="Outputs/HybridDense/hybrid_path_properties.csv",
    lifetime_csv="Outputs/Hybrid_band32/hybrid_path_lifetimes.csv",
    mag_col="mag_character",
    tau_col="tau_ps",
    cmap="rainbow",
    vmin_mag=1e-4,
    vmax_mag=1.0,
    vmin_inv_tau=1e-5,
    vmax_inv_tau=None,
    linewidth=1.2,
    s=30,
    margin=0.02,
    save_plot="Outputs/Hybrid/hybrid_bands_mag_and_inv_tau.png",
):
    """
    Figure 4: Dense band structure colored by magnon character with scattered
    inverse lifetimes (1/tau) overlaid using a clean dual-labeled colorbar.
    """
    df_disp, labels = load_path_csv(dense_csv)
    df_life, _ = load_path_csv(lifetime_csv)

    if mag_col not in df_disp.columns:
        raise ValueError(f"Column '{mag_col}' not found in '{dense_csv}'.")

    # Filter and calculate 1/tau for finite lifetimes
    valid_tau = np.isfinite(df_life[tau_col]) & (df_life[tau_col] > 0)
    df_life_valid = df_life[valid_tau].copy()
    df_life_valid["inv_tau"] = 1.0 / df_life_valid[tau_col]

    if vmax_inv_tau is None:
        vmax_inv_tau = df_life_valid["inv_tau"].max()

    # Shared normalized coordinate space [0, 1]
    norm_shared = mcolors.Normalize(vmin=0.0, vmax=1.0)

    fig, ax = plt.subplots(figsize=(16 / 2.52, 16 / 2.52))

    # 1. Plot Dense Bands colored by Normalized Magnon Character
    for band in sorted(df_disp["band"].unique()):
        subset = df_disp[df_disp["band"] == band].sort_values("path_dist")
        if len(subset) < 2:
            continue

        x = subset["path_dist"].to_numpy()
        y = subset["energy_meV"].to_numpy()

        mag_vals = np.clip(subset[mag_col].to_numpy(), vmin_mag, vmax_mag)
        norm_mag = (np.log10(mag_vals) - np.log10(vmin_mag)) / (
            np.log10(vmax_mag) - np.log10(vmin_mag)
        )
        seg_vals = 0.5 * (norm_mag[:-1] + norm_mag[1:])

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=cmap, norm=norm_shared, zorder=1)
        lc.set_array(seg_vals)
        lc.set_linewidth(linewidth)
        ax.add_collection(lc)

    # 2. Scatter Points colored by Normalized 1/tau
    if not df_life_valid.empty:
        inv_vals = np.clip(
            df_life_valid["inv_tau"].to_numpy(), vmin_inv_tau, vmax_inv_tau
        )
        norm_inv = (np.log10(inv_vals) - np.log10(vmin_inv_tau)) / (
            np.log10(vmax_inv_tau) - np.log10(vmin_inv_tau)
        )

        ax.scatter(
            df_life_valid["path_dist"],
            df_life_valid["energy_meV"],
            c=norm_inv,
            cmap=cmap,
            norm=norm_shared,
            s=s,
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )

    # 3. Construct Dual-Labeled Colorbar
    sm = plt.cm.ScalarMappable(norm=norm_shared, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, location="top", pad=0.14)
    cbar.ax.minorticks_off()

    # Configure top side (Magnon character) on the primary colorbar axis
    pos_mag, labels_mag = _get_log_ticks(vmin_mag, vmax_mag)
    cbar.set_ticks(pos_mag)
    cbar.ax.set_xticklabels(labels_mag, fontsize=10)
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")
    # Force top only: disable bottom ticks globally in case scienceplots enables them
    cbar.ax.tick_params(
        axis="x", which="both", top=True, bottom=False, labeltop=True, labelbottom=False
    )
    cbar.set_label(
        r"Magnon character ($w^{\rm mag}_{\boldsymbol{k}\mu}$)", fontsize=11, fontweight="bold", labelpad=8
    )

    # Configure bottom side (Scattering rate) via a secondary x-axis
    cax_twin = cbar.ax.secondary_xaxis("bottom")
    cax_twin.minorticks_off()
    pos_inv, labels_inv = _get_log_ticks(vmin_inv_tau, vmax_inv_tau)
    cax_twin.set_xticks(pos_inv)
    cax_twin.set_xticklabels(labels_inv, fontsize=10)
    # Force bottom only: disable top ticks
    cax_twin.tick_params(
        axis="x", which="both", top=False, bottom=True, labeltop=False, labelbottom=True
    )
    cax_twin.set_xlabel(
        r"Scattering rate $1/\tau_{\boldsymbol{k}\mu}$ (ps$^{-1}$)", fontsize=11, fontweight="bold", labelpad=8
    )

    # Axis Limits and Formatting
    x_min = df_disp["path_dist"].min()
    x_max = df_disp["path_dist"].max()
    x_dist = x_max - x_min
    ax.set_xlim(x_min - x_dist * margin, x_max + x_dist * margin)
    ax.set_ylim(bottom=0, top=df_disp["energy_meV"].max() * 1.05)
    ax.set_ylabel("Energy (meV)", fontsize=12, fontweight="bold")
    set_path_ticks(ax, labels)

    fig.tight_layout()
    plt.show()

    if save_plot:
        os.makedirs(os.path.dirname(save_plot), exist_ok=True)
        fig.savefig(save_plot, dpi=300)
        print(f"Dual mag & inverse lifetime plot saved to '{save_plot}'")

    return fig, ax


if __name__ == "__main__":
    # --- CrI3 (Hybrid) ---
    plot_dense_mag_with_inv_tau_scatter(
        dense_csv="Outputs/CrI3_Path_Hyrbid_dense/hybrid_path_properties.csv",
        lifetime_csv="Outputs/CrI3_Path_Hyrbid_Anal_sig_0.5/hybrid_path_lifetimes.csv",
        save_plot="Outputs/CrI3_Path_Hyrbid_Anal_sig_0.5/hybrid_bands_mag_inv_tau.png",
        vmin_mag=1e-4,
        vmax_mag=1.0,
        vmin_inv_tau=1e-5,
        vmax_inv_tau=None,
    )
    plot_dense_mag_character(
        dense_csv="Outputs/CrI3_Path_Hyrbid_dense/hybrid_path_properties.csv",
        save_plot="Outputs/CrI3_Path_Hyrbid_Anal_sig_0.5/hybrid_bands_mag_character.png",
        vmin=1e-4,
        vmax=1.0,
    )
    plot_dense_with_scatter(
        dense_csv="Outputs/CrI3_Path_Hyrbid_dense/hybrid_path_properties.csv",
        lifetime_csv="Outputs/CrI3_Path_Hyrbid_Anal_sig_0.5/hybrid_path_lifetimes.csv",
        save_plot="Outputs/CrI3_Path_Hyrbid_Anal_sig_0.5/hybrid_bands_lifetime_scatter.png",
        vmin=1e0,
        vmax=1e5,
    )
    plot_dense_interpolated_line(
        dense_csv="Outputs/CrI3_Path_Hyrbid_dense/hybrid_path_properties.csv",
        lifetime_csv="Outputs/CrI3_Path_Hyrbid_Anal_sig_0.5/hybrid_path_lifetimes.csv",
        save_plot="Outputs/CrI3_Path_Hyrbid_Anal_sig_0.5/hybrid_bands_lifetime_interpolated.png",
        vmin=1e0,
        vmax=1e5,
    )

    # Non Hybrid
    plot_dense_mag_with_inv_tau_scatter(
        dense_csv="Outputs/CrI3_Path_NonHyrbid_dense/hybrid_path_properties.csv",
        lifetime_csv="Outputs/CrI3_Path_NonHyrbid_Anal_sig_0.5/hybrid_path_lifetimes.csv",
        save_plot="Outputs/CrI3_Path_NonHyrbid_Anal_sig_0.5/hybrid_bands_mag_inv_tau.png",
        vmin_mag=1e-4,
        vmax_mag=1.0,
        vmin_inv_tau=1e-5,
        vmax_inv_tau=None,
    )
    plot_dense_mag_character(
        dense_csv="Outputs/CrI3_Path_NonHyrbid_dense/hybrid_path_properties.csv",
        save_plot="Outputs/CrI3_Path_NonHyrbid_Anal_sig_0.5/hybrid_bands_mag_character.png",
        vmin=1e-4,
        vmax=1.0,
    )
    plot_dense_with_scatter(
        dense_csv="Outputs/CrI3_Path_NonHyrbid_dense/hybrid_path_properties.csv",
        lifetime_csv="Outputs/CrI3_Path_NonHyrbid_Anal_sig_0.5/hybrid_path_lifetimes.csv",
        save_plot="Outputs/CrI3_Path_NonHyrbid_Anal_sig_0.5/hybrid_bands_lifetime_scatter.png",
        vmin=1e0,
        vmax=1e5,
    )
    plot_dense_interpolated_line(
        dense_csv="Outputs/CrI3_Path_NonHyrbid_dense/hybrid_path_properties.csv",
        lifetime_csv="Outputs/CrI3_Path_NonHyrbid_Anal_sig_0.5/hybrid_path_lifetimes.csv",
        save_plot="Outputs/CrI3_Path_NonHyrbid_Anal_sig_0.5/hybrid_bands_lifetime_interpolated.png",
        vmin=1e0,
        vmax=1e5,
    )