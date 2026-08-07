import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from scipy.interpolate import interp1d


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


def remap_path_dist(values, source_labels, target_labels):
    """
    Piecewise-linearly rescales `values` (path_dist from the SOURCE file) onto
    the TARGET file's path_dist scale, anchoring each matching pair of labels
    exactly. Necessary whenever combining two path files generated from
    different band.h5 sources (different lattice constants / segment lengths
    in 1/Angstrom) - guarantees shared high-symmetry points align even though
    the raw path_dist scales differ, and don't differ by a uniform factor.

    Matches source_labels[i] to target_labels[i] purely by position in the
    path sequence (both must walk the same route, e.g. Gamma-K-M-Gamma) - the
    label *names* are only used for the sanity check below, not for matching.
    """
    if len(source_labels) != len(target_labels):
        raise ValueError(
            f"Source path has {len(source_labels)} labels "
            f"{[n for n, _ in source_labels]} but target has "
            f"{len(target_labels)} {[n for n, _ in target_labels]} - "
            "these two files walk a different high-symmetry route and "
            "cannot be aligned by simple segment-anchoring."
        )
    mismatched = [(sn, tn) for (sn, _), (tn, _) in zip(source_labels, target_labels) if sn != tn]
    if mismatched:
        print(f"Warning: label name mismatch at matching path positions: {mismatched} "
              "- proceeding anyway (matched by sequence position, not name).")

    values = np.asarray(values, dtype=float)
    remapped = np.empty_like(values)
    n_seg = len(source_labels) - 1
    for i in range(n_seg):
        s0, s1 = source_labels[i][1], source_labels[i + 1][1]
        t0, t1 = target_labels[i][1], target_labels[i + 1][1]
        # Include the segment's right edge only on the last segment, so every
        # point is assigned to exactly one segment with no gaps/overlaps.
        if i < n_seg - 1:
            mask = (values >= s0) & (values < s1)
        else:
            mask = (values >= s0) & (values <= s1 + 1e-9)
        frac = (values[mask] - s0) / (s1 - s0)
        remapped[mask] = t0 + frac * (t1 - t0)
    return remapped


def nearest_q_path_dist(q_frac, dense_q_frac, dense_path_dist):
    """
    For each row in q_frac (N,3), finds the closest point in dense_q_frac
    (M,3) under the minimum-image convention (fractional coordinates, so
    periodicity in each component is accounted for) and returns that
    neighbor's path_dist and the matching distance (in fractional units).

    Needed because a "sparse" path built by snapping to the nearest points
    on a Monkhorst-Pack grid (e.g. picking near-K points on a 32x32x32 mesh)
    does not actually sit on the ideal high-symmetry line - e.g. 2/3 is not
    representable on a 32-point grid, so the nearest reachable point is
    20/32 = 0.625 instead of 0.666667. Two files built this way have
    genuinely different sets of q-points per segment, not just different
    path_dist scales, so anchoring only the segment endpoints (label-based
    linear rescaling) cannot correctly place the interior points - matching
    by actual reciprocal-space location is required instead.
    """
    q_frac = np.asarray(q_frac, dtype=float)
    dense_q_frac = np.asarray(dense_q_frac, dtype=float)
    dense_path_dist = np.asarray(dense_path_dist, dtype=float)

    diff = dense_q_frac[None, :, :] - q_frac[:, None, :]
    diff -= np.round(diff)
    dist2 = np.sum(diff ** 2, axis=-1)
    nearest_idx = np.argmin(dist2, axis=1)
    nearest_dist = np.sqrt(dist2[np.arange(len(q_frac)), nearest_idx])
    return dense_path_dist[nearest_idx], nearest_dist


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


def _load_and_align(dense_csv, lifetime_csv, tau_col):
    """
    Loads both files and reassigns the lifetime file's path_dist to that of
    its nearest neighbor (in actual fractional q-space) in the dense file.
    Returns (df_disp, df_life, labels_dense).

    This is more robust than label-anchored rescaling when the "sparse" file
    was built by snapping to the nearest points on a coarse Monkhorst-Pack
    grid: those points don't actually lie on the ideal high-symmetry line,
    so segment-endpoint anchoring alone can't place the interior points
    correctly - matching by physical location can.
    """
    df_disp, labels_dense = load_path_csv(dense_csv)
    df_life, labels_sparse = load_path_csv(lifetime_csv)

    q_cols = ["qx", "qy", "qz"]
    if all(c in df_disp.columns for c in q_cols) and all(c in df_life.columns for c in q_cols):
        dense_unique = df_disp.drop_duplicates(subset=q_cols)
        df_life = df_life.copy()
        new_path_dist, match_dist = nearest_q_path_dist(
            df_life[q_cols].to_numpy(),
            dense_unique[q_cols].to_numpy(),
            dense_unique["path_dist"].to_numpy(),
        )
        df_life["path_dist"] = new_path_dist
        if match_dist.max() > 1e-3:
            print(f"Warning: nearest dense q-point match distance up to "
                  f"{match_dist.max():.4f} (fractional units) - the sparse "
                  "path points may not lie close to the dense path (e.g. "
                  "grid-snapping error or a genuinely different route).")
    else:
        print("Warning: qx/qy/qz columns missing from one of the two files - "
              "cannot align by reciprocal-space location.")

    df_life = _prepare_tau(df_life, tau_col)
    return df_disp, df_life, labels_dense


def plot_dense_with_scatter(
    dense_csv="Outputs/HybridDense/hybrid_path_properties.csv",
    lifetime_csv="Outputs/Hybrid_band32/hybrid_path_lifetimes.csv",
    tau_col="tau_ps",
    cmap="rainbow_r",
    vmin=1.0,
    vmax=1e5,
    save_plot="Outputs/Hybrid/hybrid_bands_lifetime_scatter.png",
):
    """
    Figure 1: dense hybrid band structure (from hybrid_path_properties.csv,
    plotted as thin grey lines per band) with the (typically sparser)
    lifetimes from hybrid_path_lifetimes.csv overlaid as color-coded scatter
    points on a log scale. The lifetime file's path_dist is first remapped
    onto the dense file's scale via remap_path_dist, so this is correct even
    when the two files were generated from different band.h5 sources.
    """
    df_disp, df_life, labels = _load_and_align(dense_csv, lifetime_csv, tau_col)
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8, 6))

    for band in sorted(df_disp["band"].unique()):
        subset = df_disp[df_disp["band"] == band].sort_values("path_dist")
        ax.plot(subset["path_dist"], subset["energy_meV"], color="lightgrey", lw=1, zorder=1)

    sc = ax.scatter(
        df_life["path_dist"], df_life["energy_meV"],
        c=df_life[tau_col], cmap=cmap, norm=norm,
        s=30, zorder=2, edgecolors="k", linewidth=0.5,
    )

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"Lifetime $\tau$ (ps) [log scale]", fontsize=12, fontweight="bold")

    ax.set_xlim(df_disp["path_dist"].min(), df_disp["path_dist"].max())
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Path distance (1/A)", fontsize=12)
    ax.set_ylabel("Energy (meV)", fontsize=12, fontweight="bold")
    ax.set_title("Hybrid Bands with Lifetime Scatter", fontsize=13, fontweight="bold")
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
    band, colored by the (sparser) lifetime data log-linearly interpolated
    onto the dense path's 'path_dist' grid (after remapping the lifetime
    file's path_dist onto the dense file's scale via shared labels). Matching
    between the dense 'band' column and the sparse 'branch' column is by
    shared branch index - both are the same physical hybrid-mode ordering
    (num_phon + num_mag branches) as long as both files are for the same
    material.
    """
    df_disp, df_life, labels = _load_and_align(dense_csv, lifetime_csv, tau_col)
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8, 6))

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
        cbar = fig.colorbar(lc, ax=ax)
        cbar.set_label(r"Interpolated lifetime $\tau$ (ps) [log scale]", fontsize=12, fontweight="bold")

    ax.set_xlim(df_disp["path_dist"].min(), df_disp["path_dist"].max())
    ax.set_ylim(df_disp["energy_meV"].min(), df_disp["energy_meV"].max() * 1.05)
    ax.set_xlabel("Path distance (1/A)", fontsize=12)
    ax.set_ylabel("Energy (meV)", fontsize=12, fontweight="bold")
    ax.set_title("Hybrid Bands Colored by Interpolated Lifetime", fontsize=13, fontweight="bold")
    set_path_ticks(ax, labels)

    fig.tight_layout()
    plt.show()

    if save_plot:
        os.makedirs(os.path.dirname(save_plot), exist_ok=True)
        fig.savefig(save_plot, dpi=300)
        print(f"Plot saved to '{save_plot}'")

    return fig, ax


if __name__ == "__main__":

    df_dense, labels_dense = load_path_csv("Outputs/HybridDense/hybrid_path_properties.csv")
    df_sparse, labels_sparse = load_path_csv("Outputs/Hybrid_band32/hybrid_path_lifetimes.csv")

    print("Dense labels: ", labels_dense)
    print("Sparse labels:", labels_sparse)

    # Check a few sparse points' original vs nearest-dense-q-matched path_dist
    q_cols = ["qx", "qy", "qz"]
    dense_unique = df_dense.drop_duplicates(subset=q_cols)
    raw = df_sparse["path_dist"].to_numpy()
    remapped, match_dist = nearest_q_path_dist(
        df_sparse[q_cols].to_numpy(), dense_unique[q_cols].to_numpy(), dense_unique["path_dist"].to_numpy()
    )
    for i in range(0, len(raw), max(1, len(raw)//15)):
        print(f"branch={df_sparse['branch'].iloc[i]}  raw={raw[i]:.4f}  remapped={remapped[i]:.4f}  "
              f"match_dist={match_dist[i]:.4f}  energy={df_sparse['energy_meV'].iloc[i]:.3f}")


    plot_dense_with_scatter(
        dense_csv="Outputs/HybridDense/hybrid_path_properties.csv",
        lifetime_csv="Outputs/Hybrid_band32/hybrid_path_lifetimes.csv",
        save_plot="Outputs/Hybrid/hybrid_bands_lifetime_scatter.png",
    )
    plot_dense_interpolated_line(
        dense_csv="Outputs/HybridDense/hybrid_path_properties.csv",
        lifetime_csv="Outputs/Hybrid_band32/hybrid_path_lifetimes.csv",
        save_plot="Outputs/Hybrid/hybrid_bands_lifetime_interpolated.png",
    )
