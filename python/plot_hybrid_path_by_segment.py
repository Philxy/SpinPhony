"""
Alternative to plot_hybrid_path_lifetimes.py's alignment strategy.

Instead of trusting each file's stored cumulative 'path_dist' (which for a
coarse/grid-snapped path can drift off the true straight-line high-symmetry
segment), this script recomputes every point's position from scratch using
its own qx,qy,qz:

  1. Split the path into segments using the high-symmetry labels (e.g.
     Gamma-K, K-M, M-Gamma) and their known path_dist boundaries.
  2. Bucket every point into the segment its native path_dist falls in.
  3. Within a segment, find the two rows that sit at the segment's start/end
     label distance and use their q-vectors as the segment's endpoints.
  4. For every point in that segment, compute t = (q_point - q_start) . (q_end
     - q_start) / |q_end - q_start|^2 - the orthogonal projection onto the
     segment direction, i.e. its fractional position along the ideal line
     (not its raw distance from q_start, which would also pick up any
     perpendicular deviation from the line, e.g. from grid snapping) - and
     place it at start + t * (end - start) on the target label scale.

This works for the dense and the sparse file independently and self-
consistently: each file only ever uses its own q-vectors and its own labels,
so no cross-file q-point matching or basis assumptions are needed. The two
files are combined only through the shared label names/ordering.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from scipy.interpolate import interp1d


def load_path_csv(path):
    """
    Loads a SpinPhony.py path-output CSV. These start with a
    "# path_labels: G=0.000000,K=1.234567,..." comment line followed by the
    normal header, and every row carries qx,qy,qz plus a 'path_dist' column.

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


def place_points_on_segments(df, labels, target_labels=None, q_cols=("qx", "qy", "qz")):
    """
    Recomputes each row's path_dist from its own q-vector, segment by
    segment, rather than trusting the file's stored cumulative path_dist.

    labels:        this file's own (name, path_dist) list - used only to (a)
                   decide which segment each row belongs to, via its native
                   path_dist, and (b) find each segment's start/end q-vector
                   (the row whose native path_dist is closest to that label).
    target_labels: the label distances to place the *output* on (defaults to
                   `labels`, i.e. no cross-file remapping). Pass the OTHER
                   file's labels to align this file onto that file's scale.

    Returns a new path_dist array, same length/order as df.
    """
    if target_labels is None:
        target_labels = labels

    if len(labels) != len(target_labels):
        raise ValueError(
            f"Source path has {len(labels)} labels {[n for n, _ in labels]} "
            f"but target has {len(target_labels)} {[n for n, _ in target_labels]} "
            "- these two files walk a different high-symmetry route and "
            "cannot be aligned this way."
        )
    mismatched = [(sn, tn) for (sn, _), (tn, _) in zip(labels, target_labels) if sn != tn]
    if mismatched:
        print(f"Warning: label name mismatch at matching path positions: {mismatched} "
              "- proceeding anyway (matched by sequence position, not name).")

    q = df[list(q_cols)].to_numpy(dtype=float)
    native_dist = df["path_dist"].to_numpy(dtype=float)
    n_seg = len(labels) - 1

    # Anchor q-vector for every label: the row whose native path_dist sits
    # closest to that label's recorded distance.
    anchor_q = np.array([q[np.argmin(np.abs(native_dist - dist))] for _, dist in labels])

    new_dist = native_dist.copy()
    assigned = np.zeros(len(df), dtype=bool)

    for i in range(n_seg):
        s0, s1 = labels[i][1], labels[i + 1][1]
        t0, t1 = target_labels[i][1], target_labels[i + 1][1]
        q_start, q_end = anchor_q[i], anchor_q[i + 1]
        seg_vec = q_end - q_start
        seg_frac_len2 = np.dot(seg_vec, seg_vec)

        # Right edge included only on the last segment, so every point is
        # assigned to exactly one segment with no gaps/overlaps.
        if i < n_seg - 1:
            mask = (native_dist >= s0) & (native_dist < s1)
        else:
            mask = (native_dist >= s0) & (native_dist <= s1 + 1e-9)

        if not np.any(mask):
            continue

        if seg_frac_len2 < 1e-24:
            t = np.zeros(mask.sum())
        else:
            disp = q[mask] - q_start[None, :]
            # Orthogonal projection onto the segment direction, not the raw
            # displacement magnitude: a point that's off the ideal straight
            # line (e.g. grid-snapped) has a perpendicular component too,
            # and including that in the distance overestimates how far along
            # the segment the point actually is - projecting keeps only the
            # longitudinal component, the physically meaningful path
            # coordinate.
            t = (disp @ seg_vec) / seg_frac_len2
            t = np.clip(t, 0.0, 1.0)

        new_dist[mask] = t0 + t * (t1 - t0)
        assigned[mask] = True

    n_missing = (~assigned).sum()
    if n_missing:
        print(f"Warning: {n_missing} rows did not fall inside any labeled segment's "
              "native path_dist range - left at their original path_dist.")

    return new_dist


def set_path_ticks(ax, labels):
    """Applies high-symmetry point tick labels + vertical gridlines to ax."""
    if not labels:
        return
    ax.set_xticks([pos for _, pos in labels])
    ax.set_xticklabels([name for name, _ in labels], fontsize=13)
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


def _load_and_align(dense_csv, lifetime_csv, tau_col):
    """
    Loads both files and recomputes path_dist for every row via
    place_points_on_segments: the dense file onto its own label scale (so
    it's self-consistent even if its own native path_dist wasn't exactly
    straight-line), and the sparse file onto the dense file's label scale.
    """
    df_disp, labels_dense = load_path_csv(dense_csv)
    df_life, labels_sparse = load_path_csv(lifetime_csv)

    if labels_dense:
        df_disp = df_disp.copy()
        df_disp["path_dist"] = place_points_on_segments(df_disp, labels_dense)

    if labels_dense and labels_sparse:
        df_life = df_life.copy()
        df_life["path_dist"] = place_points_on_segments(df_life, labels_sparse, target_labels=labels_dense)
    elif labels_dense or labels_sparse:
        print("Warning: only one of the two files has path_labels - skipping "
              "path_dist alignment.")

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
    Figure 1: dense hybrid band structure (plotted as thin grey lines per
    band) with the sparser lifetimes overlaid as color-coded scatter points
    on a log scale. Both files' path_dist are recomputed via
    place_points_on_segments before plotting.
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
    band, colored by the sparser lifetime data log-linearly interpolated
    onto the dense path's (segment-recomputed) path_dist grid. Matching
    between the dense 'band' column and the sparse 'branch' column is by
    shared branch index.
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
