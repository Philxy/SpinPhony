"""
Alternative alignment strategy: purely geometric, and does not trust
anything about the sparse (lifetime) file's own segment/label bookkeeping.

Rationale: the sparse path is built by snapping intended path points onto a
coarse Monkhorst-Pack grid. Its own recorded "K" position may not correspond
to an actual sampled point at all - it may just be whichever grid point was
closest at the time the path was generated, with no guarantee that any row
in the file is truly "at K". So instead of trusting the sparse file's own
path_labels comment or its 'path_dist' column for anything, this script:

  1. Reads the DENSE file's labels (trusted - finely sampled, points really
     do sit at Gamma/K/M) and looks up each label's q-vector from the row
     nearest that label's own recorded path_dist. These give three "anchor"
     q-vectors forming n_seg reference line segments (Gamma-K, K-M, M-Gamma).
  2. For the SPARSE file, ignores its path_labels/path_dist entirely - reads
     only qx,qy,qz. For every sparse point, computes its distance to each of
     the n_seg reference *segments* (clamped to the segment, not the
     infinite line) and assigns it to whichever segment it's closest to.
  3. Places the point on that segment via orthogonal projection (t = (P-A).
     (B-A)/|B-A|^2, clamped to [0,1]) mapped onto the dense file's own label
     distances.

The dense file is placed the same way, using its own labels as both
reference and target, so both files go through an identical procedure.
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
    Loads a SpinPhony.py path-output CSV, parsing the leading
    "# path_labels: G=0.000000,K=1.234567,..." comment line (if present)
    into an ORDERED list of (name, path_dist) pairs - NOT a dict, since a
    closed path revisits the same label (e.g. Gamma at both the start and
    end), which a dict would silently collapse.

    Returns (df, labels); labels is [] if the comment line is absent.
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

    return df, labels


def get_label_anchor_q(df, labels, q_cols=("qx", "qy", "qz")):
    """
    For each label, finds the q-vector of the row whose OWN path_dist is
    closest to that label's recorded distance. Meant to be used only on a
    trusted (finely-sampled) file, whose own path_dist and labels can be
    relied on to locate the true high-symmetry points.
    """
    q = df[list(q_cols)].to_numpy(dtype=float)
    native_dist = df["path_dist"].to_numpy(dtype=float)
    return np.array([q[np.argmin(np.abs(native_dist - dist))] for _, dist in labels])


def place_points_by_geometry(q_points, anchor_q, label_dists):
    """
    Purely geometric placement: does not use any path_dist/label/segment
    information from the source of q_points - only its raw qx,qy,qz.

    For each point, computes the distance to every reference segment
    (anchor_q[i] -> anchor_q[i+1], clamped to the segment itself, not the
    infinite line) and assigns the point to whichever segment it's closest
    to. Its position along that segment is the orthogonal projection
    fraction t = (P-A).(B-A)/|B-A|^2, clamped to [0,1], mapped onto
    label_dists[i] -> label_dists[i+1].

    Returns (path_dist, segment_index, distance_to_assigned_segment) - the
    last two are useful diagnostics: a large distance_to_assigned_segment
    for many points means they don't actually sit near any of the reference
    lines (e.g. a genuinely different route or a basis mismatch), which no
    amount of projection can fix.
    """
    q_points = np.asarray(q_points, dtype=float)
    anchor_q = np.asarray(anchor_q, dtype=float)
    n_seg = len(anchor_q) - 1
    N = q_points.shape[0]

    best_dist = np.full(N, np.inf)
    best_t = np.zeros(N)
    best_seg = np.full(N, -1, dtype=int)

    for i in range(n_seg):
        A, B = anchor_q[i], anchor_q[i + 1]
        seg_vec = B - A
        seg_len2 = np.dot(seg_vec, seg_vec)
        disp = q_points - A[None, :]

        if seg_len2 < 1e-24:
            t_raw = np.zeros(N)
        else:
            t_raw = (disp @ seg_vec) / seg_len2
        t_clamped = np.clip(t_raw, 0.0, 1.0)

        closest = A[None, :] + t_clamped[:, None] * seg_vec[None, :]
        dist = np.linalg.norm(q_points - closest, axis=1)

        better = dist < best_dist
        best_dist[better] = dist[better]
        best_t[better] = t_raw[better]
        best_seg[better] = i

    path_dist = np.zeros(N)
    for i in range(n_seg):
        mask = best_seg == i
        if not np.any(mask):
            continue
        t0, t1 = label_dists[i], label_dists[i + 1]
        t = np.clip(best_t[mask], 0.0, 1.0)
        path_dist[mask] = t0 + t * (t1 - t0)

    return path_dist, best_seg, best_dist


def set_path_ticks(ax, labels):
    if not labels:
        return
    ax.set_xticks([pos for _, pos in labels])
    ax.set_xticklabels([name for name, _ in labels], fontsize=13)
    ax.grid(True, axis="x", linestyle="--", color="gray", alpha=0.5)


def _prepare_tau(df, tau_col="tau_ps"):
    df = df.copy()
    finite = df.loc[np.isfinite(df[tau_col]), tau_col]
    if finite.empty:
        raise ValueError(f"No finite values found in column '{tau_col}'.")
    max_finite = finite.max()
    df[tau_col] = df[tau_col].replace([np.inf, -np.inf], max_finite)
    min_pos = df.loc[df[tau_col] > 0, tau_col].min()
    df[tau_col] = df[tau_col].clip(lower=min_pos)
    return df


def _load_and_align(dense_csv, lifetime_csv, tau_col, verbose=True):
    """
    Loads the dense file normally (trusted labels + path_dist). Loads the
    sparse/lifetime file's qx,qy,qz only - its own path_labels/path_dist are
    never used. Both are then placed via place_points_by_geometry against
    the dense file's own label anchors.
    """
    df_disp, labels_dense = load_path_csv(dense_csv)
    if not labels_dense:
        raise ValueError(f"{dense_csv} has no path_labels - cannot use it as the geometric reference.")

    label_dists = [d for _, d in labels_dense]
    anchor_q = get_label_anchor_q(df_disp, labels_dense)

    q_cols = ["qx", "qy", "qz"]
    df_disp = df_disp.copy()
    new_dense_dist, _, dense_match_dist = place_points_by_geometry(
        df_disp[q_cols].to_numpy(), anchor_q, label_dists
    )
    df_disp["path_dist"] = new_dense_dist

    # Sparse file: read q-columns only, ignore its own path_dist/labels entirely.
    df_life, _ = load_path_csv(lifetime_csv)
    df_life = df_life.copy()
    new_sparse_dist, sparse_seg, sparse_match_dist = place_points_by_geometry(
        df_life[q_cols].to_numpy(), anchor_q, label_dists
    )
    df_life["path_dist"] = new_sparse_dist

    if verbose:
        print(f"Dense self-consistency check: max perpendicular deviation from its own "
              f"reference lines = {dense_match_dist.max():.6f} (fractional units)")
        print(f"Sparse points: max perpendicular deviation from nearest dense reference "
              f"line = {sparse_match_dist.max():.6f} (fractional units), "
              f"mean = {sparse_match_dist.mean():.6f}")
        unassigned = (sparse_seg < 0).sum()
        if unassigned:
            print(f"Warning: {unassigned} sparse points could not be assigned to any segment.")

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
    ax.set_title("Hybrid Bands with Lifetime Scatter (geometric alignment)", fontsize=13, fontweight="bold")
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
        seg_vals = np.sqrt(tau_dense[:-1] * tau_dense[1:])

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
    ax.set_title("Hybrid Bands Colored by Interpolated Lifetime (geometric alignment)", fontsize=13, fontweight="bold")
    set_path_ticks(ax, labels)

    fig.tight_layout()
    plt.show()

    if save_plot:
        os.makedirs(os.path.dirname(save_plot), exist_ok=True)
        fig.savefig(save_plot, dpi=300)
        print(f"Plot saved to '{save_plot}'")

    return fig, ax


if __name__ == "__main__":

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
