import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection


def compute_k_distances(qfrac, reciprocal_lattice=None):
    """
    Cumulative path distance from consecutive fractional q-points, using the
    minimum-image convention to avoid spurious jumps at BZ wrapping.

    If reciprocal_lattice (3x3, rows = b1,b2,b3 in 1/Angstrom - the same
    convention SpinPhony.py/yaml_to_hdf5_band.py store in band.h5) is given,
    distances are metrically exact cartesian distances. Otherwise a naive
    Euclidean distance in fractional-coordinate space is used instead - fine
    for visualization, but segment lengths won't be physically to scale for
    a non-cubic lattice.
    """
    n = qfrac.shape[0]
    dist = np.zeros(n)
    for i in range(1, n):
        dq = qfrac[i] - qfrac[i - 1]
        dq -= np.round(dq)  # minimum image convention
        if reciprocal_lattice is not None:
            dq = dq @ (reciprocal_lattice * 2.0 * np.pi)
        dist[i] = dist[i - 1] + np.linalg.norm(dq)
    return dist


def plot_hybrid_bands_with_lifetime(
    lifetimes_csv="Outputs/hybrid_path_lifetimes.csv",
    band_h5_for_distances=None,
    color_by="tau_ps",
    cmap="rainbow_r",
    save_plot="Outputs/hybrid_bands_lifetime.png",
):
    """
    Plots the hybrid (magnon-phonon polaron) band structure along the
    high-symmetry path as continuous lines, colored per-segment by lifetime.

    Parameters
    ----------
    lifetimes_csv : str
        Path to hybrid_path_lifetimes.csv, as written by SpinPhony.py's
        "Hybrid Path Lifetime Evaluation" section. Expected columns:
        q_idx, qx, qy, qz, branch, energy_meV, gamma_ps-1, tau_ps.
    band_h5_for_distances : str or None
        Optional path to a band.h5 file (as produced by yaml_to_hdf5_band.py)
        to read the true reciprocal lattice for a metrically correct x-axis.
        If omitted, a naive fractional-coordinate distance is used instead.
    color_by : "tau_ps" or "gamma_ps-1"
        Which column to color the bands by (lifetime in ps, or scattering
        rate in ps^-1). Plotted on a log color scale since both quantities
        commonly span several orders of magnitude (e.g. near band
        (anti)crossings). The colorbar range is always the actual finite
        min/max found in the data; any infinite values (tau -> inf where the
        scattering rate is essentially zero) are drawn in the color of the
        max end of the scale rather than excluded or left blank.
    cmap : str
        Matplotlib colormap name.
    save_plot : str or None
        Output image path. Set to None to skip saving.
    """
    if not os.path.exists(lifetimes_csv):
        raise FileNotFoundError(
            f"Could not find '{lifetimes_csv}'. Run the hybrid path lifetime "
            "evaluation in SpinPhony.py first."
        )

    df = pd.read_csv(lifetimes_csv)

    if color_by not in ("tau_ps", "gamma_ps-1"):
        raise ValueError("color_by must be 'tau_ps' or 'gamma_ps-1'")

    n_path = int(df["q_idx"].max()) + 1
    n_bands = int(df["branch"].max()) + 1

    qfrac = df.drop_duplicates("q_idx").sort_values("q_idx")[["qx", "qy", "qz"]].to_numpy()
    energy = df.pivot(index="q_idx", columns="branch", values="energy_meV").sort_index().to_numpy()
    value = df.pivot(index="q_idx", columns="branch", values=color_by).sort_index().to_numpy()

    assert qfrac.shape[0] == n_path and energy.shape == (n_path, n_bands)

    reciprocal_lattice = None
    if band_h5_for_distances is not None:
        import h5py
        with h5py.File(band_h5_for_distances, "r") as f:
            if "reciprocal_lattice" in f:
                reciprocal_lattice = f["reciprocal_lattice"][:]
            else:
                print(
                    f"Warning: 'reciprocal_lattice' not found in {band_h5_for_distances}; "
                    "falling back to fractional-coordinate distances."
                )

    k_dist = compute_k_distances(qfrac, reciprocal_lattice)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Colorbar range = actual finite min/max in the data. Infinite entries
    # (tau -> inf, i.e. essentially zero scattering rate) are clipped to vmax
    # below so they render as the top color of the scale instead of being
    # dropped or distorting the range.
    finite_vals = value[np.isfinite(value)]
    if finite_vals.size == 0:
        raise ValueError(f"No finite values found in column '{color_by}'.")
    vmin, vmax = finite_vals.min(), finite_vals.max()
    if vmin <= 0:
        vmin = finite_vals[finite_vals > 0].min() if np.any(finite_vals > 0) else 1e-9
    norm = mcolors.LogNorm(vmin=1E-4, vmax=vmax)

    cbar_label = r"Lifetime $\tau$ (ps)" if color_by == "tau_ps" else r"Scattering rate $\Gamma$ (ps$^{-1}$)"

    lc = None
    for b in range(n_bands):
        x = k_dist
        y = energy[:, b]
        v = np.clip(value[:, b], vmin, vmax)  # inf -> vmax color
        # Per-segment color = average of the two endpoint values
        seg_vals = 0.5 * (v[:-1] + v[1:])

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(seg_vals)
        lc.set_linewidth(2.0)
        ax.add_collection(lc)

    xlabel = "Path distance (1/Å)" if reciprocal_lattice is not None else "Path distance (fractional, arb. units)"
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Energy (meV)", fontsize=12, fontweight="bold")
    ax.set_xlim(0, k_dist[-1])
    ax.set_ylim(0, energy.max() * 1.05)
    ax.grid(True, axis="x", linestyle="--", color="gray", alpha=0.4)

    if lc is not None:
        cbar = fig.colorbar(lc, ax=ax, pad=0.02)
        cbar.set_label(cbar_label, fontsize=12, fontweight="bold")

    ax.set_title("Hybrid Magnon-Phonon Bands – Lifetime Along the Path", fontsize=13, fontweight="bold")

    plt.tight_layout()

    if save_plot:
        os.makedirs(os.path.dirname(save_plot), exist_ok=True)
        plt.savefig(save_plot, dpi=300)
        print(f"Plot saved to '{save_plot}'")

    plt.show()


if __name__ == "__main__":
    plot_hybrid_bands_with_lifetime(
        lifetimes_csv="Outputs/hybrid_path_lifetimes.csv",
        band_h5_for_distances=None,  # e.g. "Inputs/CrI3/band.h5" for a true cartesian x-axis
        color_by="tau_ps",
        save_plot="Outputs/hybrid_bands_lifetime.png",
    )
