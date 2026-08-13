"""
Effect of spin-lattice coupling on mode lifetimes.

Compares two hybrid_path_lifetimes.csv files from the SAME material and path,
one run with SLC enabled and one with --no_slc, and plots the lifetime ratio

    R = tau_SLC / tau_bare

against the magnon character and against the phonon angular momentum (both
taken from the SLC-enabled run, since those are the quantities that only exist
once the modes hybridise).

R < 1 means hybridisation shortened the lifetime.

MODE MATCHING
-------------
Branches are ordered by energy, so the branch index is NOT a stable label: when
SLC pushes levels around an avoided crossing, band n in one file can be band
n+1 in the other. Two strategies:

  --match energy  (default)  per q-point, solve the optimal assignment that
                             minimises the total |E_SLC - E_bare|. Gives a
                             guaranteed bijection and is right wherever the
                             shifts are smaller than the level spacing.
  --match index              naive pairing on the branch index. Faster to
                             reason about, wrong near crossings.

Neither can be right *at* an avoided crossing: there the hybrid modes are
genuine 50/50 mixtures and no one-to-one map to bare modes exists. Points whose
match cost |E_SLC - E_bare| exceeds --warn_dE are flagged and drawn hollow, so
you can see whether the interesting region is also the untrustworthy one.

Usage:
    python plot_slc_lifetime_ratio.py \
        --slc  Outputs/CrI3/hybrid_path_lifetimes.csv \
        --bare Outputs/CrI3_bare/hybrid_path_lifetimes.csv \
        --out  Outputs/CrI3/slc_lifetime_ratio.png
"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


def load_path_csv(path):
    """Loads a SpinPhony path CSV, skipping the '# path_labels:' comment line."""
    with open(path) as f:
        first = f.readline().strip()
    return pd.read_csv(path, skiprows=1) if first.startswith("# path_labels:") else pd.read_csv(path)


def match_modes(df_slc, df_bare, mode="energy"):
    """
    Pairs every (q_idx, branch) of the SLC run with one of the bare run.
    Returns a frame carrying both lifetimes, both energies and the match cost.
    """
    need = ["q_idx", "branch", "energy_meV", "tau_ps"]
    for name, df in (("--slc", df_slc), ("--bare", df_bare)):
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"{name} file is missing column(s): {missing}")

    rows = []
    for q, g_on in df_slc.groupby("q_idx"):
        g_off = df_bare[df_bare["q_idx"] == q]
        if g_off.empty:
            continue

        g_on = g_on.sort_values("branch").reset_index(drop=True)
        g_off = g_off.sort_values("branch").reset_index(drop=True)

        if mode == "index":
            n = min(len(g_on), len(g_off))
            pairs = list(zip(range(n), range(n)))
        else:
            # Optimal assignment minimising total |dE| - a bijection, unlike
            # nearest-neighbour matching which can map two modes onto one.
            cost = np.abs(g_on["energy_meV"].to_numpy()[:, None]
                          - g_off["energy_meV"].to_numpy()[None, :])
            pairs = list(zip(*linear_sum_assignment(cost)))

        for i, j in pairs:
            r_on, r_off = g_on.iloc[i], g_off.iloc[j]
            rows.append({
                "q_idx": q,
                "branch_slc": r_on["branch"],
                "branch_bare": r_off["branch"],
                "E_slc": r_on["energy_meV"],
                "E_bare": r_off["energy_meV"],
                "dE": abs(r_on["energy_meV"] - r_off["energy_meV"]),
                "tau_slc": r_on["tau_ps"],
                "tau_bare": r_off["tau_ps"],
                "mag_character": r_on.get("mag_character", np.nan),
                "phon_AM_z_hbar": r_on.get("phon_AM_z_hbar", np.nan),
                "path_dist": r_on.get("path_dist", np.nan),
            })

    out = pd.DataFrame(rows)
    out["ratio"] = out["tau_slc"] / out["tau_bare"]
    return out


def plot(df, out_png=None, warn_dE=1.0, cmap="viridis"):
    """Two panels: lifetime ratio against magnon character and against PAM."""
    ok = np.isfinite(df["ratio"]) & (df["ratio"] > 0)
    n_drop = int((~ok).sum())
    df = df[ok]
    if df.empty:
        raise ValueError("Nothing to plot - every ratio was non-finite "
                         "(tau = inf means a mode found no scattering channel).")
    if n_drop:
        print(f"Note: dropped {n_drop:,} modes with non-finite or non-positive ratio.")

    good = df["dE"] <= warn_dE
    print(f"Match cost |dE|: median {df['dE'].median():.4f}, max {df['dE'].max():.4f} meV")
    print(f"{int((~good).sum()):,} of {len(df):,} pairs exceed --warn_dE = {warn_dE} meV "
          "(drawn hollow - likely mismatched near avoided crossings).")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, xcol, xlabel in (
        (axes[0], "mag_character", "Magnon character"),
        (axes[1], "phon_AM_z_hbar", r"Phonon angular momentum $\ell_z$ ($\hbar$)"),
    ):
        if df[xcol].isna().all():
            ax.text(0.5, 0.5, f"'{xcol}' not in the SLC file.\nRe-run to write it.",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel(xlabel)
            continue

        sc = ax.scatter(df.loc[good, xcol], df.loc[good, "ratio"],
                        c=df.loc[good, "E_slc"], cmap=cmap, s=12, linewidths=0)
        ax.scatter(df.loc[~good, xcol], df.loc[~good, "ratio"],
                   facecolors="none", edgecolors="crimson", s=18, linewidths=0.6)


        ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.6)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(r"$\tau_{\rm SLC}\,/\,\tau_{\rm bare}$", fontsize=11)
        ax.grid(True, which="both", ls="--", alpha=0.3)

    cbar = fig.colorbar(sc, ax=axes, pad=0.02)
    cbar.set_label("Energy (meV)", fontsize=10)

    if out_png:
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        print(f"Plot saved to '{out_png}'")

    plt.show()
    return fig, axes


def main():
    p = argparse.ArgumentParser(description="Lifetime ratio with/without SLC vs magnon character and PAM.")
    p.add_argument("--slc", required=True, help="hybrid_path_lifetimes.csv from the SLC-enabled run")
    p.add_argument("--bare", required=True, help="hybrid_path_lifetimes.csv from the --no_slc run")
    p.add_argument("--out", default=None, help="Output PNG")
    p.add_argument("--match", choices=["energy", "index"], default="energy",
                   help="Mode pairing strategy (default: energy)")
    p.add_argument("--warn_dE", type=float, default=1.0,
                   help="Flag pairs whose |E_SLC - E_bare| exceeds this (meV)")
    p.add_argument("--csv", default=None, help="Optional path to dump the matched table")
    args = p.parse_args()

    df = match_modes(load_path_csv(args.slc), load_path_csv(args.bare), mode=args.match)
    print(f"Matched {len(df):,} modes across {df['q_idx'].nunique():,} q-points "
          f"using --match {args.match}.")

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"Matched table written to '{args.csv}'")

    plot(df, out_png=args.out, warn_dE=args.warn_dE)


if __name__ == "__main__":
    main()