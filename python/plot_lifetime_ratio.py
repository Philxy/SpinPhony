import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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


def plot_mag_vs_tau_pam(df, E_min=22.0, out_png=None, cmap="coolwarm"):
    """Plots Magnon character vs tau_SLC colored by PAM (normalized from -1 to +1 hbar)."""
    mask_energy = df["E_slc"] > E_min if E_min > 0 else np.ones(len(df), dtype=bool)
    df_sub = df[mask_energy]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    title = rf"Modes with $E > {E_min}$ meV" if E_min > 0 else "All Modes"

    if df_sub.empty:
        ax.text(0.5, 0.5, f"No modes found with E > {E_min} meV.",
                ha="center", va="center", transform=ax.transAxes)
        plt.show()
        return fig, ax

    if df_sub["mag_character"].isna().all() or df_sub["phon_AM_z_hbar"].isna().all():
        ax.text(0.5, 0.5, "Required columns ('mag_character' or 'phon_AM_z_hbar') missing.",
                ha="center", va="center", transform=ax.transAxes)
        plt.show()
        return fig, ax

    valid = (
        np.isfinite(df_sub["tau_slc"]) & (df_sub["tau_slc"] > 0) &
        np.isfinite(df_sub["mag_character"]) & (df_sub["mag_character"] > 0) &
        np.isfinite(df_sub["phon_AM_z_hbar"])
    )
    invalid = ~valid

    if invalid.any():
        print(f"Note: {invalid.sum():,} modes omitted from Mag vs Tau (PAM colored) plot due to non-finite values.")

    if valid.any():
        sc = ax.scatter(
            df_sub.loc[valid, "mag_character"],
            df_sub.loc[valid, "tau_slc"],
            c=df_sub.loc[valid, "phon_AM_z_hbar"],
            cmap=cmap,
            norm=mcolors.Normalize(vmin=-1.0, vmax=1.0),
            s=15,
            linewidths=0
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(r"Phonon angular momentum $\ell_z$ ($\hbar$)", fontsize=10)

    ax.set_xlabel("Magnon character", fontsize=11)
    ax.set_ylabel(r"$\tau_{\rm SLC}$ (ps)", fontsize=11)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=12)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    if out_png:
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        print(f"Mag vs Tau (PAM colored) plot saved to '{out_png}'")

    plt.show()
    return fig, ax


def plot_effective_prefactor(df, out_png=None, cmap="viridis"):
    """Plots the effective prefactor R = (1/tau_slc) / mag_character against PAM."""
    fig, ax = plt.subplots(figsize=(6.5, 5))

    if df["phon_AM_z_hbar"].isna().all() or df["mag_character"].isna().all():
        ax.text(0.5, 0.5, "Required columns ('phon_AM_z_hbar', 'mag_character') missing.",
                ha="center", va="center", transform=ax.transAxes)
        plt.show()
        return fig, ax

    valid = (
        np.isfinite(df["tau_slc"]) & (df["tau_slc"] > 0) &
        np.isfinite(df["mag_character"]) & (df["mag_character"] > 0)
    )
    
    if not valid.any():
        ax.text(0.5, 0.5, "No valid data to compute R.",
                ha="center", va="center", transform=ax.transAxes)
        plt.show()
        return fig, ax

    df_valid = df[valid].copy()
    df_valid["R_eff"] = 1.0 / (df_valid["tau_slc"] * df_valid["mag_character"])

    sc = ax.scatter(df_valid["phon_AM_z_hbar"], df_valid["R_eff"],
                    c=df_valid["E_slc"], cmap=cmap, s=15, linewidths=0)
    
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Energy (meV)", fontsize=10)

    ax.set_xlabel(r"Phonon angular momentum $\ell_z$ ($\hbar$)", fontsize=11)
    ax.set_ylabel(r"$R = (1/\tau_{\rm SLC}) \,/\, w_{\rm mag}$ (ps$^{-1}$)", fontsize=11)
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.3)

    if out_png:
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        print(f"Effective prefactor plot saved to '{out_png}'")

    plt.show()
    return fig, ax


def plot_tau_high_energy(df, E_min=22.0, out_png=None, cmap="viridis"):
    """Plots Magnon character and Phonon AM (+ and -) vs tau_SLC for modes above a given energy threshold."""
    mask_energy = df["E_slc"] > E_min if E_min > 0 else np.ones(len(df), dtype=bool)
    df_sub = df[mask_energy]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(rf"Modes with $E > {E_min}$ meV" if E_min > 0 else "All Modes", fontsize=12, y=1.02)

    if df_sub.empty:
        for ax in axes:
            ax.text(0.5, 0.5, f"No modes found with E > {E_min} meV.",
                    ha="center", va="center", transform=ax.transAxes)
        plt.show()
        return fig, axes

    valid = np.isfinite(df_sub["tau_slc"]) & (df_sub["tau_slc"] > 0)
    invalid = ~valid

    if invalid.any():
        print(f"Note: {invalid.sum():,} modes with E > {E_min} meV have infinite/undefined tau_slc "
              "and are omitted from the high-energy Tau plot.")

    sc = None

    plot_configs = [
        (axes[0], valid, "mag_character", "Magnon character", False),
        (axes[1], valid & (df_sub["phon_AM_z_hbar"] > 0), "phon_AM_z_hbar", r"Positive PAM $\ell_z$ ($\hbar$)", False),
        (axes[2], valid & (df_sub["phon_AM_z_hbar"] < 0), "phon_AM_z_hbar", r"Negative PAM $|\ell_z|$ ($\hbar$)", True)
    ]

    for ax, mask, xcol, xlabel, use_abs in plot_configs:
        if df_sub[xcol].isna().all():
            ax.text(0.5, 0.5, f"'{xcol}' not in the SLC file.",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel(xlabel)
            continue

        if mask.any():
            x_data = df_sub.loc[mask, xcol]
            if use_abs:
                x_data = x_data.abs()
                
            sc_curr = ax.scatter(x_data, 
                                 df_sub.loc[mask, "tau_slc"],
                                 c=df_sub.loc[mask, "E_slc"], cmap=cmap,
                                 s=15, linewidths=0)
            if sc_curr is not None:
                sc = sc_curr

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(r"$\tau_{\rm SLC}$ (ps)", fontsize=11)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.3)

    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes, pad=0.02)
        cbar.set_label("Energy (meV)", fontsize=10)

    if out_png:
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        print(f"High-energy Tau plot saved to '{out_png}'")

    plt.show()
    return fig, axes


def plot_mag_vs_energy(df, out_png=None, cmap="viridis"):
    """Plots Energy vs Magnon character colored by SLC lifetime (tau_slc)."""
    fig, ax = plt.subplots(figsize=(6.5, 5))

    if df["mag_character"].isna().all():
        ax.text(0.5, 0.5, "'mag_character' not in the SLC file.",
                ha="center", va="center", transform=ax.transAxes)
        plt.show()
        return fig, ax

    valid = np.isfinite(df["tau_slc"]) & (df["tau_slc"] > 0)
    invalid = ~valid
    
    if invalid.any():
        ax.scatter(df.loc[invalid, "mag_character"], df.loc[invalid, "E_slc"],
                   color="lightgrey", marker="x", s=20, lw=1, alpha=0.8,
                   label=r"$\tau_{\rm SLC} = \infty$")

    if valid.any():
        sc = ax.scatter(df.loc[valid, "mag_character"], df.loc[valid, "E_slc"],
                        c=df.loc[valid, "tau_slc"], cmap=cmap,
                        norm=mcolors.LogNorm(), s=15, linewidths=0)
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(r"$\tau_{\rm SLC}$ (ps)", fontsize=10)

    ax.set_xlabel("Magnon character", fontsize=11)
    ax.set_ylabel("Energy (meV)", fontsize=11)
    ax.set_xscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    
    if invalid.any():
        ax.legend(loc="best")

    if out_png:
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        print(f"Energy vs Magnon plot saved to '{out_png}'")

    plt.show()
    return fig, ax


def plot(df, out_png=None, warn_dE=1.0, cmap="viridis"):
    """Two panels: lifetime ratio against magnon character and against PAM."""
    ok = np.isfinite(df["ratio"]) & (df["ratio"] > 0)
    n_drop = int((~ok).sum())
    df = df[ok]
    if df.empty:
        raise ValueError("Nothing to plot - every ratio was non-finite "
                         "(tau = inf means a mode found no scattering channel).")
    if n_drop:
        print(f"Note: dropped {n_drop:,} modes with non-finite or non-positive ratio from the main plot.")

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
        print(f"Main plot saved to '{out_png}'")

    plt.show()
    return fig, axes


def main():
    p = argparse.ArgumentParser(description="Lifetime ratio with/without SLC vs magnon character and PAM.")
    p.add_argument("--slc", required=True, help="hybrid_path_lifetimes.csv from the SLC-enabled run")
    p.add_argument("--bare", required=True, help="hybrid_path_lifetimes.csv from the --no_slc run")
    p.add_argument("--out", default=None, help="Output PNG for main plot")
    p.add_argument("--out_energy", default=None, help="Output PNG for Energy vs Magnon character plot")
    p.add_argument("--out_tau", default=None, help="Output PNG for high-energy Magnon/PAM vs Tau plot")
    p.add_argument("--out_mag_tau_pam", default=None, help="Output PNG for Mag character vs Tau colored by PAM")
    p.add_argument("--out_R", default=None, help="Output PNG for Effective Prefactor vs PAM plot")
    p.add_argument("--E_min", type=float, default=22.0, help="Energy threshold for the Tau plots (meV)")
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

    # 1. Energy vs Magnon character
    plot_mag_vs_energy(df, out_png=args.out_energy)
    
    # 2. Magnon/PAM character vs Tau (for high energies)
    plot_tau_high_energy(df, E_min=args.E_min, out_png=args.out_tau)

    # 3. New Plot: Magnon character vs Tau (colored by PAM in [-1, 1] hbar)
    plot_mag_vs_tau_pam(df, E_min=args.E_min, out_png=args.out_mag_tau_pam)

    # 4. Effective Prefactor R vs PAM
    plot_effective_prefactor(df, out_png=args.out_R)

    # 5. Original Plot: Ratio vs Magnon character and PAM
    plot(df, out_png=args.out, warn_dE=args.warn_dE)


if __name__ == "__main__":
    main()