import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import linear_sum_assignment
from scipy.stats import linregress, spearmanr, mannwhitneyu


# ======================================================================
# Loading / matching (unchanged)
# ======================================================================
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
                "gamma_ps-1": r_on.get("gamma_ps-1"),
            })

    out = pd.DataFrame(rows)
    out["ratio"] = out["tau_slc"] / out["tau_bare"]
    return out


def estimate_E_mag_max(df, mag_threshold=0.9):
    """
    Data-driven estimate of the magnon band top: the highest energy among
    modes that are still nearly pure magnon (mag_character > mag_threshold).
    Used as the reference for the detuning/g^2/Delta^2 test. Falls back to
    NaN with a warning if no sufficiently pure-magnon modes are present.
    """
    pure = df[np.isfinite(df["mag_character"]) & (df["mag_character"] > mag_threshold)]
    if pure.empty:
        print(f"[warn] no modes with mag_character > {mag_threshold} -- cannot "
              f"estimate E_mag_max from this dataset. Pass --E_mag_max explicitly.")
        return np.nan
    E_max = pure["E_slc"].max()
    print(f"[estimate] E_mag_max ~= {E_max:.3f} meV "
          f"(from {len(pure)} modes with mag_character > {mag_threshold})")
    return E_max


# ======================================================================
# Existing plots (unchanged, except E_min threaded consistently through
# plot_effective_prefactor instead of a hardcoded value)
# ======================================================================
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
            linewidths=0,
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


def plot_effective_prefactor(df, E_min=1.0, out_png=None, cmap="viridis", mag_char_floor=1e-5):
    """Plots the effective prefactor R = (1/tau_slc) / mag_character against PAM."""
    fig, ax = plt.subplots(figsize=(6.5, 5))

    if df["phon_AM_z_hbar"].isna().all() or df["mag_character"].isna().all():
        ax.text(0.5, 0.5, "Required columns ('phon_AM_z_hbar', 'mag_character') missing.",
                ha="center", va="center", transform=ax.transAxes)
        plt.show()
        return fig, ax

    # mag_character_floor excludes near-zero admixture noise, not just exact
    # zero -- without this, a handful of ~1e-8-character points can dominate
    # the log-y range and squash the real structure into invisibility.
    valid = (
        np.isfinite(df["tau_slc"]) & (df["tau_slc"] > 0) &
        np.isfinite(df["mag_character"]) & (df["mag_character"] > mag_char_floor) &
        (df["E_slc"] > E_min)
    )

    if not valid.any():
        ax.text(0.5, 0.5, "No valid data to compute R.",
                ha="center", va="center", transform=ax.transAxes)
        plt.show()
        return fig, ax

    df_valid = df[valid].copy()
    df_valid["R_eff"] = df_valid["gamma_ps-1"] / df_valid["mag_character"]

    sc = ax.scatter(df_valid["phon_AM_z_hbar"], df_valid["R_eff"],
                    c=df_valid["phon_AM_z_hbar"].abs(), cmap=cmap,
                    s=15, linewidths=0, alpha=0.5)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(r"$|\ell_z|$ ($\hbar$)", fontsize=10)

    ax.set_xlabel(r"Phonon angular momentum $\ell_z$ ($\hbar$)", fontsize=11)
    ax.set_ylabel(r"$R = (1/\tau_{\rm SLC}) \,/\, w_{\rm mag}$ (ps$^{-1}$)", fontsize=11)
    ax.set_yscale("log")
    ax.axvline(0, color="gray", lw=0.6, ls="--")
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
        (axes[2], valid & (df_sub["phon_AM_z_hbar"] < 0), "phon_AM_z_hbar", r"Negative PAM $|\ell_z|$ ($\hbar$)", True),
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

            sc_curr = ax.scatter(x_data, df_sub.loc[mask, "tau_slc"],
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


# ======================================================================
# NEW: statistical hypothesis tests for the dependencies in question
# ======================================================================
def test_power_law(df, E_min=22.0, out_png=None):
    """
    Verifies 1/tau_SLC ~ mag_character^1 for the previously-forbidden
    (E > E_min) population, via a log-log linear regression. Reports the
    fitted exponent (expected ~1 from leading-order perturbation theory)
    with its uncertainty, and R^2 of the fit.
    """
    sub = df[(df["E_slc"] > E_min) & np.isfinite(df["tau_slc"]) & (df["tau_slc"] > 0) &
             np.isfinite(df["mag_character"]) & (df["mag_character"] > 0)]
    if len(sub) < 5:
        print(f"[power law] too few valid points (n={len(sub)}) for E > {E_min} meV -- skipping.")
        return None

    x = np.log10(sub["mag_character"].to_numpy())
    y = np.log10(sub["gamma_ps-1"].to_numpy())
    res = linregress(x, y)

    print(f"\n[power law: 1/tau ~ mag_character^p, E > {E_min} meV, n={len(sub)}]")
    print(f"  fitted exponent p = {res.slope:.3f} +/- {res.stderr:.3f}  (expect ~1)")
    print(f"  R^2 = {res.rvalue**2:.4f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(sub["mag_character"], sub["gamma_ps-1"], s=10, alpha=0.4, color="tab:blue")
    xx = np.logspace(x.min(), x.max(), 50)
    ax.plot(xx, 10**res.intercept * xx**res.slope, color="tab:red", lw=2,
            label=rf"fit: $p={res.slope:.2f}\pm{res.stderr:.2f}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Magnon character $w_{\\rm mag}$", fontsize=11)
    ax.set_ylabel(r"$1/\tau_{\rm SLC}$ (ps$^{-1}$)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    if out_png:
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    return res


def test_detuning_law(df, E_mag_max, E_min=None, out_png=None):
    """
    Tests the closed-form 1/tau ~ g^2/Delta^2 prediction, Delta = E - E_mag_max,
    for modes above the magnon band top. Fits log(1/tau) vs log(Delta) and
    reports the exponent (expected ~ -2).
    """
    if not np.isfinite(E_mag_max):
        print("[detuning law] E_mag_max not available -- skipping.")
        return None
    E_min = E_min if E_min is not None else E_mag_max
    sub = df[(df["E_slc"] > E_mag_max) & np.isfinite(df["tau_slc"]) & (df["tau_slc"] > 0)].copy()
    sub["delta"] = sub["E_slc"] - E_mag_max
    sub = sub[sub["delta"] > 1e-3]
    if len(sub) < 5:
        print(f"[detuning law] too few valid points (n={len(sub)}) -- skipping.")
        return None

    x = np.log10(sub["delta"].to_numpy())
    y = np.log10(sub["gamma_ps-1"].to_numpy())
    res = linregress(x, y)

    print(f"\n[detuning law: 1/tau ~ Delta^p, Delta = E - E_mag_max, n={len(sub)}]")
    print(f"  fitted exponent p = {res.slope:.3f} +/- {res.stderr:.3f}  (expect ~ -2)")
    print(f"  R^2 = {res.rvalue**2:.4f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(sub["delta"], sub["gamma_ps-1"], s=10, alpha=0.4, color="tab:green")
    xx = np.logspace(x.min(), x.max(), 50)
    ax.plot(xx, 10**res.intercept * xx**res.slope, color="tab:red", lw=2,
            label=rf"fit: $p={res.slope:.2f}\pm{res.stderr:.2f}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta = E - E_{\rm mag}^{\rm max}$ (meV)", fontsize=11)
    ax.set_ylabel(r"$1/\tau_{\rm SLC}$ (ps$^{-1}$)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    if out_png:
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    return res


def test_sign_asymmetry(df, E_min=22.0, pam_threshold=0.2, mag_char_floor=1e-5):
    """
    Tests whether R = (1/tau)/mag_character is systematically elevated for
    negative-PAM modes vs positive-PAM modes of comparable |PAM| -- the
    direct prediction of the fixed-sign-magnon (S_z always <= 0) argument.
    Reports medians and a one-sided Mann-Whitney U test (no functional form
    assumed).
    """
    sub = df[(df["E_slc"] > E_min) & np.isfinite(df["tau_slc"]) & (df["tau_slc"] > 0) &
             np.isfinite(df["mag_character"]) & (df["mag_character"] > mag_char_floor) &
             np.isfinite(df["phon_AM_z_hbar"])].copy()
    sub["R_eff"] = sub["gamma_ps-1"] / sub["mag_character"]

    neg = sub[sub["phon_AM_z_hbar"] < -pam_threshold]
    pos = sub[sub["phon_AM_z_hbar"] > pam_threshold]

    print(f"\n[sign asymmetry test, E > {E_min} meV, |PAM| > {pam_threshold}]")
    print(f"  negative-PAM: n={len(neg)}, median R = {neg['R_eff'].median():.4e}")
    print(f"  positive-PAM: n={len(pos)}, median R = {pos['R_eff'].median():.4e}")

    if len(neg) < 3 or len(pos) < 3:
        print("  too few points in one or both groups for a Mann-Whitney test.")
        return None

    stat, p_greater = mannwhitneyu(neg["R_eff"], pos["R_eff"], alternative="greater")
    stat2, p_two = mannwhitneyu(neg["R_eff"], pos["R_eff"], alternative="two-sided")
    print(f"  Mann-Whitney U (neg > pos): p = {p_greater:.4e}")
    print(f"  Mann-Whitney U (two-sided): p = {p_two:.4e}")
    return {"n_neg": len(neg), "n_pos": len(pos),
            "median_neg": neg["R_eff"].median(), "median_pos": pos["R_eff"].median(),
            "p_greater": p_greater, "p_two_sided": p_two}


def test_gaussian_selection_rule(df, E_min=22.0, mag_char_floor=1e-5, out_png=None):
    """
    Tests whether log(R) is quadratic in PAM around a target chirality
    (Gaussian selection-rule form |Gamma|^2 ~ exp(-(l_z - target)^2 / 2w^2)),
    for target = -1 and target = +1 separately, reporting which gives the
    better fit -- a quantitative test of *which* sign is actually favored,
    rather than assuming it.
    """
    sub = df[(df["E_slc"] > E_min) & np.isfinite(df["tau_slc"]) & (df["tau_slc"] > 0) &
             np.isfinite(df["mag_character"]) & (df["mag_character"] > mag_char_floor) &
             np.isfinite(df["phon_AM_z_hbar"])].copy()
    sub["R_eff"] = sub["gamma_ps-1"] / sub["mag_character"]
    sub = sub[sub["R_eff"] > 0]

    results = {}
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for target, color in [(-1.0, "tab:blue"), (1.0, "tab:orange")]:
        xx = (sub["phon_AM_z_hbar"] - target) ** 2
        yy = np.log(sub["R_eff"])
        res = linregress(xx, yy)
        results[target] = res
        print(f"\n[Gaussian selection rule test, target l_z = {target:+.0f}, "
              f"E > {E_min} meV, n={len(sub)}]")
        print(f"  slope (-1/2w^2) = {res.slope:.4f} +/- {res.stderr:.4f}")
        print(f"  R^2 = {res.rvalue**2:.4f}")
        if res.slope < 0:
            w = np.sqrt(-1.0 / (2.0 * res.slope))
            print(f"  implied selection-rule width w = {w:.3f} hbar")

        ax.scatter(xx, yy, s=8, alpha=0.3, color=color, label=f"target={target:+.0f}")
        xr = np.linspace(xx.min(), xx.max(), 50)
        ax.plot(xr, res.intercept + res.slope * xr, color=color, lw=2)

    ax.set_xlabel(r"$(\ell_z - {\rm target})^2$", fontsize=11)
    ax.set_ylabel(r"$\ln R$", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, ls="--", alpha=0.3)
    if out_png:
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()

    better = max(results, key=lambda t: results[t].rvalue**2)
    print(f"\n  => target l_z = {better:+.0f} gives the better fit "
          f"(R^2 = {results[better].rvalue**2:.4f} vs "
          f"{results[-better].rvalue**2:.4f})")
    return results


def test_spearman_conditioned(df, E_min=22.0, mag_char_floor=1e-5,
                               thresholds=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5)):
    """
    Spearman rank correlation between PAM and log(R), computed separately
    for increasingly restrictive |PAM| thresholds -- tests whether the
    correlation strengthens as the sample is restricted toward the
    near-resonant subset, without assuming a specific functional form.
    """
    sub = df[(df["E_slc"] > E_min) & np.isfinite(df["tau_slc"]) & (df["tau_slc"] > 0) &
             np.isfinite(df["mag_character"]) & (df["mag_character"] > mag_char_floor) &
             np.isfinite(df["phon_AM_z_hbar"])].copy()
    sub["R_eff"] = sub["gamma_ps-1"] / sub["mag_character"]
    sub = sub[sub["R_eff"] > 0]

    print(f"\n[Spearman correlation, PAM vs ln(R), conditioned on |PAM| threshold, E > {E_min} meV]")
    print(f"  {'|PAM| >':>10s}  {'n':>8s}  {'rho':>8s}  {'p-value':>10s}")
    rows = []
    for thr in thresholds:
        m = sub["phon_AM_z_hbar"].abs() > thr
        if m.sum() < 5:
            print(f"  {thr:>10.2f}  {int(m.sum()):>8d}  {'--':>8s}  {'--':>10s}")
            continue
        rho, p = spearmanr(sub.loc[m, "phon_AM_z_hbar"], np.log(sub.loc[m, "R_eff"]))
        print(f"  {thr:>10.2f}  {int(m.sum()):>8d}  {rho:>8.3f}  {p:>10.3e}")
        rows.append({"threshold": thr, "n": int(m.sum()), "rho": rho, "p": p})
    return pd.DataFrame(rows)


def run_full_report(df, E_min=22.0, mag_threshold_for_Emax=0.9, pam_threshold=0.2,
                    out_dir=None):
    """Runs every hypothesis test and prints a single consolidated summary."""
    print("=" * 70)
    print("DEPENDENCY TEST REPORT")
    print("=" * 70)

    def _path(name):
        return os.path.join(out_dir, name) if out_dir else None

    power_res = test_power_law(df, E_min=E_min, out_png=_path("power_law.png"))
    E_mag_max = estimate_E_mag_max(df, mag_threshold=mag_threshold_for_Emax)
    detuning_res = test_detuning_law(df, E_mag_max, out_png=_path("detuning_law.png"))
    asym_res = test_sign_asymmetry(df, E_min=E_min, pam_threshold=pam_threshold)
    gauss_res = test_gaussian_selection_rule(df, E_min=E_min, out_png=_path("gaussian_selection.png"))
    spearman_df = test_spearman_conditioned(df, E_min=E_min)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if power_res is not None:
        print(f"Power law exponent (expect ~1):      {power_res.slope:+.3f} +/- {power_res.stderr:.3f}")
    if detuning_res is not None:
        print(f"Detuning exponent (expect ~-2):       {detuning_res.slope:+.3f} +/- {detuning_res.stderr:.3f}")
    if asym_res is not None:
        ratio = asym_res["median_neg"] / asym_res["median_pos"] if asym_res["median_pos"] > 0 else np.nan
        print(f"Neg/Pos PAM median R ratio:           {ratio:.2f}x  (p={asym_res['p_greater']:.2e})")
    if not spearman_df.empty:
        tightest = spearman_df.iloc[-1]
        print(f"Spearman rho at tightest |PAM| cut:    {tightest['rho']:+.3f} "
              f"(threshold={tightest['threshold']}, n={tightest['n']})")
    print("=" * 70)

    return {
        "power_law": power_res, "E_mag_max": E_mag_max, "detuning": detuning_res,
        "sign_asymmetry": asym_res, "gaussian": gauss_res, "spearman": spearman_df,
    }


def main():
    p = argparse.ArgumentParser(description="Lifetime ratio with/without SLC vs magnon character and PAM.")
    p.add_argument("--slc", required=True, help="hybrid_path_lifetimes.csv from the SLC-enabled run")
    p.add_argument("--bare", required=True, help="hybrid_path_lifetimes.csv from the --no_slc run")
    p.add_argument("--out", default=None, help="Output PNG for main plot")
    p.add_argument("--out_energy", default=None, help="Output PNG for Energy vs Magnon character plot")
    p.add_argument("--out_tau", default=None, help="Output PNG for high-energy Magnon/PAM vs Tau plot")
    p.add_argument("--out_mag_tau_pam", default=None, help="Output PNG for Mag character vs Tau colored by PAM")
    p.add_argument("--out_R", default=None, help="Output PNG for Effective Prefactor vs PAM plot")
    p.add_argument("--out_dir_report", default=None, help="Directory to save the dependency-test figures into")
    p.add_argument("--E_min", type=float, default=22.0, help="Energy threshold for the Tau/dependency tests (meV)")
    p.add_argument("--pam_threshold", type=float, default=0.2, help="|PAM| threshold for sign-asymmetry test")
    p.add_argument("--E_mag_threshold", type=float, default=0.9,
                   help="mag_character threshold used to estimate E_mag_max")
    p.add_argument("--match", choices=["energy", "index"], default="energy",
                   help="Mode pairing strategy (default: energy)")
    p.add_argument("--warn_dE", type=float, default=1.0,
                   help="Flag pairs whose |E_SLC - E_bare| exceeds this (meV)")
    p.add_argument("--csv", default=None, help="Optional path to dump the matched table")
    p.add_argument("--skip_plots", action="store_true", help="Only run the dependency tests, skip the original plots")
    args = p.parse_args()

    df = match_modes(load_path_csv(args.slc), load_path_csv(args.bare), mode=args.match)
    print(f"Matched {len(df):,} modes across {df['q_idx'].nunique():,} q-points "
          f"using --match {args.match}.")

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"Matched table written to '{args.csv}'")

    if not args.skip_plots:
        plot_mag_vs_energy(df, out_png=args.out_energy)
        plot_tau_high_energy(df, E_min=args.E_min, out_png=args.out_tau)
        plot_mag_vs_tau_pam(df, E_min=args.E_min, out_png=args.out_mag_tau_pam)
        plot_effective_prefactor(df, E_min=args.E_min, out_png=args.out_R)
        plot(df, out_png=args.out, warn_dE=args.warn_dE)

    run_full_report(df, E_min=args.E_min, mag_threshold_for_Emax=args.E_mag_threshold,
                    pam_threshold=args.pam_threshold, out_dir=args.out_dir_report)


if __name__ == "__main__":
    main()