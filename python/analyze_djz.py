import re
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, spearmanr


# ======================================================================
# Parsing
# ======================================================================
HEADER_RE = re.compile(
    r"\[channel_table\]\s+path_idx=(?P<path_idx>\d+)\s+band=(?P<band>\d+)\s+"
    r"E\s*=\s*(?P<E>[\d.]+)\s*meV"
)
RATE_RE = re.compile(
    r"1/tau\s*=\s*(?P<gamma>[\d.eE+-]+)\s*1/ps\s+"
    r"tau\s*=\s*(?P<tau>[\d.eE+-]+)\s*ps\s+"
    r"([\d,]+)\s*channels"
)
CHANNEL_RE = re.compile(
    r"^\s*(?P<proc>coal|split)\s+"
    r"(?P<rate>[+-][\d.eE+-]+)\s+"
    r"(?P<cum>[\d.]+)\s*\|\s*"
    r"\(\s*(?P<qx>[+-]?[\d.]+)\s*,\s*(?P<qy>[+-]?[\d.]+)\s*,\s*(?P<qz>[+-]?[\d.]+)\s*\)\s+"
    r"(?P<qnorm>[\d.]+)\s+"
    r"(?P<Ek>[\d.]+)\s+"
    r"(?P<bk>\d+)\s+"
    r"(?P<magk>[\d.]+)\s+"
    r"(?P<lzk>[+-][\d.]+)\s+"
    r"(?P<szk>[+-][\d.]+)\s*\|\s*"
    r"(?P<Eo>[\d.]+)\s+"
    r"(?P<bo>\d+)\s+"
    r"(?P<mago>[\d.]+)\s+"
    r"(?P<lzo>[+-][\d.]+)\s+"
    r"(?P<szo>[+-][\d.]+)\s*\|\s*"
    r"(?P<dJz>[+-][\d.]+)"
)
MEAN_DJZ_RE = re.compile(
    r"mean\s+(?P<mean_dJz>[+-][\d.]+)\s*hbar,\s*\|dJz\|\s*<\s*0\.15:\s*"
    r"(?P<frac_near0>[\d.]+)%\s*of\s*\|rate\|"
)


def parse_channel_table(path):
    """
    Parses a SpinPhony `channel_table` diagnostic log into two DataFrames:
    - channels: one row per individually-listed top-N channel (rate, dJz,
      both legs' character/AM, process type), tagged with its parent mode.
    - modes: one row per (path_idx, band) parent mode, with E, tau, gamma,
      and the rate-weighted mean dJz / near-zero fraction summary already
      computed by the code itself.
    """
    channel_rows = []
    mode_rows = []

    cur_mode = None
    with open(path) as f:
        for line in f:
            m = HEADER_RE.search(line)
            if m:
                cur_mode = {
                    "path_idx": int(m["path_idx"]),
                    "band": int(m["band"]),
                    "E": float(m["E"]),
                }
                continue

            if cur_mode is not None and "1/tau" in line and "tau" in line:
                m = RATE_RE.search(line)
                if m:
                    cur_mode["gamma"] = float(m["gamma"])
                    cur_mode["tau"] = float(m["tau"])

            m = MEAN_DJZ_RE.search(line)
            if m and cur_mode is not None:
                cur_mode["mean_dJz"] = float(m["mean_dJz"])
                cur_mode["frac_near0_dJz"] = float(m["frac_near0"])
                mode_rows.append(dict(cur_mode))
                continue

            m = CHANNEL_RE.match(line)
            if m and cur_mode is not None:
                channel_rows.append({
                    "path_idx": cur_mode["path_idx"],
                    "band": cur_mode["band"],
                    "E_parent": cur_mode["E"],
                    "proc": m["proc"],
                    "rate": float(m["rate"]),
                    "E_k": float(m["Ek"]), "b_k": int(m["bk"]),
                    "magch_k": float(m["magk"]),
                    "l_z_k": float(m["lzk"]), "S_z_k": float(m["szk"]),
                    "E_o": float(m["Eo"]), "b_o": int(m["bo"]),
                    "magch_o": float(m["mago"]),
                    "l_z_o": float(m["lzo"]), "S_z_o": float(m["szo"]),
                    "dJz": float(m["dJz"]),
                })

    channels = pd.DataFrame(channel_rows)
    modes = pd.DataFrame(mode_rows)
    if channels.empty:
        raise ValueError(f"No channel rows parsed from '{path}' -- check the format matches "
                         "the 'top N channels by |rate|' table.")
    channels["abs_rate"] = channels["rate"].abs()
    channels["abs_dJz"] = channels["dJz"].abs()
    return channels, modes


# ======================================================================
# Tests / plots
# ======================================================================
def test_rate_vs_djz(channels, out_png=None, n_bins=12):
    """
    Core test: does the per-channel rate fall off with |dJz|? This is the
    direct microscopic selection-rule signature -- channels that conserve
    J_z should dominate the sum if angular momentum matters at the vertex
    level. Uses log(|rate|) vs |dJz|, both a raw scatter and a binned
    median trend (medians are robust to the top-N truncation bias, since
    within any dJz bin the same "top 20" cutoff logic applies).
    """
    d = channels[(channels["abs_rate"] > 0) & np.isfinite(channels["dJz"])].copy()

    x = d["abs_dJz"].to_numpy()
    y = np.log10(d["abs_rate"].to_numpy())
    res = linregress(x, y)
    rho, p = spearmanr(x, y)

    print(f"\n[rate vs |dJz|, n={len(d)} channels]")
    print(f"  log10(rate) = {res.intercept:.3f} + {res.slope:.3f} * |dJz|   "
         f"(slope err {res.stderr:.3f}, R^2={res.rvalue**2:.4f})")
    print(f"  Spearman rho = {rho:.3f}, p = {p:.3e}")

    edges = np.linspace(0, d["abs_dJz"].quantile(0.98), n_bins + 1)
    d["bin"] = pd.cut(d["abs_dJz"], edges)
    binned = d.groupby("bin", observed=True)["abs_rate"].agg(["median", "count"])
    binned["mid"] = [iv.mid for iv in binned.index]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    colors = {"coal": "tab:blue", "split": "tab:orange"}
    for proc, sub in d.groupby("proc"):
        ax.scatter(sub["abs_dJz"], sub["abs_rate"], s=8, alpha=0.25,
                  color=colors.get(proc, "gray"), label=proc)
    ax.plot(binned["mid"], binned["median"], "k-o", ms=4, lw=1.5,
           label="binned median", zorder=5)
    ax.set_yscale("log")
    ax.set_xlabel(r"$|dJ_z|$ ($\hbar$)", fontsize=11)
    ax.set_ylabel(r"channel rate (ps$^{-1}$)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    if out_png:
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    return res, binned


def test_djz_by_process(channels):
    """
    Coalescence and splitting are different vertices (different momentum
    conservation, different combinatorics) -- tests whether they show the
    same |dJz| selectivity or not, via a plain two-sample comparison of
    the rate-weighted |dJz| distribution.
    """
    print("\n[|dJz| distribution by process, rate-weighted]")
    for proc, sub in channels.groupby("proc"):
        w = sub["abs_rate"]
        mean_dJz = np.average(sub["abs_dJz"], weights=w)
        print(f"  {proc:6s}: n={len(sub):5d}  rate-weighted mean |dJz| = {mean_dJz:.3f} hbar")


def test_djz_vs_magchar(channels, out_png=None, budget_floor=0.05):
    """
    Tests whether channels are more J_z-conserving specifically when the
    scattering partner is magnon-like -- i.e. whether the AM mismatch is
    concentrated in the phonon-phonon-like channels (which need no spin AM
    bookkeeping) versus magnon-involving ones.

    A raw |dJz| vs magch_k correlation is confounded by scale: when the
    partner is strongly magnon-like, |S_z_k| ~ magch_k is itself large, so
    the achievable |dJz| grows mechanically even if relative selectivity is
    unchanged. To separate a real selectivity trend from this budget
    artifact, also test dJz normalized by the total angular-momentum
    budget in play on both legs, dJz_rel = dJz / (|S_z_k|+|l_z_k|+|S_z_o|+|l_z_o|),
    floored to avoid blowing up on near-zero-AM channels.
    """
    d = channels[np.isfinite(channels["dJz"])].copy()

    budget = (d["S_z_k"].abs() + d["l_z_k"].abs() +
             d["S_z_o"].abs() + d["l_z_o"].abs()).clip(lower=budget_floor)
    d["dJz_rel"] = d["dJz"].abs() / budget

    rho_raw, p_raw = spearmanr(d["magch_k"], d["abs_dJz"])
    rho_rel, p_rel = spearmanr(d["magch_k"], d["dJz_rel"])

    print(f"\n[|dJz| vs partner magnon character, n={len(d)}]")
    print(f"  raw:        rho = {rho_raw:.3f}, p = {p_raw:.3e}")
    print(f"  normalized: rho = {rho_rel:.3f}, p = {p_rel:.3e}  "
         f"(dJz / AM budget on both legs -- controls for the scale artifact)")
    if abs(rho_rel) < 0.5 * abs(rho_raw):
        print("  => normalized correlation collapses relative to raw: the raw trend looks "
             "like mostly a budget/scale artifact, not a selectivity effect.")
    else:
        print("  => normalized correlation survives: this looks like a real selectivity trend, "
             "not just a scale artifact.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sc0 = axes[0].scatter(d["magch_k"], d["abs_dJz"], c=np.log10(d["abs_rate"]),
                          cmap="viridis", s=10, alpha=0.5)
    axes[0].set_xlabel("partner magnon character", fontsize=11)
    axes[0].set_ylabel(r"$|dJ_z|$ ($\hbar$)  [raw]", fontsize=11)

    sc1 = axes[1].scatter(d["magch_k"], d["dJz_rel"], c=np.log10(d["abs_rate"]),
                          cmap="viridis", s=10, alpha=0.5)
    axes[1].set_xlabel("partner magnon character", fontsize=11)
    axes[1].set_ylabel(r"$|dJ_z|$ / AM budget  [normalized]", fontsize=11)

    for ax in axes:
        ax.grid(True, ls="--", alpha=0.3)
    cbar = fig.colorbar(sc1, ax=axes, pad=0.02)
    cbar.set_label(r"$\log_{10}$(rate) (ps$^{-1}$)", fontsize=10)

    if out_png:
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    return {"raw": (rho_raw, p_raw), "normalized": (rho_rel, p_rel)}


def test_djz_vs_pam(channels, out_png=None, budget_floor=0.05):
    """
    Tests whether the AM mismatch dJz depends on the partner's phonon
    angular momentum l_z_k -- i.e. whether channels are more/less
    J_z-conserving specifically when the partner carries strong chiral
    phonon character, and whether that dependence has a preferred sign
    (matches the -1 vs +1 chirality-selectivity question from the PAM
    story) rather than being symmetric in l_z.

    Three angles, since a signed variable like l_z can show a trend that a
    naive |l_z| test would wash out:
      1. |dJz| vs |l_z_k|      -- does mismatch scale with the SIZE of the
                                   partner's chirality (any sign)?
      2. dJz (signed) vs l_z_k (signed) -- does mismatch have a preferred
                                   SIGN relationship with chirality?
      3. normalized: dJz / AM budget vs l_z_k -- controls for the same
                                   budget-scale artifact as the magchar test,
                                   since large |l_z_k| mechanically allows
                                   larger |dJz|.
    """
    d = channels[np.isfinite(channels["dJz"]) & np.isfinite(channels["l_z_k"])].copy()

    budget = (d["S_z_k"].abs() + d["l_z_k"].abs() +
             d["S_z_o"].abs() + d["l_z_o"].abs()).clip(lower=budget_floor)
    d["dJz_rel"] = d["dJz"] / budget

    rho_abs, p_abs = spearmanr(d["l_z_k"].abs(), d["abs_dJz"])
    rho_signed, p_signed = spearmanr(d["l_z_k"], d["dJz"])
    rho_rel, p_rel = spearmanr(d["l_z_k"], d["dJz_rel"])

    print(f"\n[dJz vs partner PAM l_z_k, n={len(d)}]")
    print(f"  |dJz| vs |l_z_k|:            rho = {rho_abs:.3f}, p = {p_abs:.3e}")
    print(f"  dJz (signed) vs l_z_k:       rho = {rho_signed:.3f}, p = {p_signed:.3e}")
    print(f"  dJz/budget vs l_z_k:         rho = {rho_rel:.3f}, p = {p_rel:.3e}  "
         f"(normalized -- controls for the AM-budget scale artifact)")
    if abs(rho_rel) < 0.5 * abs(rho_abs):
        print("  => normalized correlation collapses relative to raw: looks like mostly "
             "a budget/scale artifact, not real l_z selectivity.")
    else:
        print("  => normalized correlation survives: this looks like a real dependence "
             "of AM balance on partner chirality, not just a scale artifact.")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].scatter(d["l_z_k"].abs(), d["abs_dJz"], c=np.log10(d["abs_rate"]),
                    cmap="viridis", s=10, alpha=0.5)
    axes[0].set_xlabel(r"$|\ell_z^k|$ ($\hbar$)", fontsize=11)
    axes[0].set_ylabel(r"$|dJ_z|$ ($\hbar$)", fontsize=11)

    sc1 = axes[1].scatter(d["l_z_k"], d["dJz"], c=np.log10(d["abs_rate"]),
                          cmap="viridis", s=10, alpha=0.5)
    axes[1].axhline(0, color="gray", lw=0.6, ls="--")
    axes[1].axvline(0, color="gray", lw=0.6, ls="--")
    axes[1].set_xlabel(r"$\ell_z^k$ ($\hbar$)  [signed]", fontsize=11)
    axes[1].set_ylabel(r"$dJ_z$ ($\hbar$)  [signed]", fontsize=11)

    axes[2].scatter(d["l_z_k"], d["dJz_rel"], c=np.log10(d["abs_rate"]),
                    cmap="viridis", s=10, alpha=0.5)
    axes[2].axhline(0, color="gray", lw=0.6, ls="--")
    axes[2].axvline(0, color="gray", lw=0.6, ls="--")
    axes[2].set_xlabel(r"$\ell_z^k$ ($\hbar$)  [signed]", fontsize=11)
    axes[2].set_ylabel(r"$dJ_z$ / AM budget  [normalized]", fontsize=11)

    for ax in axes:
        ax.grid(True, ls="--", alpha=0.3)
    cbar = fig.colorbar(sc1, ax=axes, pad=0.02)
    cbar.set_label(r"$\log_{10}$(rate) (ps$^{-1}$)", fontsize=10)

    if out_png:
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    return {"abs": (rho_abs, p_abs), "signed": (rho_signed, p_signed), "normalized": (rho_rel, p_rel)}


def test_djz_histogram(channels, out_png=None, bins=60, magnon_thresh=0.05):
    """
    Rate-weighted histogram of signed dJz, split two ways:
      - by process (coal / split), since they are different vertices
      - by whether the channel involves a magnon leg at all
        (magch_k or magch_o > magnon_thresh) vs. a purely phonon-phonon-like
        channel (both legs below threshold)

    Motivation: a peak away from dJz=0 (e.g. near +/-1) need not reflect
    magnon physics at all. A channel with no appreciable magnon character on
    either partner leg has no spin-AM reservoir to balance against, so if
    one leg happens to be a near-unit chiral phonon (|l_z|~1) while the
    others are non-chiral, that channel sits near dJz=+/-1 by construction,
    regardless of any real magnon-phonon selection rule. Splitting the
    histogram this way tests whether such a peak is concentrated in the
    magnon-irrelevant population (contaminating the pooled statistics) or is
    a genuine magnon-involving effect. An asymmetric peak (e.g. only at +1,
    not -1) is itself informative, since S_z is one-signed in this FM
    (S_z in [-1,0]) while chiral phonon l_z is not -- so a lopsided result
    can still be diagnostic of the fixed-magnon-sign story even within the
    "no magnon involved" subset if there's an indirect correlation via which
    modes tend to be top-rate partners.
    """
    d = channels[np.isfinite(channels["dJz"])].copy()
    d["magnon_involved"] = (d["magch_k"] > magnon_thresh) | (d["magch_o"] > magnon_thresh)

    edges = np.linspace(-2.0, 2.0, bins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])

    def _weighted_hist(sub):
        h, _ = np.histogram(sub["dJz"], bins=edges, weights=sub["abs_rate"])
        return h / h.sum() if h.sum() > 0 else h

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

    for proc, sub in d.groupby("proc"):
        axes[0].plot(mids, _weighted_hist(sub), drawstyle="steps-mid", label=proc)
    axes[0].set_title("by process", fontsize=11)

    labels = {True: "magnon-involving", False: "phonon-only"}
    for flag, sub in d.groupby("magnon_involved"):
        axes[1].plot(mids, _weighted_hist(sub), drawstyle="steps-mid", label=labels[flag])
    axes[1].set_title(f"by magnon involvement (threshold {magnon_thresh})", fontsize=11)

    for ax in axes:
        ax.axvline(0, color="gray", lw=0.6, ls="--")
        ax.axvline(1, color="gray", lw=0.6, ls=":")
        ax.axvline(-1, color="gray", lw=0.6, ls=":")
        ax.set_xlabel(r"$dJ_z$ ($\hbar$)", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, ls="--", alpha=0.3)
    axes[0].set_ylabel("rate-weighted fraction", fontsize=11)

    print("\n[dJz histogram summary, rate-weighted]")
    for name, sub in [("all", d)] + list(d.groupby("proc")) + \
                     [(labels[k], v) for k, v in d.groupby("magnon_involved")]:
        w = sub["abs_rate"]
        if w.sum() == 0:
            continue
        near0 = w[sub["dJz"].abs() < 0.15].sum() / w.sum() * 100
        near_p1 = w[(sub["dJz"] - 1).abs() < 0.15].sum() / w.sum() * 100
        near_m1 = w[(sub["dJz"] + 1).abs() < 0.15].sum() / w.sum() * 100
        print(f"  {name:20s} n={len(sub):5d}  near 0: {near0:5.1f}%   "
             f"near +1: {near_p1:5.1f}%   near -1: {near_m1:5.1f}%")

    if out_png:
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    return d


def test_mode_level_djz(modes, out_png=None):
    """
    Mode-level (not per-channel) test: does the OVERALL rate-weighted mean
    |dJz| or near-zero-dJz fraction of a mode correlate with its lifetime
    or energy? This tests whether globally-AM-conserving modes are the
    fast-decaying ones -- the mode-averaged analogue of the channel-level
    test above.
    """
    d = modes[np.isfinite(modes["gamma"]) & (modes["gamma"] > 0)].copy()
    if d.empty or len(d) < 3:
        print("\n[mode-level test] too few parsed modes -- skipping.")
        return None

    for xcol, xlabel in [("mean_dJz", "rate-weighted mean dJz"),
                         ("frac_near0_dJz", "% rate with |dJz|<0.15"),
                         ("E", "mode energy (meV)")]:
        rho, p = spearmanr(d[xcol], np.log10(d["gamma"]))
        print(f"\n[mode-level: log(gamma) vs {xlabel}, n={len(d)}]")
        print(f"  Spearman rho = {rho:.3f}, p = {p:.3e}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(d["mean_dJz"], d["gamma"], c=d["E"], cmap="plasma", s=30)
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"rate-weighted mean $dJ_z$ ($\hbar$)", fontsize=11)
    axes[0].set_ylabel(r"$1/\tau$ (ps$^{-1}$)", fontsize=11)

    sc = axes[1].scatter(d["frac_near0_dJz"], d["gamma"], c=d["E"], cmap="plasma", s=30)
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"% rate with $|dJ_z|<0.15$", fontsize=11)
    axes[1].set_ylabel(r"$1/\tau$ (ps$^{-1}$)", fontsize=11)
    for ax in axes:
        ax.grid(True, which="both", ls="--", alpha=0.3)
    cbar = fig.colorbar(sc, ax=axes, pad=0.02)
    cbar.set_label("mode energy (meV)", fontsize=10)

    if out_png:
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    return d


def plot_band_djz(modes, out_png=None, cmap="jet", color_by="mean_dJz",
                  vlim=None):
    """
    Band-structure-style plot: path_idx along x (the k-path), mode energy
    along y, colored by the mode's rate-weighted mean dJz (or another
    per-mode column via `color_by`, e.g. 'frac_near0_dJz'). Each band index
    is drawn as its own connected line so the dJz pattern can be read off
    against the actual dispersion, the same way you'd read PAM or magnon
    character off a band structure.
    """
    d = modes[np.isfinite(modes["E"]) & np.isfinite(modes[color_by])].copy()
    if d.empty:
        print(f"[plot_band_djz] no modes with finite E/{color_by} -- skipping.")
        return None

    if vlim is None:
        vmax = d[color_by].abs().quantile(0.98)
        vlim = (-vmax, vmax) if d[color_by].min() < 0 else (0, vmax)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for band, sub in d.groupby("band"):
        sub = sub.sort_values("path_idx")
        ax.plot(sub["path_idx"], sub["E"], color="lightgray", lw=0.8, zorder=1)

    sc = ax.scatter(d["path_idx"], d["E"], c=d[color_by], cmap=cmap,
                    vmin=vlim[0], vmax=vlim[1], s=45, edgecolors="k",
                    linewidths=0.4, zorder=2)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    label = {"mean_dJz": r"rate-weighted mean $dJ_z$ ($\hbar$)",
             "frac_near0_dJz": r"% rate with $|dJ_z|<0.15$"}.get(color_by, color_by)
    cbar.set_label(label, fontsize=10)

    ax.set_xlabel("path index (k-path)", fontsize=11)
    ax.set_ylabel("Energy (meV)", fontsize=11)
    ax.grid(True, ls="--", alpha=0.3)

    if out_png:
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    return fig, ax


def main():
    p = argparse.ArgumentParser(description="Analyze dJz dependence of scattering channel rates.")
    p.add_argument("log", help="Path to a saved channel_table log file")
    p.add_argument("--out_dir", default=None, help="Directory to save figures into")
    args = p.parse_args()

    channels, modes = parse_channel_table(args.log)
    print(f"Parsed {len(channels):,} channels from {channels[['path_idx','band']].drop_duplicates().shape[0]} modes.")

    def _p(name):
        return f"{args.out_dir.rstrip('/')}/{name}" if args.out_dir else None

    test_rate_vs_djz(channels, out_png=_p("rate_vs_djz.png"))
    test_djz_by_process(channels)
    test_djz_vs_magchar(channels, out_png=_p("djz_vs_magchar.png"))
    test_djz_vs_pam(channels, out_png=_p("djz_vs_pam.png"))
    test_djz_histogram(channels, out_png=_p("djz_histogram.png"))
    test_mode_level_djz(modes, out_png=_p("mode_level_djz.png"))
    plot_band_djz(modes, out_png=_p("band_djz.png"))


if __name__ == "__main__":
    main()
