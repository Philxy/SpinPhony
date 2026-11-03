#!/usr/bin/env python3
"""
Compare mesh sizes of a scan produced by scan_fccNi.sh (one folder per grid):

    python python/plot_grids.py Outputs/scans/fccNi_fullbz_20261102
        -> picks up Outputs/scans/fccNi_fullbz_20261102_grid20, _grid32, _grid40
    python python/plot_grids.py Outputs/scans/X_grid20 Outputs/scans/X_grid40     (explicit folders)

For every (grid, smearing, min_sigma) it reads <run>/params.txt + hybrid_path_lifetimes.csv
(--source path: path_lifetimes.csv) + job_*.out (channel-buffer fill) and writes into
<out_dir> (default: <base>_gridcompare next to the scan folders):

    grid_summary.csv                 one row per run (G_mp, buffer fill, mean gamma, ...)
    Gmp_vs_grid.png                  G_mp(N) per smearing, one panel per min_sigma; 2nd row: G_mp/G_mp(finest)
    tau_binned_by_grid_phonon.png    1/<gamma> per energy bin, phonon-like modes, lines = grids,
    tau_binned_by_grid_magnon.png    ... magnon-like modes; panels = (smearing x min_sigma)
    energy_vs_lifetime_by_grid.png   tau(E) scatter, one panel per grid, for one (smearing, min_sigma)
    tau_ratio_to_finest.png          binned tau(N) / tau(finest N), same (smearing, min_sigma)

G_mp and the loaders come from plot-hybrid-lifetimesfccNi.py (fcc primitive volume a^3/4).
G_mp is a BZ integral: only meaningful for mode=fullbz runs. Literature overlays are optional
(--lit_mag / --lit_phon, two-column files energy_meV, gamma_ps-1).
"""
import argparse
import glob
import importlib.util
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plot_scan as ps  # read_params, read_fill, binned_tau


def load_ni_lib():
    spec = importlib.util.spec_from_file_location(
        "nilib", os.path.join(HERE, "plot-hybrid-lifetimesfccNi.py"))
    lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lib)
    return lib


def expand_dirs(dirs):
    """A folder with */params.txt is used as is; otherwise <arg>_grid* is globbed."""
    out = []
    for a in dirs:
        a = a.rstrip("/")
        if glob.glob(os.path.join(a, "*", "params.txt")):
            out.append(a)
        else:
            out += sorted(d for d in glob.glob(a + "_grid*") if os.path.isdir(d))
    return list(dict.fromkeys(out))


def grid_of(run_dir, params, N):
    if "grid" in params:
        return int(params["grid"])
    m = re.search(r"_g(\d+)_sm", os.path.basename(run_dir))
    if m:
        return int(m.group(1))
    return int(round(N ** (1.0 / 3.0)))      # full BZ: N = grid^3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_dirs", nargs="+",
                    help="scan base name (expands to <base>_grid*) and/or individual grid folders")
    ap.add_argument("--source", choices=["hybrid", "path"], default="hybrid",
                    help="hybrid_path_lifetimes.csv or path_lifetimes.csv (bare modes)  [hybrid]")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--smearing", type=float, default=None,
                    help="smearing for the scatter / ratio plots (default: a value present for all grids)")
    ap.add_argument("--min_sigma", type=float, default=None,
                    help="min_sigma for the scatter / ratio plots (default: a value present for all grids)")
    ap.add_argument("--T", type=float, default=None, help="temperature for G_mp (K); default: from n_occ")
    ap.add_argument("--a", type=float, default=3.516822176, help="fcc lattice constant (A); V = a^3/4")
    ap.add_argument("--cell_volume", type=float, default=None, help="primitive cell volume (A^3)")
    ap.add_argument("--lit_mag", default=None, help="optional literature magnon rates (energy_meV, gamma_ps-1)")
    ap.add_argument("--lit_phon", default=None, help="optional literature phonon rates (energy_meV, gamma_ps-1)")
    args = ap.parse_args()

    lib = load_ni_lib()
    folders = expand_dirs(args.scan_dirs)
    if not folders:
        raise SystemExit("No scan folders with params.txt found.")
    print("Scan folders:", *folders, sep="\n  ")

    base = re.sub(r"_grid\d+$", "", folders[0])
    out = args.out_dir or base + "_gridcompare"
    os.makedirs(out, exist_ok=True)

    lit_mag = lib.load_literature(args.lit_mag)
    lit_ph = lib.load_literature(args.lit_phon)

    V_A3 = args.cell_volume if args.cell_volume is not None else args.a ** 3 / 4.0
    V_m3 = V_A3 * lib.ANG3_TO_M3
    csv_name = "hybrid_path_lifetimes.csv" if args.source == "hybrid" else "path_lifetimes.csv"

    # ------------------------------------------------------------------ load
    runs = []
    for folder in folders:
        for pfile in sorted(glob.glob(os.path.join(folder, "*", "params.txt"))):
            run_dir = os.path.dirname(pfile)
            csv = os.path.join(run_dir, csv_name)
            if not os.path.exists(csv):
                print(f"missing {csv_name}: {run_dir}")
                continue
            p = ps.read_params(pfile)
            df = lib.load_spinphony_csv(csv)
            df["tau_ps"] = pd.to_numeric(df["tau_ps"], errors="coerce")
            df["gamma_ps-1"] = pd.to_numeric(df["gamma_ps-1"], errors="coerce")
            N = df["q_idx"].nunique()
            grid = grid_of(run_dir, p, N)

            T = args.T if args.T is not None else lib.infer_temperature(df)
            if T is None:
                T = 300.0
            E = df["energy_meV"].to_numpy(float)
            g = df["gamma_ps-1"].to_numpy(float)
            pc = np.clip(df["phon_character"].to_numpy(float), 0, 1)
            ch = (df["mag_character"] - df["phon_character"]).to_numpy(float)
            keep = E > 0
            _, G = lib.compute_Gmp(E[keep], g[keep], pc[keep], T, N, V_m3)
            finite = np.isfinite(df["tau_ps"]) & (df["tau_ps"] > 0) & keep

            runs.append(dict(
                name=os.path.basename(run_dir), dir=run_dir, df=df, csv=csv, grid=grid,
                smearing=p["smearing"], min_sigma=p["min_sigma"], mode=p.get("mode", "?"),
                frac=p.get("anticipated_fraction", ""), T=T, N=N, G=G,
                mean_g_phon=np.nanmean(g[keep & (ch < -0.5)]) if (keep & (ch < -0.5)).any() else np.nan,
                mean_g_mag=np.nanmean(g[keep & (ch > 0.5)]) if (keep & (ch > 0.5)).any() else np.nan,
                frac_finite=finite.mean(), fill=ps.read_fill(run_dir),
                done=os.path.exists(os.path.join(run_dir, ".done")),
                mtime=os.path.getmtime(csv)))
            print(f"loaded {runs[-1]['name']} (grid {grid}, N={N}): G_mp = {G:.3e} W/m3/K")

    if not runs:
        raise SystemExit("No runs found.")

    # one result per (grid, smearing, min_sigma): newest wins
    newest = {}
    for r in runs:
        k = (r["grid"], r["smearing"], r["min_sigma"])
        if k not in newest or r["mtime"] > newest[k]["mtime"]:
            newest[k] = r
    runs = sorted(newest.values(), key=lambda r: (r["smearing"], r["min_sigma"], r["grid"]))

    for r in runs:
        if r["fill"] >= 100:
            print(f"WARNING: {r['name']}: channel buffer {r['fill']:.0f}% full -> channels were dropped "
                  f"(anticipated_fraction={r['frac']})")
        if not r["done"]:
            print(f"note: {r['name']} has no .done marker (job failed or still running?)")
    if any(r["mode"] != "fullbz" for r in runs):
        print("WARNING: some runs are not mode=fullbz; G_mp is only a BZ integral for full-BZ runs.")

    summary = pd.DataFrame([{k: r[k] for k in ("grid", "smearing", "min_sigma", "name", "mode", "T", "N", "G",
                                               "mean_g_phon", "mean_g_mag", "frac_finite", "fill", "frac", "done")}
                            for r in runs]).sort_values(["smearing", "min_sigma", "grid"])
    summary.to_csv(os.path.join(out, "grid_summary.csv"), index=False)
    print(f"\n -> {out}/grid_summary.csv")
    print(summary[["grid", "smearing", "min_sigma", "G", "fill"]].to_string(index=False))

    grids = sorted({r["grid"] for r in runs})
    sm_vals = sorted({r["smearing"] for r in runs})
    ms_vals = sorted({r["min_sigma"] for r in runs})
    gcol = {g: plt.get_cmap("viridis")(i / max(len(grids) - 1, 1) * 0.9) for i, g in enumerate(grids)}
    by = {(r["grid"], r["smearing"], r["min_sigma"]): r for r in runs}

    # --------------------------------------------------------- G_mp vs grid
    fig, axs = plt.subplots(2, len(ms_vals), figsize=(4.5 * len(ms_vals), 7), squeeze=False, sharex=True)
    cm = plt.get_cmap("plasma")
    for j, ms in enumerate(ms_vals):
        for i, sm in enumerate(sm_vals):
            d = summary[(summary.min_sigma == ms) & (summary.smearing == sm)].sort_values("grid")
            if d.empty:
                continue
            c = cm(i / max(len(sm_vals) - 1, 1) * 0.9)
            axs[0][j].plot(d.grid, d.G, "o-", color=c, label=f"smearing {sm:g}")
            ref = d.G.iloc[-1]
            if ref > 0:
                axs[1][j].plot(d.grid, d.G / ref, "o-", color=c)
        axs[0][j].set_yscale("log")
        axs[0][j].set_title(f"min_sigma = {ms:g} meV", fontsize=10)
        axs[1][j].axhline(1.0, color="k", ls="--", lw=1)
        axs[1][j].set_xlabel("mesh size N  (N x N x N)")
        for ax in (axs[0][j], axs[1][j]):
            ax.set_xticks(grids)
            ax.grid(True, which="both", ls="--", alpha=0.4)
    axs[0][0].set_ylabel(r"$G_{mp}$ (W m$^{-3}$ K$^{-1}$)")
    axs[1][0].set_ylabel(r"$G_{mp}$ / $G_{mp}$(finest grid)")
    axs[0][0].legend(fontsize=7)
    fig.suptitle("Ni: G_mp vs mesh size (hybrid)" if args.source == "hybrid" else "Ni: G_mp vs mesh size (bare)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "Gmp_vs_grid.png"), dpi=200)
    plt.close(fig)

    # ---------------------------------------------- binned tau, lines = grids
    allE = np.concatenate([r["df"]["energy_meV"].to_numpy(float) for r in runs])
    edges = np.geomspace(max(allE[allE > 0].min(), 0.5), allE.max(), 40)
    mid = np.sqrt(edges[1:] * edges[:-1])

    def binned(r, character):
        df = r["df"]
        E = df["energy_meV"].to_numpy(float)
        g = df["gamma_ps-1"].to_numpy(float)
        ch = (df["mag_character"] - df["phon_character"]).to_numpy(float)
        ok = (E > 0) & np.isfinite(g)
        sel = ok & (ch < -0.5 if character == "phonon" else ch > 0.5)
        return ps.binned_tau(E[sel], g[sel], edges)

    for character, lit in (("phonon", lit_ph), ("magnon", lit_mag)):
        fig, axs = plt.subplots(len(sm_vals), len(ms_vals), figsize=(3.6 * len(ms_vals), 2.9 * len(sm_vals)),
                                sharex=True, sharey=True, squeeze=False)
        for i, sm in enumerate(sm_vals):
            for j, ms in enumerate(ms_vals):
                ax = axs[i][j]
                for g in grids:
                    r = by.get((g, sm, ms))
                    if r is not None:
                        ax.plot(mid, binned(r, character), color=gcol[g], lw=1.3, label=f"{g}$^3$")
                if lit is not None:
                    ax.scatter(lit["energy_meV"], lit["tau_ps"], marker="s" if character == "phonon" else "^",
                               s=8, facecolors="none", edgecolors="k", lw=0.5)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_title(f"sm={sm:g}, ms={ms:g}", fontsize=8)
                ax.grid(True, which="both", ls="--", alpha=0.3)
                ax.tick_params(labelsize=7)
                if i == len(sm_vals) - 1:
                    ax.set_xlabel("E (meV)", fontsize=8)
                if j == 0:
                    ax.set_ylabel(r"$\tau$ (ps)", fontsize=8)
        axs[0][0].legend(fontsize=7, title="mesh", title_fontsize=7)
        fig.suptitle(rf"{character}-like modes: 1/$\langle\gamma\rangle$ per energy bin, lines = mesh size")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"tau_binned_by_grid_{character}.png"), dpi=200)
        plt.close(fig)

    # -------------------------------- chosen (smearing, min_sigma) comparison
    common = [(sm, ms) for sm in sm_vals for ms in ms_vals if all((g, sm, ms) in by for g in grids)]
    if args.smearing is not None and args.min_sigma is not None:
        pair = (args.smearing, args.min_sigma)
    elif common:
        pair = common[len(common) // 2]
    else:
        print("\nNo (smearing, min_sigma) exists for all grids; skipping scatter / ratio plots.")
        pair = None
    if pair is not None and not all((g, *pair) in by for g in grids):
        print(f"\n(smearing, min_sigma) = {pair} is not present for every grid; scatter uses those that exist.")

    if pair is not None:
        sm, ms = pair
        present = [g for g in grids if (g, sm, ms) in by]
        print(f"\nScatter / ratio plots at smearing = {sm:g}, min_sigma = {ms:g}  (override with --smearing/--min_sigma)")

        fig, axs = plt.subplots(1, len(present), figsize=(4.8 * len(present), 4.5), sharex=True, sharey=True,
                                squeeze=False)
        for ax, g in zip(axs[0], present):
            r = by[(g, sm, ms)]
            df = r["df"]
            v = np.isfinite(df["tau_ps"]) & (df["tau_ps"] > 0) & (df["energy_meV"] > 0)
            d = df[v]
            ch = d["mag_character"] - d["phon_character"]
            ax.scatter(d["energy_meV"], d["tau_ps"], c=ch, cmap="coolwarm", vmin=-1, vmax=1,
                       s=1.5, alpha=0.6, edgecolors="none", rasterized=True)
            if lit_mag is not None:
                ax.scatter(lit_mag["energy_meV"], lit_mag["tau_ps"], marker="^", s=18, facecolors="none",
                           edgecolors="k", lw=0.6)
            if lit_ph is not None:
                ax.scatter(lit_ph["energy_meV"], lit_ph["tau_ps"], marker="s", s=18, facecolors="none",
                           edgecolors="k", lw=0.6)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="both", ls="--", alpha=0.3)
            fill = f", buffer {r['fill']:.0f}%" if np.isfinite(r["fill"]) else ""
            ax.set_title(f"{g}$^3$: G={r['G']:.2e}{fill}", fontsize=9)
            ax.set_xlabel("E (meV)")
        axs[0][0].set_ylabel(r"$\tau$ (ps)")
        fig.suptitle(f"Lifetime vs energy (red = magnon, blue = phonon), smearing {sm:g}, min_sigma {ms:g}")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "energy_vs_lifetime_by_grid.png"), dpi=200)
        plt.close(fig)

        ref_g = max(present)
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
        for ax, character in zip(axs, ("phonon", "magnon")):
            ref = binned(by[(ref_g, sm, ms)], character)
            for g in present:
                if g == ref_g:
                    continue
                ax.plot(mid, binned(by[(g, sm, ms)], character) / ref, color=gcol[g], lw=1.4,
                        label=f"{g}$^3$ / {ref_g}$^3$")
            ax.axhline(1.0, color="k", ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Energy (meV)")
            ax.set_title(f"{character}-like: binned " + r"$\tau$ ratio to finest mesh", fontsize=10)
            ax.grid(True, which="both", ls="--", alpha=0.4)
            ax.legend(fontsize=8)
        axs[0].set_ylabel(r"$\tau(N)\,/\,\tau(N_{max})$")
        fig.suptitle(f"smearing {sm:g}, min_sigma {ms:g}")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "tau_ratio_to_finest.png"), dpi=200)
        plt.close(fig)

    print(f"\nPlots written to {out}/")


if __name__ == "__main__":
    main()
