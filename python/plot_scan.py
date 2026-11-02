#!/usr/bin/env python3
"""
Plot a (smearing, min_sigma) scan produced by scan_submit.sh.

    python python/plot_scan.py Outputs/scans/bccFe_path_20261030

Reads <scan_dir>/*/params.txt + hybrid_path_lifetimes.csv (or path_lifetimes.csv, see
--source auto|hybrid|path; + job_*.out for the channel-buffer fill) and writes into
<scan_dir>/plots/ (plots_path/ for --source path) :

    scan_summary.csv            one row per run (G_mp, ratio to literature, ...)
    Gmp_heatmap.png             G_mp(smearing, min_sigma) and ratio to literature
    Gmp_convergence.png         G_mp vs min_sigma (per smearing) and vs smearing (per min_sigma)
    energy_vs_lifetime_grid.png tau(E) scatter, one panel per run, literature overlaid
    tau_binned_overlay.png      energy-binned <gamma>^-1, phonon-like / magnon-like, all runs
    per_run/<run>.png           the tau(E) scatter of each run on its own

G_mp, the lifetime loaders and the literature data come from
plot-hybrid-lifetimesbccFe.py, so numbers agree with what you plotted before.
G_mp is a BZ integral: it is only meaningful for runs with mode=fullbz.
"""
import argparse
import glob
import importlib.util
import io
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def load_lib():
    spec = importlib.util.spec_from_file_location(
        "plotlib", os.path.join(HERE, "plot-hybrid-lifetimesbccFe.py"))
    lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lib)
    return lib


def read_params(path):
    p = {}
    with open(path) as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                p[k] = v
    for k in ("smearing", "min_sigma"):
        p[k] = float(p[k])
    return p


def read_fill(run_dir):
    """found/max channel percentage printed by SpinPhony (>=100 means dropped channels)."""
    pat = re.compile(r"found_channels / max_channels : ([0-9.]+)%")
    val = np.nan
    for f in sorted(glob.glob(os.path.join(run_dir, "job_*.out"))):
        with open(f, errors="ignore") as fh:
            for line in fh:
                m = pat.search(line)
                if m:
                    val = float(m.group(1))
    return val


def binned_tau(E, gamma, edges):
    """1/<gamma> per energy bin (right estimator for a sum over modes)."""
    idx = np.digitize(E, edges) - 1
    out = np.full(len(edges) - 1, np.nan)
    for b in range(len(edges) - 1):
        s = idx == b
        if s.sum() >= 3:
            g = gamma[s].mean()
            if g > 0:
                out[b] = 1.0 / g
    return out


def heatmap(ax, table, title, fmt, log=True, cmap="viridis"):
    data = table.to_numpy(float)
    shown = np.log10(np.where(data > 0, data, np.nan)) if log else data
    im = ax.imshow(shown, origin="lower", cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels([f"{c:g}" for c in table.columns])
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels([f"{i:g}" for i in table.index])
    ax.set_xlabel("min_sigma (meV)")
    ax.set_ylabel("smearing (scale factor)")
    ax.set_title(title, fontsize=11)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        fontsize=7, color="white")
    plt.colorbar(im, ax=ax, label="log10" if log else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_dir")
    ap.add_argument("--source", choices=["auto", "hybrid", "path"], default="auto",
                    help="which lifetime file to plot: 'hybrid' = hybrid_path_lifetimes.csv, "
                         "'path' = path_lifetimes.csv (bare magnons/phonons), 'auto' = hybrid "
                         "if present in a run, else path_lifetimes.csv  [auto]")
    ap.add_argument("--out_dir", default=None,
                    help="default: <scan_dir>/plots (<scan_dir>/plots_path for --source path)")
    ap.add_argument("--T", type=float, default=None,
                    help="temperature for G_mp (K); default: inferred from n_occ of each run")
    ap.add_argument("--a", type=float, default=2.8665, help="lattice constant (A)")
    ap.add_argument("--cell_volume", type=float, default=None, help="cell volume (A^3), overrides --a")
    ap.add_argument("--no_per_run", action="store_true", help="skip the per-run PNGs")
    args = ap.parse_args()

    lib = load_lib()
    out = args.out_dir or os.path.join(args.scan_dir, "plots_path" if args.source == "path" else "plots")
    os.makedirs(os.path.join(out, "per_run"), exist_ok=True)

    lit_mag = pd.read_csv(io.StringIO(lib.literature_scattering_rates), names=["E", "gamma"])
    lit_ph = pd.read_csv(io.StringIO(lib.literature_phonon_scatt_rates_per_ps), names=["E", "gamma"])
    E_lit = lit_ph["E"].to_numpy(float)
    g_lit = lit_ph["gamma"].to_numpy(float)

    V_A3 = args.cell_volume if args.cell_volume is not None else 0.5 * args.a ** 3
    V_m3 = V_A3 * lib.ANG3_TO_M3

    # ---------------------------------------------------------------- load runs
    runs = []
    for pfile in sorted(glob.glob(os.path.join(args.scan_dir, "*", "params.txt"))):
        run_dir = os.path.dirname(pfile)
        candidates = {"hybrid": ["hybrid_path_lifetimes.csv"],
                      "path": ["path_lifetimes.csv"],
                      "auto": ["hybrid_path_lifetimes.csv", "path_lifetimes.csv"]}[args.source]
        csv = next((os.path.join(run_dir, c) for c in candidates
                    if os.path.exists(os.path.join(run_dir, c))), None)
        if csv is None:
            print(f"missing {' / '.join(candidates)}: {os.path.basename(run_dir)}")
            continue
        source = "hybrid" if os.path.basename(csv).startswith("hybrid") else "path"
        p = read_params(pfile)
        df = lib.load_spinphony_csv(csv)
        df["tau_ps"] = pd.to_numeric(df["tau_ps"], errors="coerce")
        df["gamma_ps-1"] = pd.to_numeric(df["gamma_ps-1"], errors="coerce")

        T = args.T if args.T is not None else lib.infer_temperature(df)
        if T is None:      # path_lifetimes.csv has no n_occ column
            T = 300.0
            if not any(r.get("noT_warned") for r in runs):
                print("note: no n_occ column in this file - using T = 300 K for G_mp (override with --T)")
        E = df["energy_meV"].to_numpy(float)
        pc = np.clip(df["phon_character"].to_numpy(float), 0, 1)
        g = df["gamma_ps-1"].to_numpy(float)
        keep = E > 0
        N = df["q_idx"].nunique()
        G_sp_cell, G_sp = lib.compute_Gmp(E[keep], g[keep], pc[keep], T, N, V_m3)
        g_ar, _ = lib.literature_gamma_of_E(E_lit, g_lit, E[keep], mean="arith")
        _, G_lit = lib.compute_Gmp(E[keep], g_ar, pc[keep], T, N, V_m3)

        finite = np.isfinite(df["tau_ps"]) & (df["tau_ps"] > 0) & (E > 0)
        runs.append(dict(
            name=os.path.basename(run_dir), dir=run_dir, df=df, params=p, csv=csv, source=source,
            noT_warned=args.T is None and "n_occ" not in df.columns,
            smearing=p["smearing"], min_sigma=p["min_sigma"], mode=p.get("mode", "?"),
            T=T, N=N, G_sp=G_sp, G_lit=G_lit, ratio=G_sp / G_lit if G_lit > 0 else np.nan,
            frac_finite=finite.mean(), fill=read_fill(run_dir),
            done=os.path.exists(os.path.join(run_dir, ".done"))))
        print(f"loaded {runs[-1]['name']}: G_mp = {G_sp:.3e} W/m3/K  (T={T:.0f} K, N={N})")

    if not runs:
        raise SystemExit(f"No runs with the requested lifetime file found (--source {args.source}).")

    # The same (smearing, min_sigma) can appear in several directories ("ms1" vs
    # "ms1.0", reruns, ...). Keep the most recently written result of each pair.
    newest = {}
    for r in runs:
        key = (r["smearing"], r["min_sigma"])
        r["mtime"] = os.path.getmtime(r["csv"])
        if key not in newest or r["mtime"] > newest[key]["mtime"]:
            newest[key] = r
    dropped = [r["name"] for r in runs if newest[(r["smearing"], r["min_sigma"])] is not r]
    if dropped:
        print(f"\nDuplicate (smearing, min_sigma) pairs - ignoring older directories: {dropped}\n")
    runs = sorted(newest.values(), key=lambda r: (r["smearing"], r["min_sigma"]))

    sources = {r["source"] for r in runs}
    if len(sources) > 1:
        print("\nWARNING: --source auto mixed hybrid and bare (path_lifetimes.csv) runs; "
              "use --source hybrid or --source path to compare like with like.\n")
    if sources == {"path"}:
        print("\nnote: path_lifetimes.csv is the BARE magnon/phonon calculation. Its scan kernel only "
              "receives `smearing` (min_sigma is not passed on), so the min_sigma axis is expected to be flat.\n")
    modes = {r["mode"] for r in runs}
    if "path" in modes:
        print("\nWARNING: some runs have mode=path. G_mp sums over the path points only and is "
              "not a BZ integral - compare G_mp across runs only for mode=fullbz.\n")
    for r in runs:
        if r["fill"] >= 100:
            print(f"WARNING: {r['name']}: channel buffer {r['fill']:.0f}% full -> channels were dropped")
        if not r["done"]:
            print(f"note: {r['name']} has no .done marker (job failed or still running?)")

    summary = pd.DataFrame([{k: r[k] for k in ("name", "source", "mode", "smearing", "min_sigma", "T", "N",
                                               "G_sp", "G_lit", "ratio", "frac_finite", "fill", "done")}
                            for r in runs]).sort_values(["smearing", "min_sigma"])
    summary.to_csv(os.path.join(out, "scan_summary.csv"), index=False)
    print(f"\n -> {out}/scan_summary.csv")

    sm_vals = sorted({r["smearing"] for r in runs})
    ms_vals = sorted({r["min_sigma"] for r in runs})
    lbl = " (mode=path: NOT BZ-integrated)" if "path" in modes else ""
    src_lbl = {"hybrid": "hybrid", "path": "bare path_lifetimes"}.get(next(iter(sources)), "mixed sources")
    lbl += f" [{src_lbl}]"

    # ------------------------------------------------------------ G_mp heatmaps
    piv = lambda col: summary.pivot(index="smearing", columns="min_sigma", values=col) \
        .reindex(index=sm_vals, columns=ms_vals)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    heatmap(axs[0], piv("G_sp"), r"$G_{mp}$ SpinPhony [W m$^{-3}$ K$^{-1}$]" + lbl, ".2e")
    heatmap(axs[1], piv("ratio"), r"$G_{mp}$ SpinPhony / literature", ".2f", cmap="coolwarm")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "Gmp_heatmap.png"), dpi=200)
    plt.close(fig)

    # ------------------------------------------------------- G_mp convergence
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    cm = plt.get_cmap("viridis")
    for i, sm in enumerate(sm_vals):
        d = summary[summary.smearing == sm].sort_values("min_sigma")
        axs[0].plot(d.min_sigma, d.G_sp, "o-", color=cm(i / max(len(sm_vals) - 1, 1)),
                    label=f"smearing {sm:g}")
    for j, ms in enumerate(ms_vals):
        d = summary[summary.min_sigma == ms].sort_values("smearing")
        axs[1].plot(d.smearing, d.G_sp, "o-", color=cm(j / max(len(ms_vals) - 1, 1)),
                    label=f"min_sigma {ms:g}")
    G_lit_ref = summary.G_lit.median()
    for ax, xl in zip(axs, ["min_sigma (meV)", "smearing (scale factor)"]):
        ax.axhline(G_lit_ref, color="k", ls="--", lw=1, label="literature (arith.)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel(r"$G_{mp}$ (W m$^{-3}$ K$^{-1}$)")
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.legend(fontsize=8)
    axs[0].set_title("Convergence in min_sigma" + lbl, fontsize=11)
    axs[1].set_title("Convergence in smearing" + lbl, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "Gmp_convergence.png"), dpi=200)
    plt.close(fig)

    # ---------------------------------------- energy vs lifetime, all runs grid
    def scatter(ax, r, small=True):
        df = r["df"]
        v = np.isfinite(df["tau_ps"]) & (df["tau_ps"] > 0) & (df["energy_meV"] > 0)
        d = df[v]
        ch = d["mag_character"] - d["phon_character"]
        ax.scatter(d["energy_meV"], d["tau_ps"], c=ch, cmap="coolwarm", vmin=-1, vmax=1,
                   s=1.5 if small else 5, alpha=0.6, edgecolors="none", rasterized=True)
        ax.scatter(lit_mag["E"], 1 / lit_mag["gamma"], marker="^", s=8 if small else 30,
                   facecolors="none", edgecolors="k", linewidths=0.5)
        ax.scatter(lit_ph["E"], 1 / lit_ph["gamma"], marker="s", s=8 if small else 30,
                   facecolors="none", edgecolors="k", linewidths=0.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.3)

    nr, nc = len(sm_vals), len(ms_vals)
    fig, axs = plt.subplots(nr, nc, figsize=(3.2 * nc, 2.8 * nr), sharex=True, sharey=True,
                            squeeze=False)
    by = {(r["smearing"], r["min_sigma"]): r for r in runs}
    for i, sm in enumerate(sm_vals):
        for j, ms in enumerate(ms_vals):
            ax = axs[i][j]
            r = by.get((sm, ms))
            if r is None:
                ax.axis("off")
                continue
            scatter(ax, r)
            ax.set_title(f"sm={sm:g}, ms={ms:g}\nG={r['G_sp']:.2e}", fontsize=8)
            if i == nr - 1:
                ax.set_xlabel("E (meV)", fontsize=8)
            if j == 0:
                ax.set_ylabel(r"$\tau$ (ps)", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle("Lifetime vs energy (red = magnon, blue = phonon; black = literature) " + lbl,
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "energy_vs_lifetime_grid.png"), dpi=200)
    plt.close(fig)

    # --------------------------------------------- binned tau overlay (lines)
    allE = np.concatenate([r["df"]["energy_meV"].to_numpy(float) for r in runs])
    edges = np.geomspace(max(allE[allE > 0].min(), 0.5), allE.max(), 40)
    mid = np.sqrt(edges[1:] * edges[:-1])
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    ls_cycle = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
    cmap_ms = plt.get_cmap("plasma")
    for r in runs:
        df = r["df"]
        E = df["energy_meV"].to_numpy(float)
        g = df["gamma_ps-1"].to_numpy(float)
        ok = (E > 0) & np.isfinite(g)
        ch = (df["mag_character"] - df["phon_character"]).to_numpy(float)
        col = cmap_ms(ms_vals.index(r["min_sigma"]) / max(len(ms_vals) - 1, 1) * 0.9)
        ls = ls_cycle[sm_vals.index(r["smearing"]) % len(ls_cycle)]
        for ax, sel in zip(axs, [ok & (ch < -0.5), ok & (ch > 0.5)]):
            ax.plot(mid, binned_tau(E[sel], g[sel], edges), color=col, ls=ls, lw=1.2)
    axs[0].scatter(lit_ph["E"], 1 / lit_ph["gamma"], marker="s", s=12, facecolors="none",
                   edgecolors="k", lw=0.5)
    axs[1].scatter(lit_mag["E"], 1 / lit_mag["gamma"], marker="^", s=12, facecolors="none",
                   edgecolors="k", lw=0.5)
    for ax, t in zip(axs, ["phonon-like modes", "magnon-like modes"]):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Energy (meV)")
        ax.set_title(t + r" : 1/$\langle\gamma\rangle$ per energy bin", fontsize=11)
        ax.grid(True, which="both", ls="--", alpha=0.4)
    axs[0].set_ylabel(r"$\tau$ (ps)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=cmap_ms(k / max(len(ms_vals) - 1, 1) * 0.9), label=f"min_sigma {m:g}")
               for k, m in enumerate(ms_vals)]
    handles += [Line2D([], [], color="gray", ls=ls_cycle[k % len(ls_cycle)], label=f"smearing {s:g}")
                for k, s in enumerate(sm_vals)]
    axs[1].legend(handles=handles, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "tau_binned_overlay.png"), dpi=200)
    plt.close(fig)

    # --------------------------------------------------------------- per run
    if not args.no_per_run:
        for r in runs:
            fig, ax = plt.subplots(figsize=(7, 5))
            scatter(ax, r, small=False)
            ax.set_xlabel("Energy (meV)")
            ax.set_ylabel(r"$\tau$ (ps)")
            ax.set_title(f"{r['name']}   G_mp = {r['G_sp']:.2e} W/m3/K", fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(out, "per_run", r["name"] + ".png"), dpi=200)
            plt.close(fig)

    print(f"\nPlots written to {out}/")


if __name__ == "__main__":
    main()
