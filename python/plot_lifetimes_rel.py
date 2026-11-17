"""
Scattering rate vs energy of magnons and phonons in bcc Fe and fcc Ni, split into the magnon-number CONSERVING (exchange-striction,
gamma_exch_ps-1) and NON-CONSERVING (relativistic, gamma_rel_ps-1) channels. The three-phonon part (gamma_phph_ps-1) is never plotted.
One figure per layout variant and per broadening (default 0.2 and 1 meV), PNG + PDF; every layout is written twice: linear energy axis and log-log (suffix _loglog). Outliers (rates below --gmin = 1e-6 ps^-1
and, optionally, the lowest --plow percent of each set) are cut. Layouts:

    python python/plot_lifetimes_rel.py [--conv Outputs/convergence_rel/conv_rel_20261112] [--N 40] [--T 300]
                                        [--sigmas 0.2 1 | all] [--out_dir Outputs/paper/lifetimes] [--no_pdf] [--gmin 1e-6] [--plow 0] [--alpha 0.08]

Layout variants (file names lifetimes_<variant>_N<N>_s<sigma>.png):
  A_overlay_mat_x_part   2x2   rows Fe / Ni, columns magnons / phonons; conserving and non-conserving overlaid in one panel
  B_overlay_part_x_mat   2x2   rows magnons / phonons, columns Fe / Ni; conserving and non-conserving overlaid
  C_split_mat_x_4        2x4   rows Fe / Ni; columns magnon cons. | magnon non-cons. | phonon cons. | phonon non-cons.
  D_split_4_x_mat        4x2   rows magnon cons. / magnon non-cons. / phonon cons. / phonon non-cons.; columns Fe / Ni
  F_magphon_together     2x2   magnons AND phonons in the same axes: rows conserving / non-conserving, columns Fe / Ni; the log-log
                               version carries reference lines gamma ~ E and gamma ~ E^2 (through the conserving-magnon median at --Eref)
  E_split_part_x_chan    2x2 per material (one figure per material): rows magnons / phonons, columns conserving / non-conserving
All panels share the log y axis; x is shared per particle type. With --guides the conserving-magnon panels carry guide lines gamma ~ E (solid, low energies, window --e1, default 100-200 meV) and gamma ~ E^2 (dashed, high energies,
windows --e2_Fe 400+ and --e2_Ni 300+ meV), each drawn only inside its window with the amplitude fitted there; the free log-log slope of the data in each window is printed.
Points are drawn with low alpha so the density shows. A channel with no non-zero rate (e.g. relativistic part of an isotropic
coupling) gets an empty panel marked "no non-zero rates". If a file has no gamma_exch column (runs without --relativistic) the
conserving rate is gamma_ps-1 - gamma_phph_ps-1 and there is no non-conserving part.
"""
import argparse
import glob
import os

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAME = {"bccFe": "bcc Fe", "fccNi": "fcc Ni"}
MATS = ("bccFe", "fccNi")
PART = {"magnon": ("magnons", "^"), "phonon": ("phonons", "o")}
CHAN = {"cons": ("conserving", "#1f77b4"), "rel": ("non-conserving (relativistic)", "#d62728")}
E_MIN = 5


def read_params(path):
    p = {}
    if os.path.exists(path):
        for line in open(path):
            if "=" in line:
                k, v = line.strip().split("=", 1)
                p[k] = v
    return p


def find_runs(roots, N, T):
    """roots: one scan folder or a list; an entry 'folder:material' only takes that material from the folder (e.g. old Fe + new Ni runs)"""
    runs = {}
    items = []
    for r in ([roots] if isinstance(roots, str) else roots):
        r, _, only = r.partition(":")
        items += [(pf, only) for pf in glob.glob(os.path.join(r, "*", "params.txt"))]
    for pf, only in sorted(items):
        d = os.path.dirname(pf); p = read_params(pf)
        if not os.path.exists(os.path.join(d, "path_lifetimes.csv")) or p.get("material") not in MATS or (only and p["material"] != only):
            continue
        if int(float(p["N"])) != N or abs(float(p.get("T", p.get("T_mag", 300))) - T) > 1e-6:
            continue
        runs[(p["material"], float(p["smearing"]))] = d
    return runs


def load(d):
    """DataFrame with columns particle, E, cons, rel (rel = None if the file has no relativistic column)."""
    df = pd.read_csv(os.path.join(d, "path_lifetimes.csv"), comment="#")
    num = lambda c: pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy(float) if c in df.columns else None
    ex, rl, pp, tot = num("gamma_exch_ps-1"), num("gamma_rel_ps-1"), num("gamma_phph_ps-1"), num("gamma_ps-1")
    cons = ex if ex is not None else tot - (pp if pp is not None else 0.0)
    return pd.DataFrame({"particle": df.particle, "E": df.energy_meV.to_numpy(float), "cons": cons,
                         "rel": rl if rl is not None else np.zeros(len(df))}), rl is not None


def style():
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 9, "legend.fontsize": 7,
                         "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
                         "savefig.dpi": 400})


LOGX = dict(on=False)      # log-log version of the plots
CUT = dict(gmin=1e-6, plow=0.0, gmin_rel=1e-10)     # absolute floor (ps^-1) and lower percentile trimmed per particle type and channel


def keep(df, part, ch):
    """mask of the 'reasonable' points: positive rate above the absolute floor and above the plow-th percentile of the rest
    (removes the numerically-zero outliers, rates down to 1e-30 ps^-1)"""
    m = (df.particle == part) & (df.E > E_MIN) & (df[ch] > (CUT["gmin_rel"] if ch == "rel" else CUT["gmin"])) & np.isfinite(df[ch])
    if m.any() and CUT["plow"] > 0:
        m &= df[ch] >= np.percentile(df[ch][m], CUT["plow"])
    return m


GUIDE = dict(on=False, e1=(100.0, 200.0), e2={"bccFe": (400.0, np.inf), "fccNi": (300.0, np.inf)})   # energy windows (meV) of the E^1 / E^2 guide lines


def guide_win(df, n):
    return GUIDE["e1"] if n == 1 else GUIDE["e2"][df.attrs.get("mat", "bccFe")]


def guide_fit(df, n, lo, hi):
    """amplitude A of gamma = A E^n (median of gamma/E^n over the conserving magnons in [lo, hi] meV) and the free log-log slope there"""
    m = keep(df, "magnon", "cons") & (df.E >= lo) & (df.E <= hi)
    if m.sum() < 5:
        return np.nan, np.nan
    E, g = df.E[m].to_numpy(), df["cons"][m].to_numpy()
    return float(np.median(g / E ** n)), float(np.polyfit(np.log(E), np.log(g), 1)[0])


def draw(ax, df, part, chans, alpha, E0, ylim):
    """scatter one particle type, one or two channels, into ax; returns True if anything was drawn"""
    drawn = False
    for ch in chans:
        lab, col = CHAN[ch]
        m = keep(df, part, ch)
        if not m.any():
            continue
        drawn = True
        ax.scatter(df.E[m], df[ch][m], marker=PART[part][1], s=5, color=col, alpha=alpha, linewidths=0, rasterized=True)
        ax.scatter([], [], marker=PART[part][1], s=24, color=col, label=lab)
        if GUIDE["on"] and part == "magnon" and ch == "cons":
            for n, (lo, hi), ls in ((1, guide_win(df, 1), "-"), (2, guide_win(df, 2), "--")):
                A, slope = guide_fit(df, n, lo, hi)
                if np.isfinite(A):
                    hi = min(hi, float(df.E[keep(df, "magnon", "cons")].max()))
                    E = np.linspace(lo, hi, 50)
                    ax.plot(E, A * E ** n, "k", ls=ls, lw=1.2, zorder=5, label=rf"$\propto E^{n}$")
    if not drawn:
        ax.text(0.5, 0.5, "no non-zero rates", ha="center", va="center", transform=ax.transAxes, color="0.4")
    ax.set_yscale("log"); ax.set_ylim(*ylim)
    if LOGX["on"]:
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.LogFormatterSciNotation(minor_thresholds=(2, 0.4)))   # label 2, 5 on narrow decades
        ax.tick_params(axis="x", which="minor", labelsize=6)
    ax.grid(True, which="major", ls=":", alpha=0.4)
    return drawn


def ylim_of(data):
    lo, hi = [], []
    for df in data.values():
        for ch in ("cons", "rel"):
            for part in ("magnon", "phonon"):
                v = df[ch][keep(df, part, ch)]
                if len(v):
                    lo.append(v.min()); hi.append(v.max())
    return (min(lo) / 3, max(hi) * 4) if lo else (1e-6, 1e3)


def finish(fig, out, pdf):
    fig.tight_layout()
    fig.savefig(out + ".png", bbox_inches="tight")
    if pdf:
        fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(" ->", out + ".png")


def label(ax, text, i):
    ax.text(0.03, 0.95, f"({'abcdefgh'[i]}) {text}", transform=ax.transAxes, va="top", fontsize=8)


def make_all(data, mats, sigma, a):
    yl = ylim_of(data)
    tag = f"N{a.N}_s{sigma:g}" + ("_loglog" if LOGX["on"] else "")
    ylabel = r"Scattering rate $\gamma$ (ps$^{-1}$)"
    # ---- A / B: overlay conserving + non-conserving
    for variant, transpose in (("A_overlay_mat_x_part", False), ("B_overlay_part_x_mat", True)):
        nr, nc = (2, 2)
        fig, axs = plt.subplots(nr, nc, figsize=(10/2.52, 8/2.52), sharey=True, sharex="col" if not transpose else "row")
        i = 0
        for r in range(2):
            for c in range(2):
                m, part = (mats[r], ("magnon", "phonon")[c]) if not transpose else (mats[c], ("magnon", "phonon")[r])
                ax = axs[r, c]
                draw(ax, data[m], part, ("cons", "rel"), a.alpha, a.E0, yl)
                label(ax, f"{NAME[m]} {PART[part][0]}", i); i += 1
                if (r, c) == (0, 0):
                    ax.legend(frameon=False, loc="lower right" if part == "magnon" else "upper right")
                if c == 0:
                    ax.set_ylabel(ylabel)
                if r == 1:
                    ax.set_xlabel("energy (meV)")
        finish(fig, os.path.join(a.out_dir, f"lifetimes_{variant}_{tag}"), a.pdf)
    # ---- C: 2x4
    cols = [("magnon", "cons"), ("magnon", "rel"), ("phonon", "cons"), ("phonon", "rel")]
    fig, axs = plt.subplots(2, 4, figsize=(10/2.52, 8/2.52), sharey=True)
    for r, m in enumerate(mats):
        for c, (part, ch) in enumerate(cols):
            ax = axs[r, c]; draw(ax, data[m], part, (ch,), a.alpha, a.E0, yl)
            ax.set_title(f"{PART[part][0]}, {CHAN[ch][0].split(' ')[0]}", fontsize=7, pad=2)
            if r == 1:
                ax.set_xlabel("energy (meV)")
            if c == 0:
                ax.set_ylabel(NAME[m] + "\n" + ylabel)
    finish(fig, os.path.join(a.out_dir, f"lifetimes_C_split_mat_x_4_{tag}"), a.pdf)
    # ---- D: 4x2
    fig, axs = plt.subplots(4, 2, figsize=(10/2.52, 8/2.52), sharey=True)
    for c, m in enumerate(mats):
        for r, (part, ch) in enumerate(cols):
            ax = axs[r, c]; draw(ax, data[m], part, (ch,), a.alpha, a.E0, yl)
            label(ax, f"{NAME[m]} {PART[part][0]}, {CHAN[ch][0].split(' ')[0]}", r + 4 * c)
            if r == 3:
                ax.set_xlabel("energy (meV)")
            if c == 0:
                ax.set_ylabel(ylabel)
    finish(fig, os.path.join(a.out_dir, f"lifetimes_D_split_4_x_mat_{tag}"), a.pdf)
    # ---- E: one figure per material, rows magnons / phonons, columns conserving / non-conserving
    for m in mats:
        fig, axs = plt.subplots(2, 2, figsize=(10/2.52, 8/2.52), sharey=True, sharex="row")
        for r, part in enumerate(("magnon", "phonon")):
            for c, ch in enumerate(("cons", "rel")):
                ax = axs[r, c]; draw(ax, data[m], part, (ch,), a.alpha, a.E0, yl)
                label(ax, f"{PART[part][0]}, {CHAN[ch][0].split(' ')[0]}", 2 * r + c)
                if r == 1:
                    ax.set_xlabel("energy (meV)")
                if c == 0:
                    ax.set_ylabel(ylabel)
        fig.suptitle(NAME[m], fontsize=9)
        finish(fig, os.path.join(a.out_dir, f"lifetimes_E_split_part_x_chan_{m}_{tag}"), a.pdf)


PCOL = {"magnon": "#d95f02", "phonon": "#1f77b4"}      # magnons: dark tab-orange, phonons: tab-blue
RCOL = {"magnon": "#7b3294", "phonon": "#1b9e55"}      # non-conserving (relativistic) rates: magnons purple, phonons green


def make_F(data, mats, sigma, a):
    """F: magnons AND phonons in the same axes, columns Fe / Ni (PRB two-column width): magnon-number-conserving rate (orange / blue) and,
    unless --no_rel, the relativistic non-conserving rate gamma_rel (purple magnons / green phonons).
    Optional reference lines gamma ~ E, E^2 with --ref_lines (log-log version, through the magnon median at --Eref)."""
    yl = ylim_of({m: data[m] for m in mats})
    tag = f"N{a.N}_s{sigma:g}" + ("_loglog" if LOGX["on"] else "")
    fig, axs = plt.subplots(1, 2, figsize=(10/2.52, 8/2.52), sharex=True, sharey=True)
    for i, (ax, m) in enumerate(zip(axs, mats)):
        df = data[m]
        for part in ("phonon", "magnon"):            # magnons on top
            k = keep(df, part, "cons")
            if k.any():
                ax.scatter(df.E[k], df["cons"][k], marker=PART[part][1], s=a.size, color=PCOL[part], alpha=a.alpha, linewidths=0, rasterized=True)
            ax.scatter([], [], marker=PART[part][1], s=24, color=PCOL[part], label=PART[part][0] + (", conserving" if not a.no_rel else ""))
        if not a.no_rel:
            for part in ("phonon", "magnon"):
                k = keep(df, part, "rel")
                if k.any():
                    ax.scatter(df.E[k], df["rel"][k], marker=PART[part][1], s=a.size * 2, color=RCOL[part], alpha=a.alpha_rel, linewidths=0, rasterized=True)
                ax.scatter([], [], marker=PART[part][1], s=24, color=RCOL[part], label=PART[part][0] + ", relativistic")
        if LOGX["on"] and a.ref_lines:
            w = keep(df, "magnon", "cons") & (abs(df.E - a.Eref) < 0.2 * a.Eref)
            if w.any():
                g0 = float(np.median(df["cons"][w])); E = np.geomspace(a.Eref / 20, df.E.max(), 100)
                ax.plot(E, g0 * E / a.Eref, "k-", lw=1.0, zorder=5, label=r"$\propto E$")
                ax.plot(E, g0 * (E / a.Eref) ** 2, "k--", lw=1.0, zorder=5, label=r"$\propto E^2$")
        ax.set_yscale("log"); ax.set_ylim(*yl)
        if LOGX["on"]:
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(matplotlib.ticker.LogFormatterSciNotation(minor_thresholds=(2, 0.4)))
            ax.tick_params(axis="x", which="minor", labelsize=6)
        ax.set_xlim(left=a.xmin)      # left limit = E_MIN, the same energy cut that removes the points
        #ax.grid(True, which="major", ls=":", alpha=0.4)
        label(ax, NAME[m], i)
        ax.set_xlabel("Energy (meV)")
    axs[0].set_ylabel(r"Scattering rate $\gamma$ (ps$^{-1}$)")
    if a.no_rel:
        axs[0].legend(frameon=False, loc="lower right", fontsize=7)
    else:                                   # four entries: above the panels, so the legend never covers data
        h, l = axs[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=6.5, columnspacing=1.2, handletextpad=0.2)
    finish(fig, os.path.join(a.out_dir, f"lifetimes_F_magphon_together_{tag}"), a.pdf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", nargs="+", default="Outputs/convergence_rel/conv_rel_20261112", help="scan folder with the runs (any scan holding N=--N runs at --T)")
    ap.add_argument("--N", type=int, default=40)
    ap.add_argument("--T", type=float, default=300.0)
    ap.add_argument("--sigmas", nargs="+", default=["0.2", "1"], help="broadenings (meV) or 'all' (every broadening found for both materials)")
    ap.add_argument("--out_dir", default="Outputs/paper/lifetimes")
    ap.add_argument("--e1", nargs=2, type=float, default=GUIDE["e1"], metavar=("LO", "HI"), help="meV window of the E^1 guide line, both materials")
    ap.add_argument("--e2_Fe", nargs=2, type=float, default=GUIDE["e2"]["bccFe"], metavar=("LO", "HI"), help="meV window of the E^2 line for Fe (HI may be inf)")
    ap.add_argument("--e2_Ni", nargs=2, type=float, default=GUIDE["e2"]["fccNi"], metavar=("LO", "HI"), help="meV window of the E^2 line for Ni")
    ap.add_argument("--alpha", type=float, default=0.04, help="marker alpha")
    ap.add_argument("--alpha_rel", type=float, default=0.35, help="marker alpha of the (sparse) relativistic points")
    ap.add_argument("--size", type=float, default=2.0, help="marker size (pt^2) of the F layout")
    ap.add_argument("--Eref", type=float, default=100.0, help="meV: the E^1 / E^2 reference lines of variant F (log-log) pass through the conserving-magnon median here")
    ap.add_argument("--xmin", type=float, default=E_MIN, help="left limit (meV) of the energy axis, linear and log-log; default = E_MIN, the lower energy cut of the plot (points below it are not drawn)")
    ap.add_argument("--all", action="store_true", help="also write the layout variants A-E (default: only F)")
    ap.add_argument("--ref_lines", action="store_true", help="E^1 / E^2 reference lines in the log-log version of F (default: off)")
    ap.add_argument("--guides", action="store_true", help="draw the E^1 / E^2 guide lines (default: off)")
    ap.add_argument("--no_rel", action="store_true", help="leave out the relativistic (non-conserving) rates in variant F")
    ap.add_argument("--no_pdf", action="store_true", help="PNG only (default: PNG + vector PDF)")
    ap.add_argument("--gmin", type=float, default=CUT["gmin"], help="drop rates below this (ps^-1)")
    ap.add_argument("--gmin_rel", type=float, default=CUT["gmin_rel"], help="floor (ps^-1) for the relativistic rates (they are orders of magnitude smaller than the conserving ones)")
    ap.add_argument("--plow", type=float, default=CUT["plow"], help="also drop the lowest PLOW percent of every particle/channel set (outliers)")
    a = ap.parse_args()
    a.pdf = not a.no_pdf; CUT.update(gmin=a.gmin, plow=a.plow, gmin_rel=a.gmin_rel); GUIDE.update(on=a.guides, e1=tuple(a.e1), e2={"bccFe": tuple(a.e2_Fe), "fccNi": tuple(a.e2_Ni)}); a.E0 = None
    style(); os.makedirs(a.out_dir, exist_ok=True)
    runs = find_runs(a.conv, a.N, a.T)
    avail = sorted({s for (m, s) in runs if all((mm, s) in runs for mm in MATS)})
    if not avail:
        raise SystemExit(f"no broadening with both materials at N={a.N}, T={a.T:g} K below {a.conv} (found {sorted(runs)})")
    sig = avail if a.sigmas == ["all"] else [float(s) for s in a.sigmas]
    print("broadenings available for both materials:", avail)
    for s in sig:
        if not all((m, s) in runs for m in MATS):
            print(f"skip sigma={s:g}: missing run"); continue
        data = {}
        for m in MATS:
            data[m], has_rel = load(runs[(m, s)]); data[m].attrs["mat"] = m
            n_rel = int(((data[m].rel > 0) & (data[m].particle == "phonon")).sum() + ((data[m].rel > 0) & (data[m].particle == "magnon")).sum())
            sl = ", ".join(f"E^{n} window {guide_win(data[m], n)[0]:g}-{guide_win(data[m], n)[1]:g} meV: free slope {guide_fit(data[m], n, *guide_win(data[m], n))[1]:.2f}" for n in (1, 2))
            print(f"sigma={s:g} {NAME[m]}: conserving magnon log-log slope  ({sl})")
            print(f"sigma={s:g} {NAME[m]}: relativistic column {'present' if has_rel else 'absent'}, {n_rel} modes with non-zero relativistic rate")
        for lg in (False, True):          # linear-x and log-log version of every layout
            LOGX["on"] = lg
            if a.all:
                make_all(data, MATS, s, a)
            make_F(data, MATS, s, a)


if __name__ == "__main__":
    main()
