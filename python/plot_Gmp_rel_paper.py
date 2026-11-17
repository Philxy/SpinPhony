"""
G_mp convergence and temperature dependence of bcc Fe and fcc Ni (bare lifetimes incl. the relativistic channels), PRB style.

    python python/plot_Gmp_rel_paper.py [--conv Outputs/convergence_rel/conv_rel_20261112]
                                        [--tscan Outputs/convergence_rel/Tscan_rel_20261112]
                                        [--out_dir Outputs/paper] [--T 300] [--channel total|exch|rel]
                                        [--ref 3e16] [--exp Inputs/exp/Ca9La5Cu24O41_Gmp.csv]

Figure 1 (gmp_convergence.pdf/png), 2x2, all at T = --T (300 K):
    (a) Fe: G_mp vs grid size N, one curve per Gaussian broadening       (b) Fe: G_mp vs broadening (log x), one curve per N
    (c) Ni: same as (a)                                                  (d) Ni: same as (b)
Figure 2 (gmp_vs_T.pdf/png): G_mp(T) of Fe and Ni (N = --tscan_N, broadening --tscan_s) in one plot, a horizontal reference line
    at --ref (W m^-3 K^-1) and the experimental ladder-compound data Ca9La5Cu24O41 (G column x --exp_scale).
G_mp = (1/N_q) sum_{phonons} E gamma dn_BE/dT   [meV K^-1 ps^-1 per cell] -> W m^-3 K^-1 with the primitive cell volume
(bcc Fe a^3/2, a = 2.8665 A; fcc Ni a^3/4, a = 3.516822176 A). --channel picks the rate:
mp = magnon-phonon (default), exch = gamma_exch_ps-1 (magnon-number conserving only), rel = gamma_rel_ps-1, total = gamma_ps-1.
Default mp = exch + rel, the magnon-phonon rate WITHOUT the three-phonon part that the bcc Fe runs (phph=1) also put into gamma_ps-1
(if the file has no exch column, gamma_ps-1 minus gamma_phph_ps-1 is used). total = gamma_ps-1 as written.
A table of all values (mp / exch / rel / total) is written to <out_dir>/gmp_table.csv and printed.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KB = 0.08617333262          # meV / K
MEV_PER_PS_TO_W = 1.602176634e-10
ANG3 = 1e-30
VOL = {"bccFe": 2.8665 ** 3 / 2.0, "fccNi": 3.516822176 ** 3 / 4.0}      # primitive cell volume (A^3)
TC = {"bccFe": 1043.0, "fccNi": 627.0}      # Curie temperatures (K)
NAME = {"bccFe": "bcc Fe", "fccNi": "fcc Ni"}
COL = {"bccFe": "#1f77b4", "fccNi": "#d95f02"}
E_MIN = 0.2                  # meV, same cut as the other G_mp scripts


def read_params(path):
    p = {}
    if os.path.exists(path):
        for line in open(path):
            if "=" in line:
                k, v = line.strip().split("=", 1)
                p[k] = v
    return p


def dn_dT(E, T):
    x = E / (KB * T)
    out = np.zeros_like(x)
    ok = (x > 0) & (x < 700)
    out[ok] = (x[ok] / T) / (4.0 * np.sinh(0.5 * x[ok]) ** 2)
    return out


def rate(ph, key):
    """Phonon scattering rate (ps^-1) of one channel group. Files from runs with three-phonon scattering (bcc Fe, phph=1) hold
    gamma_ps-1 = exch + rel + phph; magnon-phonon coupling needs only exch + rel."""
    num = lambda c: pd.to_numeric(ph[c], errors="coerce").fillna(0.0).to_numpy(float) if c in ph.columns else None
    tot, ex, rl, pp = num("gamma_ps-1"), num("gamma_exch_ps-1"), num("gamma_rel_ps-1"), num("gamma_phph_ps-1")
    if key == "total":
        return tot
    if key == "exch":
        return ex
    if key == "rel":
        return rl
    # "mp": magnon-phonon rate, conserving + relativistic
    if ex is not None:
        return ex + (rl if rl is not None else 0.0)
    return tot - (pp if pp is not None else 0.0)


def gmp_SI(df, T, vol, key):
    ph = df[df.particle == "phonon"]
    g = rate(ph, key)
    if g is None:
        return np.nan
    E = ph.energy_meV.to_numpy(float)
    g = np.where(np.isfinite(g), g, 0.0)
    k = E > E_MIN
    G = np.sum(E[k] * g[k] * dn_dT(E[k], T)) / df.q_idx.nunique()
    return G * MEV_PER_PS_TO_W / (vol * ANG3)


def load_scan(root):
    """One row per finished run: material, N, smearing, T and G_mp (SI) for total / exch / rel. root: one folder or a list of folders (e.g. Fe and Ni scans)."""
    rows = []
    roots = [root] if isinstance(root, str) else list(root)
    items = []                      # "folder" or "folder:material" (only that material is taken from the folder)
    for r in roots:
        r, _, only = r.partition(":")
        items += [(f, only) for f in glob.glob(os.path.join(r, "*", "params.txt"))]
    for pfile, only in sorted(items):
        d = os.path.dirname(pfile)
        csv = os.path.join(d, "path_lifetimes.csv")
        p = read_params(pfile)
        if not os.path.exists(csv) or "material" not in p or p["material"] not in VOL or (only and p["material"] != only):
            continue
        df = pd.read_csv(csv, comment="#")
        T = float(p.get("T", p.get("T_mag", 300.0)))
        r = dict(material=p["material"], N=int(float(p["N"])), smearing=float(p["smearing"]), T=T, run=os.path.basename(d))
        for key in ("mp", "exch", "rel", "total"):
            r["G_" + key] = gmp_SI(df, T, VOL[p["material"]], key)
        rows.append(r)
    return pd.DataFrame(rows)


def style():
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 9, "legend.fontsize": 7,
        "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.minor.size": 2, "ytick.minor.size": 2,
        "lines.linewidth": 1.3, "lines.markersize": 4.5, "savefig.dpi": 600})


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), bbox_inches="tight")
    print(f" -> {os.path.join(out_dir, name)}.pdf / .png")


def fig_convergence(tab, T, chan, out_dir):
    G = "G_" + chan
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.6))
    letters = iter("abcd")
    for i, m in enumerate(("bccFe", "fccNi")):
        d = tab[(tab.material == m) & (abs(tab["T"] - T) < 1e-6) & np.isfinite(tab[G]) & (tab[G] > 0)]
        axN, axS = axs[i]
        if d.empty:
            for ax in (axN, axS):
                ax.text(0.5, 0.5, f"no {NAME[m]} runs at {T:g} K", ha="center", va="center", transform=ax.transAxes)
            continue
        sm = sorted(d.smearing.unique()); Ns = sorted(d.N.unique())
        cs = plt.cm.viridis(np.linspace(0.05, 0.9, len(sm)))
        for c, s in zip(cs, sm):
            g = d[d.smearing == s].sort_values("N")
            axN.plot(g.N, g[G], "o-", color=c, label=f"{s:g}")
        cn = plt.cm.plasma(np.linspace(0.1, 0.75, len(Ns)))
        for c, n in zip(cn, Ns):
            g = d[d.N == n].sort_values("smearing")
            axS.plot(g.smearing, g[G], "s-", color=c, label=f"{n}")
        axN.set_xticks(Ns); axN.set_xlabel(r"grid size $N$ ($N\times N\times N$)")
        axS.set_xscale("log"); axS.set_xlabel(r"Gaussian broadening $\sigma$ (meV)")
        axN.legend(title=r"$\sigma$ (meV)", title_fontsize=7, ncol=2, frameon=False, loc="best")
        axS.legend(title=r"$N$", title_fontsize=7, frameon=False, loc="best")
        for ax in (axN, axS):
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
            ax.set_ylabel(r"$G_{mp}$ (W m$^{-3}$ K$^{-1}$)")
            ax.set_title(NAME[m], fontsize=8, loc="right", pad=2)
            ax.text(0.03, 0.93, f"({next(letters)})", transform=ax.transAxes, fontweight="bold", va="top")
            ax.grid(True, which="major", ls=":", alpha=0.4)
    fig.tight_layout()
    save(fig, out_dir, "gmp_convergence")


def fig_temperature(tab, a, chan, out_dir, logx=False, t3=False, name="gmp_vs_T"):
    G = "G_" + chan
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    for m in ("bccFe", "fccNi"):
        d = tab[(tab.material == m) & (tab.N == a.tscan_N) & (abs(tab.smearing - a.tscan_s) < 1e-9) & np.isfinite(tab[G]) & (tab[G] > a.gmin)]
        if not a.above_tc:
            d = d[d["T"] <= TC[m] + 1e-6]
        d = d.sort_values("T")
        if d.empty:
            print(f"no {m} runs with N={a.tscan_N}, sigma={a.tscan_s:g} in {a.tscan}")
            continue
        ax.plot(d["T"], d[G], "o-", color=COL[m], label=NAME[m])
    # guide for the eye: G ~ T^3, shifted below the data (through scale x the Fe/Ni geometric mean at T0)
    if t3:
        g0 = []
        for m in ("bccFe", "fccNi"):
            q = tab[(tab.material == m) & (tab.N == a.tscan_N) & (abs(tab.smearing - a.tscan_s) < 1e-9) & (abs(tab["T"] - a.t3_T0) < 1e-6) & (tab[G] > 0)]
            if len(q):
                g0.append(float(q[G].iloc[0]))
        if g0:
            Tg = np.linspace(a.t3_range[0], a.t3_range[1], 50)
            Gg = a.t3_scale * np.exp(np.mean(np.log(g0))) * (Tg / a.t3_T0) ** 3
            ax.plot(Tg, Gg, color="0.45", ls="-", lw=1.1, zorder=1)
            ax.text(Tg[-1], Gg[-1], r" $\propto T^3$", fontsize=7, color="0.3", ha="left", va="center")
    ax.axhline(a.ref, color="0.35", ls="--", lw=1.0)
    # label directly above the line (left end, axes-fraction x / data y): no legend entry, no value
    ax.text(0.03, a.ref * 1.25, a.ref_label, transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=7, color="0.2")
    if a.exp and a.exp.lower() != "none" and os.path.exists(a.exp):
        e = pd.read_csv(a.exp, skipinitialspace=True)
        ax.plot(e["T_K"], e["G"].to_numpy(float) * a.exp_scale, "k^", mfc="none", ms=4.5, ls="none",
                label=r" Ca$_9$La$_5$Cu$_{24}$O$_{41}$  (experiment)")
    ax.set_yscale("log"); ax.set_xlabel(r"$T$ (K)"); ax.set_ylabel(r"$G_{mp}$ (W m$^{-3}$ K$^{-1}$)")
    if logx:
        ax.set_xscale("log")
    ax.grid(True, which="major", ls=":", alpha=0.4); ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, out_dir, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", nargs="+", default="Outputs/convergence_rel/conv_rel_20261112")
    ap.add_argument("--tscan", nargs="+", default="Outputs/convergence_rel/Tscan_rel_20261112")
    ap.add_argument("--tmax", type=float, default=800.0, help="highest temperature plotted in the T scans (K)")
    ap.add_argument("--out_dir", default="Outputs/paper")
    ap.add_argument("--T", type=float, default=300.0, help="temperature of the convergence plots (K)")
    ap.add_argument("--channel", choices=("mp", "exch", "rel", "total"), default="mp")
    ap.add_argument("--tscan_N", type=int, default=40)
    ap.add_argument("--tscan_s", type=float, default=0.2)
    ap.add_argument("--ref", type=float, default=0.3e17, help="horizontal reference line, W m^-3 K^-1 (0.3e17)")
    ap.add_argument("--ref_label", default=r"Beaurepaire $et\ al.$ (1996)", help="text above the reference line")
    ap.add_argument("--t3_dummy", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--t3_scale", type=float, default=0.25, help="vertical position of the T^3 guide: factor times the Fe/Ni geometric-mean G at --t3_T0")
    ap.add_argument("--t3_T0", type=float, default=300.0)
    ap.add_argument("--t3_range", nargs=2, type=float, default=(230.0, 600.0), metavar=("LO", "HI"))
    ap.add_argument("--exp", default="Inputs/exp/Ca9La5Cu24O41_Gmp.csv")
    ap.add_argument("--exp_scale", type=float, default=1e15, help="exp. G column -> W m^-3 K^-1 (as in plot_Gmp_SI_vs_TTc.py)")
    ap.add_argument("--gmin", type=float, default=1e9, help="drop points below this G in the T plot (exponentially small low-T tail)")
    ap.add_argument("--above_tc", action="store_true", help="also plot T > T_C (Fe 1043 K, Ni 627 K); default: T <= T_C only")
    a = ap.parse_args()
    style()
    conv = load_scan(a.conv); ts = load_scan(a.tscan); ts = ts[ts["T"] <= a.tmax + 1e-6]
    os.makedirs(a.out_dir, exist_ok=True)
    pd.concat([conv.assign(scan="conv"), ts.assign(scan="tscan")], ignore_index=True).to_csv(os.path.join(a.out_dir, "gmp_table.csv"), index=False)
    for nm, t in (("convergence", conv), ("T-scan", ts)):
        print(f"\n{nm}: {len(t)} runs")
        if len(t):
            print(t.sort_values(["material", "N", "smearing", "T"]).drop(columns="run").to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    if len(conv):
        fig_convergence(conv, a.T, a.channel, a.out_dir)
    if len(ts):
        fig_temperature(ts, a, a.channel, a.out_dir)                                                       # as before, no guide
        fig_temperature(ts, a, a.channel, a.out_dir, logx=False, t3=True, name="gmp_vs_T_T3_linlog")      # lin-log + T^3
        fig_temperature(ts, a, a.channel, a.out_dir, logx=True, t3=True, name="gmp_vs_T_T3_loglog")       # log-log + T^3


if __name__ == "__main__":
    main()
