"""
G_mp(T) of bcc Fe and fcc Ni on log-log axes with power-law fits  G ~ T^alpha.

    python python/plot_Gmp_powerlaw.py [--tscan Outputs/convergence_rel/Tscan_rel_20261112] [--channel mp|exch|rel|total]
                                       [--out_dir Outputs/paper] [--above_tc] [--win LO HI [LO HI ...]]

(a) G_mp vs T (log-log), points = runs of the T scan (N = 40, sigma = 0.2 meV), straight lines = least-squares power laws in
    log-log (fit windows below), exponent printed next to each line; T > T_C is excluded unless --above_tc.
(b) local exponent alpha(T) = d ln G / d ln T between neighbouring temperatures (points at the geometric mean of the two T).
Fit windows (K), default: low T 100-300 and high T 300-T_C per material, plus the whole range below T_C; change with --win.
Prints a table of alpha, its standard error (from the regression, N points - 2 dof) and the prefactor G(T = 300 K) of every fit.
Output: gmp_powerlaw.pdf/png and gmp_powerlaw_fits.csv in --out_dir. Rates/G_mp as in plot_Gmp_rel_paper.py.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_Gmp_rel_paper as P          # load_scan, style, save, NAME, COL, TC


def fit_power(T, G):
    """least squares ln G = ln A + alpha ln T; returns alpha, se(alpha), A (G at T = 1 K)"""
    x, y = np.log(T), np.log(G)
    n = len(x)
    if n < 2:
        return np.nan, np.nan, np.nan
    alpha, lnA = np.polyfit(x, y, 1)
    if n > 2:
        res = y - (lnA + alpha * x)
        se = np.sqrt(np.sum(res ** 2) / (n - 2) / np.sum((x - x.mean()) ** 2))
    else:
        se = np.nan
    return float(alpha), float(se), float(np.exp(lnA))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tscan", nargs="+", default="Outputs/convergence_rel/Tscan_rel_20261112")
    ap.add_argument("--channel", choices=("mp", "exch", "rel", "total"), default="mp")
    ap.add_argument("--tscan_N", type=int, default=40)
    ap.add_argument("--tscan_s", type=float, default=0.2)
    ap.add_argument("--tmax", type=float, default=800.0, help="highest temperature used (K)")
    ap.add_argument("--out_dir", default="Outputs/paper")
    ap.add_argument("--above_tc", action="store_true")
    ap.add_argument("--fits", action="store_true", help="the earlier two-panel figure with fitted power laws and local exponent (default: raw data + pure T^3)")
    ap.add_argument("--t3_T0", type=float, default=300.0, help="the T^3 curve passes through the Fe/Ni geometric-mean G at this T")
    ap.add_argument("--t3_scale", type=float, default=1.0, help="vertical shift of the T^3 curve (factor)")
    ap.add_argument("--win", nargs="+", type=float, default=None,
                    help="fit windows in K as LO HI pairs, applied to both materials (default: 100-300, 300-T_C, 100-T_C)")
    a = ap.parse_args()
    P.style()
    tab = P.load_scan(a.tscan)
    tab = tab[tab["T"] <= a.tmax + 1e-6]
    G = "G_" + a.channel
    tab = tab[(tab.N == a.tscan_N) & (abs(tab.smearing - a.tscan_s) < 1e-9) & np.isfinite(tab[G]) & (tab[G] > 0)]

    if not a.fits:
        fig, ax = plt.subplots(figsize=(3.4, 2.9))
        g0 = []
        for m in ("bccFe", "fccNi"):
            d = tab[tab.material == m].sort_values("T")
            if not a.above_tc:
                d = d[d["T"] <= P.TC[m] + 1e-6]
            if len(d):
                ax.plot(d["T"], d[G], "o-", color=P.COL[m], label=P.NAME[m])
            q = d[abs(d["T"] - a.t3_T0) < 1e-6]
            if len(q):
                g0.append(float(q[G].iloc[0]))
        if g0:
            Tg = np.geomspace(tab["T"].min(), (tab["T"].max() if a.above_tc else max(P.TC.values())), 50)
            Tg = Tg[Tg <= tab["T"].max()]
            Gg = a.t3_scale * np.exp(np.mean(np.log(g0))) * (Tg / a.t3_T0) ** 3
            ax.plot(Tg, Gg, color="k", ls="--", lw=1.1, label=r"$\propto T^3$")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$T$ (K)"); ax.set_ylabel(r"$G_{mp}$ (W m$^{-3}$ K$^{-1}$)")
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        P.save(fig, a.out_dir, "gmp_loglog_T3")
        return

    fig, (ax, axa) = plt.subplots(1, 2, figsize=(7.0, 3.2))
    rows = []
    for m in ("bccFe", "fccNi"):
        d = tab[tab.material == m].sort_values("T")
        if not a.above_tc:
            d = d[d["T"] <= P.TC[m] + 1e-6]
        if len(d) < 2:
            print(f"{m}: fewer than 2 points"); continue
        T, g = d["T"].to_numpy(float), d[G].to_numpy(float)
        col = P.COL[m]
        ax.plot(T, g, "o", color=col, label=P.NAME[m])
        # local exponent between neighbouring runs
        al = np.diff(np.log(g)) / np.diff(np.log(T)); Tm = np.sqrt(T[1:] * T[:-1])
        axa.plot(Tm, al, "o-", color=col, label=P.NAME[m])
        # fit windows
        top = T.max()
        wins = ([(a.win[i], a.win[i + 1]) for i in range(0, len(a.win) - 1, 2)] if a.win
                else [(100.0, 300.0), (300.0, top), (100.0, top)])
        for k, (lo, hi) in enumerate(wins):
            s = (T >= lo - 1e-6) & (T <= hi + 1e-6)
            alpha, se, A = fit_power(T[s], g[s])
            rows.append(dict(material=m, T_lo=lo, T_hi=hi, n_points=int(s.sum()), alpha=alpha, alpha_se=se, G_at_300K_fit=A * 300.0 ** alpha if np.isfinite(A) else np.nan))
            if not np.isfinite(alpha) or k == len(wins) - 1 and len(wins) == 3:
                continue          # the whole-range fit is only tabulated, not drawn
            Tf = np.geomspace(lo, hi, 20)
            ax.plot(Tf, A * Tf ** alpha, "-", color=col, lw=1.4, alpha=0.9)
            ax.annotate(rf"$\propto T^{{{alpha:.2f}}}$", (Tf[len(Tf) // 2], A * Tf[len(Tf) // 2] ** alpha), textcoords="offset points",
                        xytext=(6, -12), fontsize=7, color=col)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$T$ (K)"); ax.set_ylabel(r"$G_{mp}$ (W m$^{-3}$ K$^{-1}$)")
    ax.legend(frameon=False, loc="lower right"); ax.text(0.03, 0.95, "(a)", transform=ax.transAxes, va="top", fontweight="bold")
    axa.set_xscale("log"); axa.set_xlabel(r"$T$ (K)"); axa.set_ylabel(r"local exponent $d\ln G_{mp}/d\ln T$")
    axa.axhline(1, color="0.5", ls=":", lw=0.8); axa.axhline(2, color="0.5", ls=":", lw=0.8)
    axa.text(0.03, 0.95, "(b)", transform=axa.transAxes, va="top", fontweight="bold")
    fig.tight_layout()
    P.save(fig, a.out_dir, "gmp_powerlaw")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out_dir, "gmp_powerlaw_fits.csv"), index=False)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    for m in ("bccFe", "fccNi"):
        d = tab[tab.material == m].sort_values("T")
        if len(d) > 1:
            T, g = d["T"].to_numpy(float), d[G].to_numpy(float)
            print(m, "local exponents:", ", ".join(f"{t1:g}-{t2:g} K: {x:.2f}" for t1, t2, x in zip(T[:-1], T[1:], np.diff(np.log(g)) / np.diff(np.log(T)))))


if __name__ == "__main__":
    main()
