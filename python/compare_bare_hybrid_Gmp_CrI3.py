"""
G_mp of CrI3 from the BARE magnon-phonon lifetimes (eigenmode-basis vertices, theory/bare_lifetimes.tex) against the HYBRID basis, same grid / T / broadening.

    python python/compare_bare_hybrid_Gmp_CrI3.py [--bare DIR [DIR ...]] [--hybrid SIGMA=DIR ...] [--out_dir Outputs/paper]

bare   : path_lifetimes.csv of a full-BZ bare run (12^3 grid); G_mp = (1/N_q) sum_{phonon modes} E gamma_ph dn_BE/dT, gamma_ph = gamma_exch + gamma_rel
         (the magnon-phonon rate of the phonon rows), run broadening and temperature read from params.txt
hybrid : hybrid_path_lifetimes.csv of a full-BZ hybrid run; G_mp = (1/N_q) sum_{hybrid modes} w_ph E gamma dn_BE/dT with w_ph the phonon character
SI via the primitive cell V = 281.794 A^3 (8 atoms). Printed per broadening: both G_mp, their ratio, the phonon energy below which half of G_mp
is collected, and the median ratio of the single-phonon rates (bare phonon vs the hybrid mode of the same q and nearest energy, phonon character > 0.5).
Output: gmp_CrI3_bare_vs_hybrid.png/pdf and .csv in --out_dir.
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
import plot_Gmp_rel_paper as P

KB = 0.08617333262
V_CRI3 = 281.794
ROOT = "Outputs"
DEF_BARE = [f"{ROOT}/bare_multisub/cri3_bare_20261117/CrI3_N12_s{s}" for s in ("0.2", "1", "2")]
DEF_HYB = {0.2: f"{ROOT}/CrI3_Path_Hyrbid_Full_sig_0.2_15K", 1.0: f"{ROOT}/CrI3_Full_Hyrbid_sig_1.0_15K", 2.0: f"{ROOT}/CrI3_Full_Hyrbid_sig_2.0_15K",
           0.01: f"{ROOT}/CrI3_Path_Full_Hyrbid_sig_0.01_15K"}


def params(d):
    p = {}
    fn = os.path.join(d, "params.txt")
    if os.path.exists(fn):
        for line in open(fn):
            if "=" in line:
                k, v = line.strip().split("=", 1); p[k] = v
    return p


def to_SI(x):
    return x * P.MEV_PER_PS_TO_W / (V_CRI3 * P.ANG3)


def bare_modes(d, T):
    df = pd.read_csv(os.path.join(d, "path_lifetimes.csv"), comment="#")
    nq = df.q_idx.nunique()
    ph = df[df.particle == "phonon"].reset_index(drop=True)
    g = np.nan_to_num(P.rate(ph, "mp"))
    E = ph.energy_meV.to_numpy(float)
    c = np.where(E > P.E_MIN, E * g * P.dn_dT(E, T) / nq, 0.0)
    return pd.DataFrame(dict(q_idx=ph.q_idx, branch=ph.branch, E=E, gamma=g, c=to_SI(c)))


def hybrid_modes(d):
    df = pd.read_csv(os.path.join(d, "hybrid_path_lifetimes.csv"), comment="#")
    s = df[(df.energy_meV > 0.5) & (df.n_occ > 1e-8)]
    T = float(np.median(s.energy_meV / (KB * np.log1p(1.0 / s.n_occ))))
    nq = df.q_idx.nunique()
    E = df.energy_meV.to_numpy(float)
    w = np.clip(df.phon_character.to_numpy(float), 0, 1)
    g = np.nan_to_num(pd.to_numeric(df["gamma_ps-1"], errors="coerce").to_numpy(float))
    c = np.where(E > 0, w * E * g * P.dn_dT(E, T) / nq, 0.0)
    return T, pd.DataFrame(dict(q_idx=df.q_idx, E=E, gamma=g, w=w, c=to_SI(c)))


def half_energy(E, c):
    o = np.argsort(E)
    cs = np.cumsum(c[o]) / c.sum()
    return E[o][np.searchsorted(cs, 0.5)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bare", nargs="+", default=DEF_BARE)
    ap.add_argument("--hybrid", nargs="*", default=[f"{k}={v}" for k, v in DEF_HYB.items()], help="SIGMA=DIR")
    ap.add_argument("--out_dir", default="Outputs/paper")
    a = ap.parse_args()
    P.style()
    hyb = {float(x.split("=")[0]): x.split("=", 1)[1] for x in a.hybrid}
    rows, fig_data = [], []
    for d in a.bare:
        if not os.path.exists(os.path.join(d, "path_lifetimes.csv")):
            print(f"skip {d}: no path_lifetimes.csv"); continue
        p = params(d)
        s, T = float(p.get("smearing", "nan")), float(p.get("T", p.get("T_mag", 15)))
        b = bare_modes(d, T)
        row = dict(smearing=s, T=T, bare_dir=d, G_bare=b.c.sum(), E50_bare=half_energy(b.E.to_numpy(), b.c.to_numpy()))
        if s in hyb and os.path.exists(os.path.join(hyb[s], "hybrid_path_lifetimes.csv")):
            Th, h = hybrid_modes(hyb[s])
            row.update(T_hybrid=Th, G_hybrid=h.c.sum(), E50_hybrid=half_energy(h.E.to_numpy(), h.c.to_numpy()))
            row["ratio_bare_over_hybrid"] = row["G_bare"] / row["G_hybrid"]
            # single-phonon rates: bare phonon (q, branch) vs hybrid mode of the same q with the nearest energy and phonon character > 0.5
            rat = []
            for q, gb in b[b.E > 3.0].groupby("q_idx"):
                hq = h[(h.q_idx == q) & (h.w > 0.5)]
                if not len(hq):
                    continue
                for _, r in gb.iterrows():
                    j = (hq.E - r.E).abs().idxmin()
                    if abs(hq.E[j] - r.E) < 0.5 and hq.gamma[j] > 1e-8 and r.gamma > 1e-8:
                        rat.append(r.gamma / hq.gamma[j])
            row["median_rate_ratio"] = float(np.median(rat)) if rat else np.nan
            row["n_rate_pairs"] = len(rat)
            fig_data.append((s, b, h))
        rows.append(row)
    res = pd.DataFrame(rows).sort_values("smearing")
    print(res.drop(columns=["bare_dir"]).to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    os.makedirs(a.out_dir, exist_ok=True)
    res.to_csv(os.path.join(a.out_dir, "gmp_CrI3_bare_vs_hybrid.csv"), index=False)

    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
    ax = axs[0]
    ax.plot(res.smearing, res.G_bare, "o-", color="#1f77b4", label="bare (eigenmode basis)")
    if "G_hybrid" in res:
        ax.plot(res.smearing, res.G_hybrid, "s-", color="#2ca02c", label="hybrid")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"broadening $\sigma$ (meV)"); ax.set_ylabel(r"$G_{mp}$ (W m$^{-3}$ K$^{-1}$)")
    ax.set_title(r"CrI$_3$, $12^3$ grid, 15 K", fontsize=8); ax.legend(frameon=False, fontsize=7)
    ax = axs[1]
    for (s, b, h), col in zip(sorted(fig_data, key=lambda x: x[0]), ("#4c78a8", "#f28e2b", "#e15759", "#59a14f")):
        for dat, ls, lab in ((b, "-", "bare"), (h, "--", "hybrid")):
            o = np.argsort(dat.E.to_numpy())
            ax.plot(dat.E.to_numpy()[o], np.cumsum(dat.c.to_numpy()[o]) / dat.c.sum(), ls, color=col, lw=1.0, label=rf"{lab}, $\sigma={s:g}$")
    ax.set_xlabel("mode energy (meV)"); ax.set_ylabel(r"cumulative share of $G_{mp}$"); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=5.5, ncol=2, loc="lower right")
    for x, l in zip(axs, "ab"):
        x.text(0.03, 0.95, f"({l})", transform=x.transAxes, fontweight="bold", va="top")
    fig.tight_layout()
    P.save(fig, a.out_dir, "gmp_CrI3_bare_vs_hybrid")


if __name__ == "__main__":
    main()
