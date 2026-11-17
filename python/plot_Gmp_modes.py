"""
Which phonons carry G_mp?  Mode-resolved decomposition of G_mp at one temperature (default 300 K) for bcc Fe and fcc Ni.

    python python/plot_Gmp_modes.py [--fe DIR] [--ni DIR] [--T 300] [--channel mp|exch|rel] [--out_dir Outputs/paper] [--bin 1.0]

G_mp = (1/N_q) sum_{phonon modes (q,lambda)} E gamma dn_BE/dT  (as in plot_Gmp_rel_paper.py, SI via the primitive cell volume), so every phonon
mode has its own exact contribution c(q,lambda) = E gamma dn_BE/dT / N_q  and  G_mp = sum c.  Shown (columns Fe | Ni):
  (a,b) spectral function G(E) = sum_{E in bin} c / bin width, stacked by phonon branch (labelled by energy order at each q: lowest / middle / highest acoustic
        band; this is not a TA/LA assignment, which would need the polarisation vectors), black line = total
  (c,d) cumulative share of G_mp versus phonon energy per branch (dotted: total) with the energy below which 50 % is collected
  (e,f) the ingredients at the same energy: mean scattering rate <gamma>(E) (left axis) and the thermal weight E dn/dT per mode (right axis)
Magnons are NOT resolved here: the lifetime files hold only the rate of each mode, not which magnon it scatters with. The magnon-side
sum E_mag gamma_mag dn/dT is not the transferred heat (a magnon-number-conserving process hands over the phonon energy, not E_mag) and is
therefore not used; a magnon-resolved G_mp needs the process-resolved (E_magnon, E_phonon) weights from the scattering loop.
Output: gmp_modes_<T>K.png/pdf and gmp_modes_<T>K.csv (spectral function per branch) in --out_dir; the printout gives each branch's share,
the energies at 10/50/90 % of G_mp and the number of modes carrying 50 % of it.
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
import plot_Gmp_rel_paper as P          # rate, dn_dT, VOL, E_MIN, MEV_PER_PS_TO_W, ANG3, style, save, NAME, COL

BCOL = ["#4c78a8", "#59a14f", "#e15759"]
BNAME = ["lowest", "middle", "highest"]      # energy order at each q, NOT polarisation (no eigenvectors in the lifetime files)


def modes(d, m, T, chan):
    df = pd.read_csv(os.path.join(d, "path_lifetimes.csv"), comment="#")
    nq = df.q_idx.nunique()
    ph = df[df.particle == "phonon"].reset_index(drop=True)
    g = P.rate(ph, chan)
    g = np.where(np.isfinite(g), g, 0.0)
    E = ph.energy_meV.to_numpy(float)
    c = np.where(E > P.E_MIN, E * g * P.dn_dT(E, T) / nq, 0.0) * P.MEV_PER_PS_TO_W / (P.VOL[m] * P.ANG3)
    return pd.DataFrame(dict(q_idx=ph.q_idx, qx=ph.qx, qy=ph.qy, qz=ph.qz, branch=ph.branch.astype(int), E=E, gamma=g, c=c, w=P.dn_dT(E, T) * E))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fe", default="Outputs/convergence_rel/Tscan_rel_20261112/bccFe_N40_s0.2_T300K")
    ap.add_argument("--ni", default="Outputs/convergence_rel/Tscan_rel_NEW/fccNi_N40_s0.2_T300K")
    ap.add_argument("--T", type=float, default=300.0)
    ap.add_argument("--channel", choices=("mp", "exch", "rel"), default="mp")
    ap.add_argument("--bin", type=float, default=1.0, help="energy bin (meV)")
    ap.add_argument("--out_dir", default="Outputs/paper")
    a = ap.parse_args()
    P.style()
    fig, axs = plt.subplots(3, 2, figsize=(7.0, 7.4))
    rows = []
    for j, (m, d) in enumerate((("bccFe", a.fe), ("fccNi", a.ni))):
        t = modes(d, m, a.T, a.channel)
        tot = t.c.sum()
        Emax = np.ceil(t.E.max() / a.bin) * a.bin
        edges = np.arange(0.0, Emax + a.bin, a.bin)
        ctr = 0.5 * (edges[1:] + edges[:-1])
        nb = t.branch.max() + 1
        spec = np.array([np.histogram(t.E[t.branch == b], bins=edges, weights=t.c[t.branch == b])[0] / a.bin for b in range(nb)])
        ax = axs[0, j]
        ax.stackplot(ctr, spec, colors=BCOL[:nb], labels=BNAME[:nb], alpha=0.9, lw=0)
        ax.plot(ctr, spec.sum(0), "k", lw=0.9)
        ax.set_ylabel(r"$G_{mp}(E)$ (W m$^{-3}$ K$^{-1}$ meV$^{-1}$)")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.set_title(P.NAME[m] + f", $T={a.T:g}$ K", fontsize=8)
        ax.legend(frameon=False, loc="upper left", ncol=1)
        # cumulative
        ax = axs[1, j]
        o = np.argsort(t.E.to_numpy())
        Es = t.E.to_numpy()[o]
        ax.plot(Es, np.cumsum(t.c.to_numpy()[o]) / tot, "k:", lw=1.2, label="total")
        e50 = Es[np.searchsorted(np.cumsum(t.c.to_numpy()[o]) / tot, 0.5)]
        ax.axvline(e50, color="0.5", lw=0.6); ax.axhline(0.5, color="0.5", lw=0.6)
        for b in range(nb):
            s = t[t.branch == b].sort_values("E")
            ax.plot(s.E, np.cumsum(s.c) / tot, color=BCOL[b], label=f"{BNAME[b]} ({s.c.sum() / tot * 100:.0f}%)")
        ax.set_ylabel(r"cumulative share of $G_{mp}$"); ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, loc="upper left", fontsize=6.5)
        # ingredients: <gamma>(E) and E dn/dT
        ax = axs[2, j]
        for b in range(nb):
            s = t[(t.branch == b) & (t.E > P.E_MIN)]
            gm = [s.gamma[(s.E >= lo) & (s.E < hi)].mean() if ((s.E >= lo) & (s.E < hi)).any() else np.nan for lo, hi in zip(edges[:-1], edges[1:])]
            ax.plot(ctr, gm, color=BCOL[b], lw=1.0)
        ax.set_yscale("log"); ax.set_ylabel(r"$\langle\gamma\rangle(E)$ (ps$^{-1}$)")
        ax2 = ax.twinx()
        Eg = np.linspace(0.3, Emax, 300)
        ax2.plot(Eg, Eg * P.dn_dT(Eg, a.T) / (Eg * P.dn_dT(Eg, a.T)).max(), "k--", lw=0.9)
        ax2.set_ylabel(r"$E\,\partial n_{BE}/\partial T$ (norm.)"); ax2.set_ylim(0, 1.05)
        for r in range(3):
            axs[r, j].set_xlim(0, Emax)
        axs[2, j].set_xlabel(r"phonon energy (meV)")
        # printout / table
        cs = np.cumsum(t.c.to_numpy()[o]) / tot
        pct = [Es[np.searchsorted(cs, f)] for f in (0.1, 0.5, 0.9)]
        srt = np.sort(t.c.to_numpy())[::-1]
        n50 = int(np.searchsorted(np.cumsum(srt) / tot, 0.5)) + 1
        print(f"\n{m}  T={a.T:g} K  channel {a.channel}:  G_mp = {tot:.4g} W m^-3 K^-1  ({len(t)} phonon modes, {n50} modes = {n50 / len(t) * 100:.2f}% carry 50 %)")
        print(f"  phonon energy at 10/50/90 % of G_mp: {pct[0]:.1f} / {pct[1]:.1f} / {pct[2]:.1f} meV   (max phonon energy {t.E.max():.1f} meV)")
        for b in range(nb):
            s = t[t.branch == b]
            print(f"  branch {b + 1} ({BNAME[b].replace('$', '').replace('_', '')}): share {s.c.sum() / tot * 100:5.1f} %   <E> weighted {np.average(s.E, weights=s.c):5.1f} meV   mean gamma {s.gamma[s.E > P.E_MIN].mean():.3g} ps^-1   share of modes {len(s) / len(t) * 100:.0f} %")
        for b in range(nb):
            for lo, hi, v in zip(edges[:-1], edges[1:], spec[b]):
                rows.append(dict(material=m, T=a.T, channel=a.channel, branch=b + 1, E_lo=lo, E_hi=hi, G_per_meV=v, G_bin=v * a.bin))
        # top contributing wave vectors
        top = t.sort_values("c", ascending=False).head(5)
        print("  largest single-mode contributions (q in the units of the file, branch, E, gamma, share):")
        for _, r in top.iterrows():
            print(f"     q=({r.qx:.3f},{r.qy:.3f},{r.qz:.3f}) branch {int(r.branch) + 1}  E={r.E:.1f} meV  gamma={r.gamma:.3g} ps^-1  {r.c / tot * 100:.2f} %")
    for ax, l in zip(axs.flat, "abcdef"):
        ax.text(0.0, 1.03, f"({l})", transform=ax.transAxes, ha="left", va="bottom", fontweight="bold")
    fig.tight_layout()
    tag = f"gmp_modes_{a.T:g}K" + ("" if a.channel == "mp" else "_" + a.channel)
    P.save(fig, a.out_dir, tag)
    pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, tag + ".csv"), index=False)
    print(f" -> {os.path.join(a.out_dir, tag)}.csv")


if __name__ == "__main__":
    main()
