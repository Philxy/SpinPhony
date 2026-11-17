"""
Magnon-resolved G_mp (which magnons carry the magnon-phonon heat transfer), from the --partner_diag output magnon_Gmp_contrib.csv.

    python python/plot_Gmp_magnons.py [--run_dir Outputs/partner_diag/partner_sig1_20261117] [--N 40] [--channel mp|exch|rel] [--out_dir Outputs/paper]

Every scattering event (phonon (q,lam) <-> magnon pair (k,n),(p,m)) delivers G_mp share  rate * E_ph dn/dT / N_q  to the phonon; it is split
equally between the two magnons of the event, so  G_mp = sum over grid magnons (k,n) of c(k,n)  exactly (checked against the phonon-row sum).
Shown (columns Fe | Ni): (a,b) G_mp(E_magnon) spectral function per magnon branch, (c,d) cumulative share, (e,f) N=20 vs N=40 (convergence).
Units: SI via the primitive cell volume (Fe a^3/2, Ni a^3/4). Attribution convention: each event is split equally between its two magnons (spectrum/cumulative above); the last row shows the bounds when the whole share
goes to the lower- or higher-energy magnon of the event (needs the hi/lo columns of runs made with the current partner_diag.py). Phonon energies are <= ~38 meV, so the
convention shifts a contribution by at most ~eps_ph/2 ~ 19 meV. Only the phonon-row based file is used: the magnon-row columns of
path_partner_magnons.csv lack the absorption (c_type 1) events and are not used here.
"""
import argparse, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_Gmp_rel_paper as P

BCOL = ["#4c78a8", "#f28e2b", "#59a14f", "#e15759"]
HL = {"mp": ["G_exch_{}_meV_ps-1_K-1_per_cell", "G_rel_{}_meV_ps-1_K-1_per_cell"], "exch": ["G_exch_{}_meV_ps-1_K-1_per_cell"], "rel": ["G_rel_{}_meV_ps-1_K-1_per_cell"]}
CH = {"mp": ["G_exch_meV_ps-1_K-1_per_cell", "G_rel_meV_ps-1_K-1_per_cell"], "exch": ["G_exch_meV_ps-1_K-1_per_cell"], "rel": ["G_rel_meV_ps-1_K-1_per_cell"]}

def load(run_dir, m, N, chan, conv=None):
    pre = "bccFe" if m == "bccFe" else "fccNi"
    d = pd.read_csv(f"{run_dir}/{pre}_N{N}_s1/magnon_Gmp_contrib.csv")
    cols = CH[chan] if conv is None else [k.format(conv) for k in HL[chan]]
    if conv is not None and cols[0] not in d: return None          # run made before the hi/lo attribution columns existed
    c = sum(d[k] for k in cols) * P.MEV_PER_PS_TO_W / (P.VOL[m] * P.ANG3)
    return d.assign(c=c)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="Outputs/partner_diag/partner_sig1_20261117")
    ap.add_argument("--N", type=int, default=40); ap.add_argument("--channel", choices=CH, default="mp")
    ap.add_argument("--bin", type=float, default=5.0); ap.add_argument("--out_dir", default="Outputs/paper")
    a = ap.parse_args(); P.style()
    has_hl = load(a.run_dir, 'bccFe', a.N, a.channel, 'hi') is not None
    nr = 4 if has_hl else 3
    fig, axs = plt.subplots(nr, 2, figsize=(7.0, 2.5 * nr)); rows = []
    for j, m in enumerate(("bccFe", "fccNi")):
        t = load(a.run_dir, m, a.N, a.channel); tot = t.c.sum()
        Emax = np.ceil(t.energy_meV.max() / a.bin) * a.bin; edges = np.arange(0, Emax + a.bin, a.bin); ctr = .5 * (edges[1:] + edges[:-1])
        nb = int(t.branch.max()) + 1
        spec = np.array([np.histogram(t.energy_meV[t.branch == b], bins=edges, weights=t.c[t.branch == b])[0] / a.bin for b in range(nb)])
        ax = axs[0, j]; ax.stackplot(ctr, spec, colors=BCOL[:nb], labels=[f"magnon band {b+1}" for b in range(nb)], lw=0); ax.plot(ctr, spec.sum(0), "k", lw=.9)
        ax.set_ylabel(r"$G_{mp}(E_{mag})$ (W m$^{-3}$ K$^{-1}$ meV$^{-1}$)"); ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.set_title(P.NAME[m] + rf", $T=300$ K, $N={a.N}$", fontsize=8); ax.legend(frameon=False, fontsize=6.5)
        ax = axs[1, j]; o = np.argsort(t.energy_meV.to_numpy()); Es = t.energy_meV.to_numpy()[o]; cs = np.cumsum(t.c.to_numpy()[o]) / tot
        ax.plot(Es, cs, "k:", label="total")
        for b in range(nb):
            s = t[t.branch == b].sort_values("energy_meV"); ax.plot(s.energy_meV, np.cumsum(s.c) / tot, color=BCOL[b], label=f"band {b+1} ({s.c.sum()/tot*100:.0f}%)")
        pct = [Es[np.searchsorted(cs, f)] for f in (.1, .5, .9)]; ax.axhline(.5, color=".6", lw=.6); ax.axvline(pct[1], color=".6", lw=.6)
        ax.set_ylabel(r"cumulative share of $G_{mp}$"); ax.set_ylim(0, 1.02); ax.legend(frameon=False, fontsize=6.5, loc="lower right")
        ax = axs[2, j]
        for N, ls in ((20, "--"), (40, "-")):
            u = load(a.run_dir, m, N, a.channel); h = np.histogram(u.energy_meV, bins=edges, weights=u.c)[0] / a.bin
            ax.plot(ctr, h, ls, color="k", lw=.9, label=f"N={N}  (G={u.c.sum():.3g})")
        ax.set_ylabel(r"$G_{mp}(E_{mag})$"); ax.set_xlabel("" if has_hl else "magnon energy (meV)"); ax.legend(frameon=False, fontsize=6.5)
        if has_hl:
            ax = axs[3, j]
            for conv, col, lab in (("lo", "#4c78a8", "all to lower-E magnon"), (None, "k", "half to each (used above)"), ("hi", "#e15759", "all to higher-E magnon")):
                u = load(a.run_dir, m, a.N, a.channel, conv) if conv else t
                cs_ = np.cumsum(u.c.to_numpy()[np.argsort(u.energy_meV.to_numpy())]) / u.c.sum(); Eu = np.sort(u.energy_meV.to_numpy())
                ax.plot(Eu, cs_, color=col, label=lab); p_ = [Eu[np.searchsorted(cs_, f)] for f in (.1, .5, .9)]
                print(f"   attribution {lab:28s} 10/50/90 %: {p_[0]:6.1f} / {p_[1]:6.1f} / {p_[2]:6.1f} meV")
            ax.axhline(.5, color=".6", lw=.6); ax.set_ylim(0, 1.02); ax.set_ylabel(r"cumulative share, by convention"); ax.set_xlabel("magnon energy (meV)"); ax.legend(frameon=False, fontsize=6.5, loc="lower right")
        for r in range(nr): axs[r, j].set_xlim(0, Emax)
        print(f"\n{m}  N={a.N} channel {a.channel}: G_mp = {tot:.4g} W/m3/K;  magnon energy at 10/50/90 %: {pct[0]:.1f}/{pct[1]:.1f}/{pct[2]:.1f} meV (max {t.energy_meV.max():.0f})")
        for b in range(nb):
            s = t[t.branch == b]; print(f"   band {b+1}: share {s.c.sum()/tot*100:5.1f} %  <E>_w = {np.average(s.energy_meV, weights=s.c):6.1f} meV")
        for b in range(nb):
            for lo, hi, v in zip(edges[:-1], edges[1:], spec[b]): rows.append(dict(material=m, N=a.N, channel=a.channel, band=b + 1, E_lo=lo, E_hi=hi, G_per_meV=v))
    for ax, l in zip(axs.flat, "abcdefgh"): ax.text(0.03, 0.95, f"({l})", transform=ax.transAxes, fontweight="bold", va="top")
    fig.tight_layout(); tag = "gmp_magnons_300K" + ("" if a.channel == "mp" else "_" + a.channel); P.save(fig, a.out_dir, tag)
    pd.DataFrame(rows).to_csv(f"{a.out_dir}/{tag}.csv", index=False)
if __name__ == "__main__": main()
