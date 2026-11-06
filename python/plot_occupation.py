"""
Occupation plots of the hybrid dynamics (needs occupation_snapshots.npz and observables_dynamics.txt of a run with
SpinPhony.py --hybrid_dynamics; --pump is optional, without it the plots show the relaxation of the thermal start state).

    python python/plot_occupation.py --dir Outputs [--nlines 8]

The metric is the mode temperature  T_eff(k,mu) = eps / ( k_B ln(1 + 1/n) )  -  the temperature of the Bose-Einstein
distribution that has the occupation n at the energy eps. A thermal state is a horizontal line at T; a pumped mode
sticks out far above it, and the daughters it feeds show up as warm modes at lower energy.

  dyn_occupation.png      T_eff of ALL active modes vs energy (median and 16-84 % band per energy bin; one colour per
                          snapshot time; stars = pumped modes), separately for magnon-like (w_mag > 0.5) and
                          phonon-like modes; n - n_eq vs energy; pump-response traces
  dyn_occupation_map.png  where in k-space the excess sits: log10(n / n_eq) of every mode in the (|q|, energy) plane at
                          the first, a middle and the last snapshot
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_table(path):
    """Whitespace table with a header row -> {column: float array} (no pandas needed on the cluster)."""
    with open(path) as f:
        names = f.readline().split()
    data = np.atleast_2d(np.genfromtxt(path, skip_header=1))
    return {n: data[:, i] for i, n in enumerate(names)}

KB = 0.08617333262  # meV / K


def teff(eps, n):
    out = np.full(np.shape(n), np.nan)
    ok = np.asarray(n) > 1e-14
    out[ok] = eps[ok] / (KB * np.log1p(1.0 / np.asarray(n)[ok]))
    return out


def binned(x, y, edges):
    """median and 16/84 percentiles of y in bins of x (nan for empty bins)"""
    idx = np.digitize(x, edges) - 1
    med = np.full(len(edges) - 1, np.nan); lo = med.copy(); hi = med.copy(); cnt = np.zeros(len(edges) - 1, int)
    for b in range(len(edges) - 1):
        v = y[(idx == b) & np.isfinite(y)]
        cnt[b] = v.size
        if v.size:
            med[b], lo[b], hi[b] = np.median(v), np.percentile(v, 16), np.percentile(v, 84)
    return med, lo, hi, cnt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="Outputs", help="run directory (contains occupation_snapshots.npz)")
    ap.add_argument("--out", default=None, help="where to write the figures (default: --dir)")
    ap.add_argument("--nlines", type=int, default=8, help="number of snapshots drawn as lines")
    ap.add_argument("--bins", type=int, default=40, help="energy bins")
    args = ap.parse_args()
    out = args.out or args.dir

    snap_path = os.path.join(args.dir, "occupation_snapshots.npz")
    if not os.path.exists(snap_path):
        raise SystemExit(f"{snap_path} not found: run SpinPhony.py --hybrid_dynamics (--snapshots > 0) first.")
    z = np.load(snap_path, allow_pickle=False)
    t, n_all, eps = z["t_ps"], z["n"], z["eps"]
    wm, n_eq, T_eq = z["w_mag"], z["n_eq"], float(z["T_eq"])
    pump = z["pump_flat"].astype(int)
    labels = list(z["pump_labels"]) if "pump_labels" in z.files else []
    emin = float(z["min_energy"]) if "min_energy" in z.files else 0.2
    act = eps >= emin
    nq, nb = eps.shape
    mag_like = (wm > 0.5) & act
    ph_like = (wm <= 0.5) & act
    e_flat, wm_flat = eps.reshape(-1), wm.reshape(-1)

    sel = np.unique(np.round(np.linspace(0, len(t) - 1, min(args.nlines, len(t)))).astype(int))
    cmap = plt.cm.plasma
    colors = {i: cmap(k / max(1, len(sel) - 1)) for k, i in enumerate(sel)}
    edges = np.linspace(emin, float(eps[act].max()) * 1.0001, args.bins + 1)
    ecen = 0.5 * (edges[:-1] + edges[1:])

    # ----------------------------------------------------------------------------------------- summary table
    print(f" snapshots: {len(t)} (t = {t[0]:.3g} .. {t[-1]:.3g} ps),  T_eq = {T_eq:.2f} K,  {int(act.sum()):,} active modes, "
          f"{len(pump)} pumped")
    print(f" {'t/ps':>10} {'<T_eff> magnon-like':>20} {'<T_eff> phonon-like':>20} {'max T_eff (non-pumped)':>24}"
          + (f" {'n_pumped':>10}" if len(pump) else ""))
    notp = act.copy().reshape(-1); notp[pump] = False
    for i in sel:
        T = teff(eps, n_all[i])
        Em = np.sum(eps[mag_like] * n_all[i][mag_like]); Ep = np.sum(eps[ph_like] * n_all[i][ph_like])
        Tm = np.nanmean(T[mag_like]); Tp = np.nanmean(T[ph_like])
        Tmax = np.nanmax(np.where(notp.reshape(nq, nb), T, np.nan))
        print(f" {t[i]:10.4g} {Tm:20.2f} {Tp:20.2f} {Tmax:24.2f}"
              + (f" {np.mean(n_all[i].reshape(-1)[pump]):10.4g}" if len(pump) else ""))

    # ------------------------------------------------------------------------------------------ figure 1
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    for a, sub, title in ((ax[0, 0], mag_like, "magnon-like modes (w_mag > 0.5)"),
                          (ax[0, 1], ph_like, "phonon-like modes (w_mag <= 0.5)")):
        for i in sel:
            T = teff(eps, n_all[i])
            keep = sub.copy().reshape(-1); keep[pump] = False
            keep = keep.reshape(nq, nb)
            e_sub = eps[sub]
            ed = np.linspace(e_sub.min(), e_sub.max() * 1.0001, args.bins + 1)     # bins span this subset only
            ec = 0.5 * (ed[:-1] + ed[1:])
            med, lo, hi, cnt = binned(eps[keep], T[keep], ed)
            good = cnt > 0
            a.plot(ec[good], med[good], "-o", ms=3, lw=1.6, color=colors[i], label=f"t = {t[i]:.3g} ps")
            if i in (sel[0], sel[-1]):
                a.fill_between(ec[good], lo[good], hi[good], color=colors[i], alpha=0.15, lw=0)
            pm = pump[sub.reshape(-1)[pump]] if len(pump) else []
            for p in pm:
                a.plot(e_flat[p], T.reshape(-1)[p], "*", ms=15, color=colors[i], mec="k", zorder=5)
        a.axhline(T_eq, color="k", ls="--", lw=1, label=f"T_eq = {T_eq:.1f} K")
        a.set_yscale("log"); a.set_xlabel("mode energy  (meV)"); a.set_ylabel(r"$T_{\rm eff}=\varepsilon/[k_B\ln(1+1/n)]$  (K)")
        a.set_title(title + "\nmedian and 16-84 % per energy bin; stars = pumped modes")
        a.grid(alpha=0.3, which="both")
    ax[0, 0].legend(fontsize=7, ncol=2, loc="best")

    a = ax[1, 0]
    isel = [sel[0], sel[len(sel) // 2], sel[-1]]
    for k, i in enumerate(isel):
        d = (n_all[i] - n_eq)[act]
        a.scatter(eps[act], d, s=2, alpha=0.35, color=colors[i], label=f"t = {t[i]:.3g} ps", rasterized=True)
        if len(pump):
            a.plot(e_flat[pump], (n_all[i] - n_eq).reshape(-1)[pump], "*", ms=14, color=colors[i], mec="k", zorder=5)
    a.axhline(0, color="k", lw=0.8)
    a.set_yscale("symlog", linthresh=1e-3); a.set_xlabel("mode energy  (meV)"); a.set_ylabel(r"$n-n_{eq}(T_{eq})$")
    a.set_title("occupation excess of every active mode"); a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1, 1]
    obs_path = os.path.join(args.dir, "observables_dynamics.txt")
    if os.path.exists(obs_path):
        o = _read_table(obs_path)
        tt = o['Time_ps']
        a.plot(tt, (o['M_over_M0'] - 1.0) * 100.0, "k", lw=2, label=r"$\Delta M/M(0)$  (%)")
        a.set_xscale("symlog", linthresh=max(1e-4, tt[1] if len(tt) > 1 else 1e-3)); a.set_xlabel("time  (ps)")
        a.set_ylabel(r"$[M(t)/M(0)-1]\times 100$  (%)"); a.grid(alpha=0.3)
        a.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useOffset=False)
        b2 = a.twinx()
        if len(pump):
            for k, p in enumerate(pump):
                b2.plot(t, n_all[:, p // nb, p % nb], "o-", ms=4, lw=1.4, color=f"C{k + 1}",
                        label=f"n of {labels[k] if k < len(labels) else p}")
            b2.set_yscale("log"); b2.set_ylabel("occupation of the pumped mode")
        else:
            b2.plot(tt, o['T_mag_fit_K'], color="C0", lw=1.5, label=r"$T_{\rm mag}$ fit")
            b2.plot(tt, o['T_phon_fit_K'], color="C3", lw=1.5, label=r"$T_{\rm phon}$ fit")
            b2.axhline(float(o['T_eq_K'][0]), color="k", ls="--", lw=0.8); b2.set_ylabel("temperature (K)")
        h1, l1 = a.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels()
        a.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")
        a.set_title("response of the magnetization" + (" to the pump" if len(pump) else ""))
    fig.suptitle("Hybrid dynamics: mode occupation" + (f"   (pump: {', '.join(map(str, labels))})" if labels else ""), y=0.995)
    fig.tight_layout()
    f1 = os.path.join(out, "dyn_occupation.png"); fig.savefig(f1, dpi=150); plt.close(fig)

    # ------------------------------------------------------------------------------------------ figure 2: k-space map
    if "q_frac" in z.files:
        qf = z["q_frac"]; qf = qf - np.round(qf)
        qn = np.linalg.norm(qf, axis=1)
        fig, ax = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
        for a, i in zip(ax, isel):
            ratio = np.log10(np.maximum(n_all[i], 1e-30) / np.maximum(n_eq, 1e-30))
            qq = np.repeat(qn[:, None], nb, axis=1)
            sc = a.scatter(qq[act], eps[act], c=ratio[act], s=4, cmap="coolwarm", vmin=-2, vmax=2, rasterized=True)
            if len(pump):
                a.plot(qq.reshape(-1)[pump], e_flat[pump], "*", ms=16, mfc="none", mec="k", mew=1.5)
            a.set_title(f"t = {t[i]:.3g} ps"); a.set_xlabel(r"$|\mathbf{q}|$  (fractional, centred)")
        ax[0].set_ylabel("mode energy  (meV)")
        fig.colorbar(sc, ax=ax, label=r"$\log_{10}(n/n_{eq})$", shrink=0.9)
        fig.suptitle("where the excess sits (red: more populated than at T_eq; blue: depleted); stars = pumped modes")
        f2 = os.path.join(out, "dyn_occupation_map.png"); fig.savefig(f2, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f" wrote {f1}\n       {f2}")
    else:
        print(f" wrote {f1}  (no q_frac in the snapshot file: k-space map skipped)")


if __name__ == "__main__":
    main()
