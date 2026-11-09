"""
Compare a bare run without and with three-phonon scattering: dynamics and lifetimes in one figure.

    python python/plot_phph_runs.py Outputs/phph_runs/g20_off Outputs/phph_runs/g20_on [--out fig.png] [--no_show]

Top row:    thermal-fit temperatures T_mag, T_phon vs time | magnon chemical potential and energy drift | ΔT = T_mag - T_phon (log)
Bottom row: phonon scattering rate vs energy, both runs | phonon rate ratio on/off vs energy | phonon channel rates in the run with phph
Prints the final values and the exponential relaxation time of ΔT.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("off_dir"); ap.add_argument("on_dir")
ap.add_argument("--out", default=None)
ap.add_argument("--no_show", action="store_true")
args = ap.parse_args()
if args.no_show:
    matplotlib.use("Agg")

obs = {k: pd.read_csv(f"{d}/observables_dynamics.txt", sep=r"\s+") for k, d in (("off", args.off_dir), ("on", args.on_dir))}
lt = {k: pd.read_csv(f"{d}/path_lifetimes.csv", comment="#") for k, d in (("off", args.off_dir), ("on", args.on_dir))}
col = {"off": "C0", "on": "C3"}; lab = {"off": "without phph", "on": "with phph"}

fig, ax = plt.subplots(2, 3, figsize=(17, 9))
for k, o in obs.items():
    t = o.Time_ps
    ax[0, 0].plot(t, o.T_mag_fit_K, col[k], label=f"magnons, {lab[k]}")
    ax[0, 0].plot(t, o.T_phon_fit_K, col[k], ls="--", label=f"phonons, {lab[k]}")
    ax[0, 1].plot(t, o.mu_mag_meV, col[k], label=lab[k])
    dT = (o.T_mag_fit_K - o.T_phon_fit_K).abs()
    ax[0, 2].semilogy(t, dT, col[k], label=lab[k])
    # exponential relaxation time of dT over the part with dT > 5 % of its start value
    s = (dT > 0.05 * dT.iloc[0]).to_numpy() & (t > 1).to_numpy()
    tau = -1 / np.polyfit(t[s], np.log(dT[s]), 1)[0] if s.sum() > 5 else np.nan
    print(f"{lab[k]:13s}: final T_mag_fit {o.T_mag_fit_K.iloc[-1]:.1f} K, T_phon_fit {o.T_phon_fit_K.iloc[-1]:.1f} K, mu {o.mu_mag_meV.iloc[-1]:.3f} meV, "
          f"E drift {o.E_tot_meV.iloc[-1] / o.E_tot_meV.iloc[0] - 1:+.2e}, tau(dT) ≈ {tau:.1f} ps")
ax[0, 0].set_ylabel("thermal-fit T (K)"); ax[0, 1].set_ylabel(r"$\mu_{mag}$ (meV)"); ax[0, 2].set_ylabel(r"$|T_{mag}-T_{phon}|$ (K)")
ax2 = ax[0, 1].twinx()
for k, o in obs.items():
    ax2.plot(o.Time_ps, o.E_tot_meV / o.E_tot_meV.iloc[0] - 1, col[k], ls=":", alpha=0.7)
ax2.set_ylabel(r"$E_{tot}/E_{tot}(0)-1$ (dotted)")
for a in ax[0]:
    a.set_xlabel("time (ps)"); a.grid(alpha=0.3); a.legend(fontsize=8)

# lifetimes: phonons, rate vs energy
ph = {k: d[d.particle == "phonon"].reset_index(drop=True) for k, d in lt.items()}
for k, d in ph.items():
    m = d["gamma_ps-1"] > 1e-10
    ax[1, 0].scatter(d.energy_meV[m], d["gamma_ps-1"][m], s=5, color=col[k], alpha=0.4, label=lab[k], rasterized=True)
ax[1, 0].set_xscale("log"); ax[1, 0].set_yscale("log")
ax[1, 0].set_xlabel("energy (meV)"); ax[1, 0].set_ylabel(r"phonon $\gamma$ (ps$^{-1}$)"); ax[1, 0].legend(markerscale=3, fontsize=8)
a, b = ph["off"], ph["on"]
ok = (a["gamma_ps-1"] > 1e-10) & (b["gamma_ps-1"] > 1e-10) & (a.q_idx == b.q_idx) & (a.branch == b.branch)
ratio = b["gamma_ps-1"][ok] / a["gamma_ps-1"][ok]
sc = ax[1, 1].scatter(a.energy_meV[ok], ratio, s=5, c=a.branch[ok], cmap="tab10", vmin=0, vmax=9, alpha=0.5, rasterized=True)
ax[1, 1].axhline(1, color="k", lw=0.8)
ax[1, 1].set_xscale("log"); ax[1, 1].set_yscale("log")
ax[1, 1].set_xlabel("energy (meV)"); ax[1, 1].set_ylabel(r"$\gamma_{on}/\gamma_{off}$ (phonons, colour = branch)")
print(f"phonons: gamma_on/gamma_off median {np.median(ratio):.2f}; {int((ratio > 2).sum())} of {int(ok.sum())} modes more than doubled; "
      f"{int((b['gamma_phph_ps-1'] > 0).sum())} modes with a phph rate")
for key, c, l in (("gamma_exch_ps-1", "C0", "magnon-phonon"), ("gamma_phph_ps-1", "C2", "phonon-phonon")):
    m = b[key] > 1e-10
    ax[1, 2].scatter(b.energy_meV[m], b[key][m], s=5, color=c, alpha=0.4, label=l, rasterized=True)
ax[1, 2].set_xscale("log"); ax[1, 2].set_yscale("log")
ax[1, 2].set_xlabel("energy (meV)"); ax[1, 2].set_ylabel(r"phonon $\gamma$ by channel, with phph (ps$^{-1}$)"); ax[1, 2].legend(markerscale=3, fontsize=8)
for a_ in ax[1]:
    a_.grid(alpha=0.3, which="both")
fig.tight_layout()
out = args.out or f"{args.on_dir}/phph_comparison.png"
fig.savefig(out, dpi=150)
print("wrote", out)
if not args.no_show:
    plt.show()
