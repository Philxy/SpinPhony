import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, TwoSlopeNorm

# Plots the time-resolved magnon-phonon Boltzmann dynamics written by SpinPhony.py:
#
#   <run>/observables_dynamics.txt   scalar time series (T, mu, entropy, flux, ...)
#   <run>/spectral_dynamics.csv      energy-binned n, n_fit and mode temperature vs time
#   <run>/flux_dynamics.csv          magnon -> phonon energy flux by phonon branch / phonon energy
#   <run>/final_populations.npz      (optional) final n(q, branch)
#
# Optional comparison with the two-temperature model:
#   --twotm Outputs/2TM_CrI3_Dynamics.csv      columns Time(ps), T_mag(K), T_phon(K), M_over_M0
#   --gmp   <run>/G_mp_temperature_scan.csv    linear-response G_mp(T), C and tau_mp(T)
#
# Figures (written to --out_dir, default: the run folder):
#   dyn_temperatures.png   thermal-fit T_mag, T_phon (+ legacy T_eff, + 2TM), magnon mu, N_mag
#   dyn_energy_flux.png    subsystem energies, flux P(t), effective coupling G_eff(t), energy conservation
#   dyn_entropy.png        entropy production, distance to the thermal state (dS, eta)
#   dyn_spectral.png       mode temperature and n/n_fit maps over (time, energy), both subsystems
#   dyn_flux_spectrum.png  who carries the energy: flux by phonon energy and branch
#   dyn_relaxation.png     Delta T = T_mag - T_phon: exponential fit vs linear-response tau and 2TM
#   dynamics_summary.txt   numbers quoted from these plots


def load_obs(run):
    path = os.path.join(run, "observables_dynamics.txt")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found")
    return pd.read_csv(path, sep=r"\s+")


def maybe(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def tfmt(ax, logt):
    ax.set_xlabel("Time (ps)")
    ax.grid(True, ls=":", alpha=0.5)
    if logt:
        ax.set_xscale("log")


def fit_decay(t, dT, window=(0.8, 0.1)):
    """tau from ln(dT) = a - t/tau between the times where dT has dropped to `window[0]` and `window[1]` of dT(0)."""
    ok = np.isfinite(dT) & (np.abs(dT) > 0)
    t, dT = t[ok], dT[ok]
    if len(t) < 5 or dT[0] == 0:
        return float("nan"), None
    frac = dT / dT[0]
    sel = (frac < window[0]) & (frac > window[1])
    if sel.sum() < 4:
        return float("nan"), None
    slope, icpt = np.polyfit(t[sel], np.log(np.abs(dT[sel])), 1)
    return (-1.0 / slope if slope < 0 else float("nan")), (slope, icpt, t[sel][[0, -1]])


def main():
    ap = argparse.ArgumentParser(description="Plot the magnon-phonon Boltzmann dynamics")
    ap.add_argument("run_dir")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--twotm", default=None, help="2TM csv: Time(ps), T_mag(K), T_phon(K), M_over_M0")
    ap.add_argument("--gmp", default=None, help="G_mp_temperature_scan.csv for the linear-response tau_mp")
    ap.add_argument("--T_ref", type=float, default=None, help="temperature for the tau_mp lookup (default: final T)")
    ap.add_argument("--logt", action="store_true", help="logarithmic time axis")
    ap.add_argument("--tmax", type=float, default=None, help="restrict plots to t <= tmax (ps)")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    run = a.run_dir
    out = a.out_dir or run
    os.makedirs(out, exist_ok=True)

    obs = load_obs(run)
    new_cols = "T_mag_fit_K" in obs.columns
    spec = maybe(os.path.join(run, "spectral_dynamics.csv"))
    flux = maybe(os.path.join(run, "flux_dynamics.csv"))
    tm = maybe(a.twotm) if a.twotm else None
    gm = maybe(a.gmp) if a.gmp else None
    if a.tmax is not None:
        obs = obs[obs.Time_ps <= a.tmax]
        if spec is not None:
            spec = spec[spec.time_ps <= a.tmax]
        if flux is not None:
            flux = flux[flux.time_ps <= a.tmax]
        if tm is not None:
            tm = tm[tm["Time(ps)"] <= a.tmax]
    t = obs.Time_ps.to_numpy()
    summary = []

    def save(fig, name):
        fig.tight_layout()
        p = os.path.join(out, name)
        fig.savefig(p, dpi=250)
        print(f" -> {p}")
        if a.show:
            plt.show()
        plt.close(fig)

    # ---------------------------------------------------------------- temperatures
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.5, 8), sharex=True)
    if new_cols:
        ax.plot(t, obs.T_mag_fit_K, color="#d62728", lw=2, label=r"$T_{mag}$ (thermal fit, $N$,$E$)")
        ax.plot(t, obs.T_phon_fit_K, color="#1f77b4", lw=2, label=r"$T_{phon}$ (thermal fit, $E$)")
    ax.plot(t, obs.T_eff_mag_K, color="#d62728", lw=1, ls="--", alpha=0.6, label=r"$T_{eff}$ mag (assumes $\mu=0$)")
    ax.plot(t, obs.T_eff_phon_K, color="#1f77b4", lw=1, ls="--", alpha=0.6, label=r"$T_{eff}$ phon (assumes $\mu=0$)")
    if tm is not None:
        ax.plot(tm["Time(ps)"], tm["T_mag(K)"], color="#d62728", lw=1.5, ls=":", label="2TM magnons")
        ax.plot(tm["Time(ps)"], tm["T_phon(K)"], color="#1f77b4", lw=1.5, ls=":", label="2TM phonons")
    ax.set_ylabel("Temperature (K)")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Subsystem temperatures")
    tfmt(ax, a.logt)

    if new_cols:
        ax2.plot(t, obs.mu_mag_meV, color="#9467bd", lw=2)
        ax2.set_ylabel(r"magnon chemical potential $\mu_{mag}$ (meV)", color="#9467bd")
        ax3 = ax2.twinx()
        ax3.plot(t, obs.N_mag / obs.N_mag.iloc[0] - 1.0, color="k", lw=1.2, ls="--")
        ax3.set_ylabel(r"$N_{mag}/N_{mag}(0) - 1$  (magnetization change)")
        ax3.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax2.set_title(r"Conserved magnon number $\Rightarrow$ $\mu_{mag} \neq 0$")
    tfmt(ax2, a.logt)
    save(fig, "dyn_temperatures.png")

    if tm is not None and "M_over_M0" in tm.columns:
        summary.append(f"2TM M/M0 at t_end: {tm['M_over_M0'].iloc[-1]:.5f}   Boltzmann N_mag/N_mag(0): "
                       f"{obs.N_mag.iloc[-1] / obs.N_mag.iloc[0]:.6f}")

    # ---------------------------------------------------------------- energy / flux
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    E0 = obs.E_tot_meV.iloc[0]
    ax = axs[0, 0]
    ax.plot(t, 100 * (obs.E_mag_meV - obs.E_mag_meV.iloc[0]) / E0, color="#d62728", lw=2, label="magnons")
    ax.plot(t, 100 * (obs.E_phon_meV - obs.E_phon_meV.iloc[0]) / E0, color="#1f77b4", lw=2, label="phonons")
    ax.plot(t, 100 * (obs.E_tot_meV - E0) / E0, color="k", lw=1, ls="--", label="total")
    ax.set_ylabel(r"$\Delta E / E_{tot}(0)$ (%)")
    ax.legend(fontsize=9)
    ax.set_title("Energy redistribution")
    tfmt(ax, a.logt)

    ax = axs[0, 1]
    if new_cols:
        ax.plot(t, obs.P_mp_cell, color="#2ca02c", lw=2)
        ax.set_ylabel(r"$P_{m\to p} = -dE_{mag}/dt$ (meV ps$^{-1}$ / cell)")
        ax.set_yscale("symlog", linthresh=max(1e-12, np.nanmax(np.abs(obs.P_mp_cell)) * 1e-4))
    ax.set_title("Magnon $\\to$ phonon energy flux")
    tfmt(ax, a.logt)

    ax = axs[1, 0]
    if new_cols:
        ax.plot(t, obs.G_eff_cell, color="#8c564b", lw=2, label=r"$G_{eff}(t) = P/(T_{mag}-T_{phon})$")
        if gm is not None:
            Tref = a.T_ref if a.T_ref is not None else float(obs.T_phon_fit_K.iloc[-1])
            row = gm.iloc[(gm.Temperature_K - Tref).abs().argmin()]
            ax.axhline(row.G_mp_meV_per_K_ps_per_cell, color="k", ls="--",
                       label=f"2TM linear $G_{{mp}}$ ({row.Temperature_K:.0f} K)")
            summary.append(f"2TM linear G_mp({row.Temperature_K:.0f} K) = {row.G_mp_meV_per_K_ps_per_cell:.4e} meV/(K ps)/cell")
        ax.set_ylabel(r"$G_{eff}$ (meV K$^{-1}$ ps$^{-1}$ / cell)")
        ax.legend(fontsize=9)
        g = obs.G_eff_cell.replace([np.inf, -np.inf], np.nan).dropna()
        if len(g) > 4:
            summary.append(f"G_eff: first-quarter median {g.iloc[:max(1, len(g)//4)].median():.4e}, "
                           f"last-quarter median {g.iloc[-max(1, len(g)//4):].median():.4e} meV/(K ps)/cell")
    ax.set_title("Effective coupling (constant for a valid 2TM)")
    tfmt(ax, a.logt)

    ax = axs[1, 1]
    ax.plot(t, (obs.E_tot_meV - E0) / E0, color="k", lw=1.5, label=r"$(E_{tot}-E_0)/E_0$")
    if new_cols:
        Pmax = max(float(np.nanmax(np.abs(obs.P_mp_cell))), 1e-300)
        rel = obs.dEtot_rate_cell / Pmax
        ax.plot(t, rel, color="#e377c2", lw=1.2, label=r"$d E_{tot}/dt\ /\ \max|P|$ (broadening error)")
        summary.append(f"energy: max |E_tot-E_0|/E_0 = {np.nanmax(np.abs((obs.E_tot_meV - E0) / E0)):.3e}; "
                       f"max |dE_tot/dt| / max|P| = {np.nanmax(np.abs(rel)):.3e}")
    ax.set_yscale("symlog", linthresh=1e-9)
    ax.legend(fontsize=9)
    ax.set_title("Energy conservation check")
    tfmt(ax, a.logt)
    save(fig, "dyn_energy_flux.png")

    # ---------------------------------------------------------------- entropy / non-thermality
    if new_cols:
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
        ax = axs[0]
        dS = obs.S_tot - obs.S_tot.iloc[0]
        ax.plot(t, dS, color="k", lw=2)
        ax.set_ylabel(r"$S_{tot}(t) - S_{tot}(0)$  ($k_B$, summed over modes)")
        ax.set_title("Entropy production (monotonic)")
        tfmt(ax, a.logt)
        n_dec = int((np.diff(obs.S_tot.to_numpy()) < -1e-9 * np.abs(obs.S_tot.to_numpy()[:-1])).sum())
        summary.append(f"entropy: S_tot(end)-S_tot(0) = {dS.iloc[-1]:.5e} k_B; decreasing output intervals: {n_dec} "
                       f"(H-theorem: should be 0 up to broadening/Euler error)")

        ax = axs[1]
        ax.plot(t, obs.dS_mag, color="#d62728", lw=2, label="magnons")
        ax.plot(t, obs.dS_phon, color="#1f77b4", lw=2, label="phonons")
        ax.set_yscale("log")
        ax.set_ylabel(r"$S[n_{fit}] - S[n]$  ($k_B$)")
        ax.legend()
        ax.set_title("Entropy still to gain inside each subsystem")
        tfmt(ax, a.logt)

        ax = axs[2]
        ax.plot(t, obs.eta_mag, color="#d62728", lw=2, label="magnons")
        ax.plot(t, obs.eta_phon, color="#1f77b4", lw=2, label="phonons")
        ax.set_yscale("log")
        ax.set_ylabel(r"$\eta = \sum \omega\,|n-n_{fit}| / \sum \omega\, n$")
        ax.legend()
        ax.set_title("Energy-weighted non-thermal fraction")
        tfmt(ax, a.logt)
        save(fig, "dyn_entropy.png")
        summary.append(f"non-thermal fraction eta: magnons max {np.nanmax(obs.eta_mag):.3e} (end {obs.eta_mag.iloc[-1]:.3e}), "
                       f"phonons max {np.nanmax(obs.eta_phon):.3e} (end {obs.eta_phon.iloc[-1]:.3e})")

    # ---------------------------------------------------------------- spectral maps
    if spec is not None and len(spec) > 0:
        fig, axs = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
        for row, sub in enumerate(["magnon", "phonon"]):
            d = spec[spec.subsystem == sub]
            if d.empty:
                continue
            times = np.sort(d.time_ps.unique())
            if a.logt:
                times = times[times > 0]
                d = d[d.time_ps > 0]
            ebins = d.groupby(d.groupby("time_ps").cumcount()).E_meV.mean().to_numpy()
            Tm = np.full((len(ebins), len(times)), np.nan)
            ratio = np.full_like(Tm, np.nan)
            for j, tt in enumerate(times):
                dd = d[d.time_ps == tt].reset_index(drop=True)
                Tm[:len(dd), j] = dd.T_mode_K.replace(0, np.nan).to_numpy()
                ratio[:len(dd), j] = (dd.n_mean / dd.n_fit_mean.replace(0, np.nan)).to_numpy()
            tx = np.append(times, times[-1] * (1 + 1e-6) + 1e-12) if len(times) == 1 else \
                np.append(times, times[-1] + (times[-1] - times[-2]))
            ey = np.append(ebins, ebins[-1] + (ebins[-1] - ebins[-2] if len(ebins) > 1 else 1.0))
            ax = axs[row, 0]
            pc = ax.pcolormesh(tx, ey, Tm, cmap="inferno", shading="flat")
            fig.colorbar(pc, ax=ax).set_label(r"mode temperature $\omega/(k_B\ln(1+1/n))$ (K)")
            ax.set_ylabel(f"{sub} energy (meV)")
            ax.set_title(f"{sub.capitalize()}s: mode temperature")
            ax = axs[row, 1]
            r = np.where(np.isfinite(ratio) & (ratio > 0), ratio, np.nan)
            lim = float(np.nanmax(np.abs(np.log(r)))) if np.isfinite(r).any() else 0.1
            lim = max(lim, 0.02)
            pc = ax.pcolormesh(tx, ey, r, cmap="RdBu_r", shading="flat",
                               norm=TwoSlopeNorm(vcenter=1.0, vmin=np.exp(-lim), vmax=np.exp(lim)))
            fig.colorbar(pc, ax=ax).set_label(r"$n / n_{fit}$  (1 = thermal)")
            ax.set_title(f"{sub.capitalize()}s: deviation from thermal fit")
        for ax in axs[-1]:
            tfmt(ax, a.logt)
        save(fig, "dyn_spectral.png")

    # ---------------------------------------------------------------- flux spectrum
    if flux is not None and len(flux) > 0:
        fig, axs = plt.subplots(1, 3, figsize=(17, 4.9))
        fe = flux[flux.kind == "energy"]
        times = np.sort(fe.time_ps.unique())
        if len(times) > 0:
            pick = np.unique(np.linspace(0, len(times) - 1, min(6, len(times))).astype(int))
            cols = plt.cm.viridis(np.linspace(0, 0.95, len(pick)))
            ax = axs[0]
            for c, i in zip(cols, pick):
                dd = fe[fe.time_ps == times[i]]
                ax.plot(dd.E_meV, dd.P_meV_per_ps_cell, color=c, lw=1.6, label=f"t = {times[i]:.3g} ps")
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xlabel("phonon energy (meV)")
            ax.set_ylabel(r"$P(\omega_p)$ (meV ps$^{-1}$ / cell / bin)")
            ax.set_title("Spectral magnon $\\to$ phonon flux")
            ax.legend(fontsize=8)
            ax.grid(True, ls=":", alpha=0.5)

            ax = axs[1]
            totals = fe.groupby("time_ps").P_meV_per_ps_cell.sum()
            tpk = totals.abs().idxmax()
            dd = fe[fe.time_ps == tpk].sort_values("E_meV")
            ax.plot(dd.E_meV, dd.P_meV_per_ps_cell.cumsum() / dd.P_meV_per_ps_cell.sum(), color="k", lw=2)
            ax.set_xlabel("phonon energy (meV)")
            ax.set_ylabel(r"cumulative fraction of $P$")
            ax.set_title(f"Where the flux goes (t = {tpk:.3g} ps, peak flux)")
            ax.grid(True, ls=":", alpha=0.5)
            half = dd.E_meV[(dd.P_meV_per_ps_cell.cumsum() / dd.P_meV_per_ps_cell.sum()) >= 0.5]
            if len(half):
                summary.append(f"spectral flux at peak (t={tpk:.3g} ps): half of P carried by phonons below {half.iloc[0]:.2f} meV")

        fb = flux[flux.kind == "branch"]
        tb = np.sort(fb.time_ps.unique())
        if len(tb) > 1:
            piv = fb.pivot(index="time_ps", columns="index", values="P_meV_per_ps_cell").sort_index()
            integ = np.trapezoid(piv.to_numpy(), piv.index.to_numpy(), axis=0) if hasattr(np, "trapezoid") \
                else np.trapz(piv.to_numpy(), piv.index.to_numpy(), axis=0)
            Ebr = fb.groupby("index").E_meV.first().reindex(piv.columns).to_numpy()
            ax = axs[2]
            ax.bar(piv.columns.to_numpy(), integ, color=plt.cm.plasma(Ebr / max(Ebr.max(), 1e-9)))
            ax.set_xlabel("phonon branch (colour: mean energy)")
            ax.set_ylabel(r"$\int P\, dt$ (meV / cell)")
            ax.set_title("Energy delivered to each phonon branch")
            ax.grid(True, ls=":", alpha=0.5, axis="y")
            top = np.argsort(integ)[::-1][:3]
            summary.append("energy delivered per branch (top 3): " +
                           ", ".join(f"branch {int(piv.columns[i])} ({Ebr[i]:.1f} meV): {100 * integ[i] / integ.sum():.1f}%" for i in top))
        save(fig, "dyn_flux_spectrum.png")

    # ---------------------------------------------------------------- relaxation vs 2TM / linear response
    if new_cols:
        dT = (obs.T_mag_fit_K - obs.T_phon_fit_K).to_numpy()
        tau_fit, info = fit_decay(t, dT)
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        ax.plot(t, np.abs(dT), color="k", lw=2, label=r"Boltzmann $|T_{mag}-T_{phon}|$")
        if info is not None:
            slope, icpt, (t0, t1) = info
            tt = np.linspace(t[0], t[-1], 200)
            ax.plot(tt, np.exp(icpt + slope * tt), color="#d62728", ls="--", label=fr"exp. fit, $\tau$ = {tau_fit:.4g} ps")
            ax.axvspan(t0, t1, color="#d62728", alpha=0.08)
            summary.append(f"relaxation time from exponential fit of T_mag-T_phon: tau = {tau_fit:.5g} ps")
        if tm is not None:
            ax.plot(tm["Time(ps)"], np.abs(tm["T_mag(K)"] - tm["T_phon(K)"]), color="#1f77b4", ls=":", lw=2, label="2TM")
            tau2, _ = fit_decay(tm["Time(ps)"].to_numpy(), (tm["T_mag(K)"] - tm["T_phon(K)"]).to_numpy())
            summary.append(f"2TM relaxation time (same fit): tau = {tau2:.5g} ps")
        if gm is not None:
            Tref = a.T_ref if a.T_ref is not None else float(obs.T_phon_fit_K.iloc[-1])
            row = gm.iloc[(gm.Temperature_K - Tref).abs().argmin()]
            ax.plot(t, np.abs(dT[0]) * np.exp(-t / row.tau_ps), color="#2ca02c", ls="-.",
                    label=fr"linear response $\tau_{{mp}}$({row.Temperature_K:.0f} K) = {row.tau_ps:.4g} ps")
            summary.append(f"linear-response tau_mp({row.Temperature_K:.0f} K) = {row.tau_ps:.5g} ps")
        ax.set_yscale("log")
        ax.set_ylabel("temperature difference (K)")
        ax.legend(fontsize=9)
        ax.set_title("Magnon-phonon temperature relaxation")
        tfmt(ax, a.logt)
        save(fig, "dyn_relaxation.png")

        summary.append(f"final: T_mag_fit = {obs.T_mag_fit_K.iloc[-1]:.3f} K, T_phon_fit = {obs.T_phon_fit_K.iloc[-1]:.3f} K, "
                       f"mu_mag = {obs.mu_mag_meV.iloc[-1]:.4f} meV (initial T_mag/T_phon = "
                       f"{obs.T_mag_fit_K.iloc[0]:.1f}/{obs.T_phon_fit_K.iloc[0]:.1f} K)")

    with open(os.path.join(out, "dynamics_summary.txt"), "w") as f:
        f.write("\n".join(summary) + "\n")
    print("\n" + "\n".join(summary))


if __name__ == "__main__":
    main()
