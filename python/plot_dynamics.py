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
# No G_mp table is needed. The coupling is read off the dynamics itself (G_eff(t) = P / (T_mag - T_phon)).
# Optionally assume a constant coupling and compare with the linear two-temperature model:
#   --gmp_value G     constant G_mp in meV/(K ps)/cell. The 2TM with this G and constant heat capacities has the
#                     exact solution  dT = dT0 exp(-t G (1/C_mag + 1/C_phon)),  energy conservation fixes T_final.
#   --c_mag, --c_phon heat capacities in meV/(K cell); default: secant estimate dE/dT_fit from the run itself
#   --n_cells N       number of q-points (default: read from final_populations.npz)
# and/or overlay an external 2TM trajectory:
#   --twotm file.csv  columns Time(ps), T_mag(K), T_phon(K), M_over_M0
#
# Figures (written to --out_dir, default: the run folder):
#   dyn_temperatures.png   thermal-fit T_mag, T_phon (+ legacy T_eff, + 2TM), magnon mu, N_mag
#   dyn_energy_flux.png    subsystem energies, flux P(t), effective coupling G_eff(t), energy conservation
#   dyn_entropy.png        entropy production, distance to the thermal state (dS, eta)
#   dyn_spectral.png       mode temperature and n/n_fit maps over (time, energy), both subsystems
#   dyn_flux_spectrum.png  who carries the energy: flux by phonon energy and branch
#   dyn_relaxation.png     Delta T = T_mag - T_phon: exponential fit vs the constant-G 2TM (and external 2TM)
#   dyn_angular_momentum.png   (hybrid dynamics only, SpinPhony.py --hybrid_dynamics) M/M(T_eq), spin / phonon AM
#                          and their rates, entropy production rate
#
# Hybrid-dynamics output (columns Jz_cell, ...) is detected automatically: T_mag/T_phon are then the character-weighted
# effective temperatures of the theory, the "thermal" reference is the common equilibrium T_eq, mu_mag is 0.
#   dynamics_summary.txt   numbers quoted from these plots


def load_obs(run):
    path = os.path.join(run, "observables_dynamics.txt")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found")
    return pd.read_csv(path, sep=r"\s+")


def two_temp_const_G(t, Tm0, Tp0, G, Cm, Cp):
    """Exact linear 2TM: constant G and heat capacities. Returns T_mag(t), T_phon(t), tau."""
    lam = G * (1.0 / Cm + 1.0 / Cp)
    dT = (Tm0 - Tp0) * np.exp(-lam * t)
    Tf = (Cm * Tm0 + Cp * Tp0) / (Cm + Cp)
    return Tf + Cp / (Cm + Cp) * dT, Tf - Cm / (Cm + Cp) * dT, 1.0 / lam


def maybe(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def tfmt(ax, logt):
    ax.set_xlabel("Time (ps)")
    ax.grid(True, ls=":", alpha=0.5)
    if logt:
        ax.set_xscale("log")


def fit_offset_exp(t, dT):
    """
    dT(t) = A exp(-(t - t0)/tau) + dT_inf, fitted to the SIGNED difference over the whole run (grid search in tau,
    linear least squares for A and dT_inf). dT_inf != 0 means the two fitted temperatures do not meet, which is
    a diagnostic in itself (energy non-conservation, frozen modes, non-thermal end state).
    """
    ok = np.isfinite(dT)
    t, dT = t[ok], dT[ok]
    if len(t) < 6 or abs(dT[0]) < 1e-9:
        return None
    span = t[-1] - t[0]
    best = None
    for tau in np.geomspace(span * 1e-3, span * 1e2, 400):
        X = np.c_[np.exp(-(t - t[0]) / tau), np.ones_like(t)]
        coef = np.linalg.lstsq(X, dT, rcond=None)[0]
        r = float(np.sqrt(np.mean((X @ coef - dT) ** 2)))
        if best is None or r < best[0]:
            best = (r, tau, coef)
    r, tau, (A, Dinf) = best
    return dict(tau=tau, A=A, dinf=Dinf, rms=r / abs(dT[0]), t0=t[0],
                at_bound=tau > 50 * span or tau < 2e-3 * span,
                fn=lambda tt: A * np.exp(-(tt - t[0]) / tau) + Dinf)


def decay_rate(t, dT, smooth=5, floor=0.01):
    """Instantaneous rate lambda(t) = -d ln|dT| / dt (a single exponential gives a constant)."""
    ad = np.abs(dT)
    ok = np.isfinite(ad) & (ad > floor * ad[0])
    if ok.sum() < smooth + 3:
        return t[:0], dT[:0]
    k = np.ones(smooth) / smooth
    ln = np.convolve(np.log(ad[ok]), k, mode="valid")
    tt = np.convolve(t[ok], k, mode="valid")
    return tt[1:-1], -np.gradient(ln, tt)[1:-1]


def main():
    ap = argparse.ArgumentParser(description="Plot the magnon-phonon Boltzmann dynamics")
    ap.add_argument("run_dir")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--twotm", default=None, help="2TM csv: Time(ps), T_mag(K), T_phon(K), M_over_M0")
    ap.add_argument("--gmp_value", type=float, default=None,
                    help="assumed constant G_mp in meV/(K ps)/cell for the linear 2TM comparison")
    ap.add_argument("--c_mag", type=float, default=None, help="magnon heat capacity, meV/(K cell) (default: from the run)")
    ap.add_argument("--c_phon", type=float, default=None, help="phonon heat capacity, meV/(K cell) (default: from the run)")
    ap.add_argument("--n_cells", type=int, default=None, help="number of q-points (default: from final_populations.npz)")
    ap.add_argument("--logt", action="store_true", help="logarithmic time axis")
    ap.add_argument("--tmax", type=float, default=None, help="restrict plots to t <= tmax (ps)")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    run = a.run_dir
    out = a.out_dir or run
    os.makedirs(out, exist_ok=True)

    obs = load_obs(run)
    new_cols = "T_mag_fit_K" in obs.columns
    hyb = "Jz_cell" in obs.columns
    spec = maybe(os.path.join(run, "spectral_dynamics.csv"))
    flux = maybe(os.path.join(run, "flux_dynamics.csv"))
    tm = maybe(a.twotm) if a.twotm else None
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

    # ---- constant-G linear 2TM (optional): needs per-cell heat capacities
    G0 = a.gmp_value
    twotm_const = None
    if G0 is not None and new_cols:
        n_cells = a.n_cells
        npz = os.path.join(run, "final_populations.npz")
        if n_cells is None and os.path.exists(npz):
            n_cells = int(np.load(npz)["n_mag"].shape[0])
        Cm, Cp = a.c_mag, a.c_phon
        if (Cm is None or Cp is None) and n_cells is None:
            print("WARNING: --gmp_value needs heat capacities: give --c_mag/--c_phon or --n_cells; "
                  "constant-G 2TM curves are skipped")
        else:
            dTm = obs.T_mag_fit_K.iloc[-1] - obs.T_mag_fit_K.iloc[0]
            dTp = obs.T_phon_fit_K.iloc[-1] - obs.T_phon_fit_K.iloc[0]
            if Cm is None:
                Cm = (obs.E_mag_meV.iloc[-1] - obs.E_mag_meV.iloc[0]) / n_cells / dTm if abs(dTm) > 1e-9 else float("nan")
            if Cp is None:
                Cp = (obs.E_phon_meV.iloc[-1] - obs.E_phon_meV.iloc[0]) / n_cells / dTp if abs(dTp) > 1e-9 else float("nan")
            if np.isfinite(Cm) and np.isfinite(Cp) and Cm > 0 and Cp > 0:
                Tm_c, Tp_c, tau_c = two_temp_const_G(t, obs.T_mag_fit_K.iloc[0], obs.T_phon_fit_K.iloc[0], G0, Cm, Cp)
                twotm_const = dict(Tm=Tm_c, Tp=Tp_c, tau=tau_c, Cm=Cm, Cp=Cp)
                src = "given" if (a.c_mag is not None and a.c_phon is not None) else "secant estimate from the run"
                summary.append(f"constant-G 2TM: G = {G0:.4e} meV/(K ps)/cell, C_mag = {Cm:.4e}, C_phon = {Cp:.4e} "
                               f"meV/(K cell) ({src}) -> tau = {tau_c:.5g} ps, T_final = {Tm_c[-1]:.2f} K (t_end)")
            else:
                print("WARNING: heat capacities from the run are not positive/finite; give --c_mag and --c_phon")

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
    if twotm_const is not None:
        ax.plot(t, twotm_const["Tm"], color="#d62728", lw=1.6, ls="-.", label=f"linear 2TM, G = {G0:.3g}")
        ax.plot(t, twotm_const["Tp"], color="#1f77b4", lw=1.6, ls="-.")
    ax.set_ylabel("Temperature (K)")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Subsystem temperatures")
    tfmt(ax, a.logt)

    if new_cols:
        ax2.plot(t, obs.mu_mag_meV, color="#9467bd", lw=2)
        ax2.set_ylabel(r"magnon chemical potential $\mu_{mag}$ (meV)", color="#9467bd")
        ax3 = ax2.twinx()
        ax3.plot(t, obs.N_mag / obs.N_mag.iloc[0] - 1.0, color="k", lw=1.2, ls="--")
        ax3.set_ylabel(r"$N_{mag}/N_{mag}(0) - 1$  (magnetization change)" if not hyb
                       else r"$\sum w_{mag} n\,/\,$initial $- 1$  (magnon content)")
        ax3.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax2.set_title(r"Polariton number not conserved: $\mu = 0$, magnon content can change" if hyb
                  else r"Conserved magnon number $\Rightarrow$ $\mu_{mag} \neq 0$")
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
        ax.plot(t, obs.P_mp_cell, color="#2ca02c", lw=2, label="Boltzmann P")
        if G0 is not None:
            ax.plot(t, G0 * (obs.T_mag_fit_K - obs.T_phon_fit_K), color="k", lw=1.3, ls="--",
                    label=r"$G\,(T_{mag}-T_{phon})$, constant $G$")
            ax.legend(fontsize=9)
        ax.set_ylabel(r"$P_{m\to p} = -dE_{mag}/dt$ (meV ps$^{-1}$ / cell)")
        ax.set_yscale("symlog", linthresh=max(1e-12, np.nanmax(np.abs(obs.P_mp_cell)) * 1e-4))
    ax.set_title("Magnon $\\to$ phonon energy flux")
    tfmt(ax, a.logt)

    ax = axs[1, 0]
    if new_cols:
        ax.plot(t, obs.G_eff_cell, color="#8c564b", lw=2, label=r"$G_{eff}(t) = P/(T_{mag}-T_{phon})$")
        if G0 is not None:
            ax.axhline(G0, color="k", ls="--", label=f"assumed constant $G$ = {G0:.3g}")
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

    # ---------------------------------------------------------------- hybrid: M, S^z, L^z, J^z
    if hyb:
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        ax = axs[0, 0]
        ax.plot(t, obs.M_over_Meq, color="k", lw=2, label=r"hybrid Boltzmann $M(t)/M(T_{eq})$")
        if tm is not None and "M_over_M0" in tm.columns:
            ax.plot(tm["Time(ps)"], tm["M_over_M0"], color="#1f77b4", ls=":", lw=2, label="external 2TM $M/M_0$ (own normalisation)")
        ax.axhline(1.0, color="0.5", lw=0.8)
        ax.set_ylabel(r"$M / M(T_{eq})$")
        ax.legend(fontsize=8)
        ax.set_title("Magnetization from the magnon content of the polaritons")
        tfmt(ax, a.logt)
        summary.append(f"M/M(T_eq): {obs.M_over_Meq.iloc[0]:.5f} -> {obs.M_over_Meq.iloc[-1]:.5f}")

        ax = axs[0, 1]
        for col, lab, c in (("Sz_cell", r"$S^z$", "#d62728"), ("Lz_cell", r"$L^z$ (phonon)", "#1f77b4"), ("Jz_cell", r"$J^z = S^z + L^z$", "k")):
            ax.plot(t, obs[col] - obs[col].iloc[0], color=c, lw=2 if col != "Jz_cell" else 1.5,
                    ls="-" if col != "Jz_cell" else "--", label=lab)
        ax.set_ylabel(r"change of angular momentum per cell ($\hbar$)")
        ax.legend(fontsize=9)
        ax.set_title("Spin and phonon angular momentum")
        tfmt(ax, a.logt)
        summary.append(f"angular momentum per cell (hbar): dS^z = {obs.Sz_cell.iloc[-1] - obs.Sz_cell.iloc[0]:+.4e}, "
                       f"dL^z = {obs.Lz_cell.iloc[-1] - obs.Lz_cell.iloc[0]:+.4e}, dJ^z = {obs.Jz_cell.iloc[-1] - obs.Jz_cell.iloc[0]:+.4e}")

        ax = axs[1, 0]
        for col, lab, c in (("dSz_rate_cell", r"$\dot S^z$", "#d62728"), ("dLz_rate_cell", r"$\dot L^z$", "#1f77b4"), ("dJz_rate_cell", r"$\dot J^z$", "k")):
            ax.plot(t, obs[col], color=c, lw=1.8, ls="-" if col != "dJz_rate_cell" else "--", label=lab)
        ax.set_yscale("symlog", linthresh=max(1e-12, float(np.nanmax(np.abs(obs.dSz_rate_cell))) * 1e-4))
        ax.set_ylabel(r"rate per cell ($\hbar$ ps$^{-1}$)")
        ax.legend(fontsize=9)
        ax.set_title(r"Angular-momentum exchange ($\dot J^z \neq 0$: hybridization does not conserve $J^z$)")
        tfmt(ax, a.logt)

        ax = axs[1, 1]
        ax.plot(t, obs.dSdt_cell, color="#2ca02c", lw=2)
        ax.set_yscale("symlog", linthresh=max(1e-12, float(np.nanmax(np.abs(obs.dSdt_cell))) * 1e-4))
        ax.set_ylabel(r"entropy production per cell ($k_B$ ps$^{-1}$)")
        ax.set_title(r"Entropy production rate ($\geq 0$)")
        tfmt(ax, a.logt)
        neg = int((obs.dSdt_cell < -1e-12 * np.nanmax(np.abs(obs.dSdt_cell))).sum())
        summary.append(f"entropy production rate negative at {neg} of {len(obs)} outputs (H-theorem: 0 up to broadening)")
        save(fig, "dyn_angular_momentum.png")

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
        ax.set_ylabel(r"$S[n_{eq}] - S[n]$  ($k_B$)" if hyb else r"$S[n_{fit}] - S[n]$  ($k_B$)")
        ax.legend()
        ax.set_title(r"Entropy still to gain towards $T_{eq}$ (character split)" if hyb
                     else "Entropy still to gain inside each subsystem")
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
            fig.colorbar(pc, ax=ax).set_label(r"$n / n_{BE}(T_{eq})$" if hyb else r"$n / n_{fit}$  (1 = thermal)")
            ax.set_title(f"{sub.capitalize()}-like modes: deviation from $n_{{BE}}(T_{{eq}})$" if hyb
                         else f"{sub.capitalize()}s: deviation from thermal fit")
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
            ax.set_xlabel("parent energy (meV)" if hyb else "phonon energy (meV)")
            ax.set_ylabel(r"$P(\omega)$ (meV ps$^{-1}$ / cell / bin)")
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
            ax.set_xlabel("parent hybrid band (colour: mean energy)" if hyb else "phonon branch (colour: mean energy)")
            ax.set_ylabel(r"$\int P\, dt$ (meV / cell)")
            ax.set_title("Magnon-to-phonon energy by parent band" if hyb else "Energy delivered to each phonon branch")
            ax.grid(True, ls=":", alpha=0.5, axis="y")
            top = np.argsort(integ)[::-1][:3]
            summary.append("energy delivered per branch (top 3): " +
                           ", ".join(f"branch {int(piv.columns[i])} ({Ebr[i]:.1f} meV): {100 * integ[i] / integ.sum():.1f}%" for i in top))
        save(fig, "dyn_flux_spectrum.png")

    # ---------------------------------------------------------------- relaxation vs 2TM / constant G
    if new_cols:
        dT = (obs.T_mag_fit_K - obs.T_phon_fit_K).to_numpy()
        fit = fit_offset_exp(t, dT)
        fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.5, 9), sharex=True)
        ax.plot(t, np.abs(dT), color="k", lw=2, label=r"Boltzmann $|T_{mag}-T_{phon}|$")
        crosses = int((np.diff(np.sign(dT)) != 0).sum())
        if crosses:
            summary.append(f"WARNING: T_mag_fit - T_phon_fit changes sign {crosses}x (the log plot shows |dT|)")
        if fit is not None:
            ax.plot(t, np.abs(fit["fn"](t)), color="#d62728", ls="--",
                    label=fr"fit $A e^{{-t/\tau}} + \Delta T_\infty$: $\tau$ = {fit['tau']:.4g} ps, "
                          fr"$\Delta T_\infty$ = {fit['dinf']:.3g} K")
            ax.axhline(abs(fit["dinf"]), color="#d62728", lw=0.8, ls=":")
            summary.append(f"relaxation fit dT = A exp(-t/tau) + dT_inf over the whole run: tau = {fit['tau']:.5g} ps, "
                           f"dT_inf = {fit['dinf']:.4g} K, rms residual = {100 * fit['rms']:.2f}% of dT(0)"
                           + ("  [tau at the edge of the search range: no clean exponential]" if fit["at_bound"] else ""))
        if tm is not None:
            ax.plot(tm["Time(ps)"], np.abs(tm["T_mag(K)"] - tm["T_phon(K)"]), color="#1f77b4", ls=":", lw=2, label="external 2TM")
        if twotm_const is not None:
            ax.plot(t, np.abs(twotm_const["Tm"] - twotm_const["Tp"]), color="#2ca02c", ls="-.", lw=2,
                    label=fr"linear 2TM, constant $G$ = {G0:.3g}: $\tau$ = {twotm_const['tau']:.4g} ps")
        ax.set_yscale("log")
        ax.set_ylabel("temperature difference (K)")
        ax.legend(fontsize=8)
        ax.set_title("Magnon-phonon temperature relaxation")
        ax.grid(True, ls=":", alpha=0.5)

        tr, lam = decay_rate(t, dT)
        if len(tr):
            axr.plot(tr, lam, color="k", lw=2, label=r"$\lambda(t) = -d\ln|\Delta T|/dt$")
            if twotm_const is not None:
                axr.axhline(1.0 / twotm_const["tau"], color="#2ca02c", ls="-.", label="linear 2TM, constant G")
            if fit is not None and not fit["at_bound"]:
                axr.axhline(1.0 / fit["tau"], color="#d62728", ls="--", label="fit")
            axr.legend(fontsize=8)
            summary.append(f"instantaneous rate lambda(t): first quarter median {np.median(lam[:max(1, len(lam)//4)]):.4g}, "
                           f"last quarter median {np.median(lam[-max(1, len(lam)//4):]):.4g} ps^-1 "
                           "(equal for a single exponential)")
        axr.set_ylabel(r"relaxation rate $\lambda$ (ps$^{-1}$)")
        tfmt(axr, a.logt)
        save(fig, "dyn_relaxation.png")

        summary.append(f"final: T_mag_fit = {obs.T_mag_fit_K.iloc[-1]:.3f} K, T_phon_fit = {obs.T_phon_fit_K.iloc[-1]:.3f} K, "
                       f"mu_mag = {obs.mu_mag_meV.iloc[-1]:.4f} meV (initial T_mag/T_phon = "
                       f"{obs.T_mag_fit_K.iloc[0]:.1f}/{obs.T_phon_fit_K.iloc[0]:.1f} K)")

    with open(os.path.join(out, "dynamics_summary.txt"), "w") as f:
        f.write("\n".join(summary) + "\n")
    print("\n" + "\n".join(summary))


if __name__ == "__main__":
    main()
