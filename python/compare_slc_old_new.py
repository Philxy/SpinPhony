"""
Compare bare lifetimes and G_mp of two inputs (e.g. old and new SLC tensor) of the same material.

    python python/compare_slc_old_new.py --old Outputs/compare_newSLC/old/Tscan_20261110 \
                                         --new Outputs/compare_newSLC/new/Tscan_newSLC_20261112 \
                                         [--material fccNi] [--N 40] [--smearing 0.2] [--T 300] [--out_dir Outputs/compare_newSLC]

Both folders hold run directories <material>_N<N>_s<smearing>_T<T>K/ with params.txt and path_lifetimes.csv (a temperature scan made with
Submit_convergence/make_runs.py --temps ...). Runs are paired by (N, smearing, T).
  G_mp(T) = (1/N_q) sum_{phonons} E gamma dn_BE/dT   [meV K^-1 ps^-1 per cell]  -> W m^-3 K^-1 with the primitive cell volume (fcc: a^3/4,
            a = 3.516822176 A; --cell_volume overrides). T is the run temperature (T_mag = T_phon).
Outputs in --out_dir:
  gmp_old_vs_new.png / .csv   G_mp(T) of both inputs and their ratio; the table is also printed
  lifetimes_old_vs_new.png    scattering rate (log) against energy (linear), magnons and phonons, old (open) and new (filled), one plot (--T)
"""
import argparse, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KB = 0.08617333262; MEV_PER_PS_TO_W = 1.602176634e-10; ANG3 = 1e-30

def read_params(p):
    r = {}
    if os.path.exists(p):
        for line in open(p):
            if "=" in line: k, v = line.strip().split("=", 1); r[k] = v
    return r

def dn_dT(E, T):
    x = E / (KB * T); out = np.zeros_like(x); ok = (x > 0) & (x < 700)
    out[ok] = (x[ok] / T) / (4.0 * np.sinh(0.5 * x[ok]) ** 2)
    return out

def collect(root, material, N, smearing):
    runs = {}
    for pf in glob.glob(os.path.join(root, "*", "params.txt")):
        p = read_params(pf); d = os.path.dirname(pf)
        if p.get("material") != material or not os.path.exists(os.path.join(d, "path_lifetimes.csv")): continue
        if N is not None and int(float(p.get("N", -1))) != N: continue
        if smearing is not None and abs(float(p.get("smearing", -1)) - smearing) > 1e-9: continue
        runs[(int(float(p["N"])), float(p["smearing"]), float(p["T"]))] = d
    return runs

def gmp(df, T, V_A3):
    ph = df[df.particle == "phonon"]
    E = ph.energy_meV.to_numpy(float); g = pd.to_numeric(ph["gamma_ps-1"], errors="coerce").to_numpy(float)
    g = np.where(np.isfinite(g), g, 0.0); N = df.q_idx.nunique(); k = E > 0.2
    G = np.sum(E[k] * g[k] * dn_dT(E[k], T)) / N
    return G, G * MEV_PER_PS_TO_W / (V_A3 * ANG3)

ap = argparse.ArgumentParser()
ap.add_argument("--old", required=True); ap.add_argument("--new", required=True)
ap.add_argument("--material", default="fccNi"); ap.add_argument("--N", type=int, default=None); ap.add_argument("--smearing", type=float, default=None)
ap.add_argument("--T", type=float, default=300.0, help="temperature of the lifetime plot (K)")
ap.add_argument("--a", type=float, default=3.516822176); ap.add_argument("--cell_volume", type=float, default=None)
ap.add_argument("--out_dir", default=".")
ap.add_argument("--E0", type=float, default=50.0, help="energy (meV) where the E^1 / E^2 guide lines are anchored to the data (magnon plot)")
a = ap.parse_args()
V_A3 = a.cell_volume or a.a ** 3 / 4.0
old = collect(a.old, a.material, a.N, a.smearing); new = collect(a.new, a.material, a.N, a.smearing)
keys = sorted(set(old) & set(new), key=lambda k: (k[0], k[1], k[2]))
if not keys:
    raise SystemExit(f"no common (N, smearing, T) runs: old {sorted(old)[:4]}..., new {sorted(new)[:4]}...")
os.makedirs(a.out_dir, exist_ok=True)

rows = []
for key in keys:
    Go, Gos = gmp(pd.read_csv(os.path.join(old[key], "path_lifetimes.csv"), comment="#"), key[2], V_A3)
    Gn, Gns = gmp(pd.read_csv(os.path.join(new[key], "path_lifetimes.csv"), comment="#"), key[2], V_A3)
    rows.append(dict(N=key[0], smearing=key[1], T_K=key[2], G_old_meV=Go, G_new_meV=Gn, G_old_SI=Gos, G_new_SI=Gns, ratio_new_over_old=Gn / Go if Go else np.nan))
res = pd.DataFrame(rows); res.to_csv(os.path.join(a.out_dir, "gmp_old_vs_new.csv"), index=False)
print(f"{a.material}, cell volume {V_A3:.4f} A^3\n" + res[["N", "smearing", "T_K", "G_old_SI", "G_new_SI", "ratio_new_over_old"]].to_string(index=False, float_format=lambda x: f"{x:.4g}"))

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.5, 8), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.4]))
for (N, s), g in res.groupby(["N", "smearing"]):
    ax.plot(g.T_K, g.G_old_SI, "o--", lw=1.6, ms=6, mfc="none", label=f"old input (N={N}, s={s:g})")
    ax.plot(g.T_K, g.G_new_SI, "s-", lw=1.8, ms=5, label=f"new input (N={N}, s={s:g})")
    ax2.plot(g.T_K, g.ratio_new_over_old, "k.-")
ax.set_yscale("log"); ax.set_ylabel(r"$G_{mp}$ (W m$^{-3}$ K$^{-1}$)"); ax.legend(fontsize=8); ax.grid(True, which="both", ls=":", alpha=0.5)
ax.set_title(f"{a.material}: G_mp, old vs new input")
ax2.axhline(1.0, color="0.5", lw=0.8); ax2.set_xlabel("Temperature (K)"); ax2.set_ylabel("new / old"); ax2.grid(True, ls=":", alpha=0.5)
fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "gmp_old_vs_new.png"), dpi=250)

# ---- lifetimes: energy (linear) vs scattering rate (log), old open / new filled, magnons and phonons in ONE plot
kT = [k for k in keys if abs(k[2] - a.T) < 1e-6]
if not kT:
    print(f"no common run at T = {a.T:g} K for the lifetime plot (have {sorted({k[2] for k in keys})})")
else:
    k = kT[0]; fig, axs = plt.subplots(1, 2, figsize=(14, 5.8))
    sty = {("new", "magnon"): "#ff7f0e", ("new", "phonon"): "#17becf", ("old", "magnon"): "#8b0000", ("old", "phonon"): "#00008b"}
    for lab, d, filled in (("old", old[k], False), ("new", new[k], True)):
        df = pd.read_csv(os.path.join(d, "path_lifetimes.csv"), comment="#")
        g = pd.to_numeric(df["gamma_ps-1"], errors="coerce"); ok = np.isfinite(g) & (g > 0) & (df.energy_meV > 0.2)
        for ax, part, mk in ((axs[0], "magnon", "^"), (axs[1], "phonon", "o")):
            col = sty[(lab, part)]; s_ = ok & (df.particle == part)
            ax.scatter(df.energy_meV[s_], g[s_], marker=mk, s=14 if filled else 10, facecolors=col if filled else "none",
                       edgecolors="none" if filled else col, linewidths=0.6, alpha=0.6 if filled else 0.9, rasterized=True,
                       label=f"{lab} input", zorder=3 if filled else 2)
    for ax, part in zip(axs, ("magnons", "phonons")):
        ax.set_yscale("log"); ax.set_xlabel("energy (meV)"); ax.set_title(f"{a.material} {part}, T = {a.T:g} K (N={k[0]}, s={k[1]:g})")
        ax.legend(fontsize=9, markerscale=2); ax.grid(True, which="both", ls=":", alpha=0.4)
    # guide lines gamma ~ E^1 and E^2 in the magnon panel, anchored on the median rate of each input near --E0
    Eg = np.linspace(a.E0, 380.0, 100)
    for lab, d, col in (("old", old[k], "#8b0000"), ("new", new[k], "#ff7f0e")):
        df = pd.read_csv(os.path.join(d, "path_lifetimes.csv"), comment="#")
        g = pd.to_numeric(df["gamma_ps-1"], errors="coerce"); m = (df.particle == "magnon") & np.isfinite(g) & (g > 0) & (abs(df.energy_meV - a.E0) < 0.1 * a.E0)
        g0 = float(np.median(g[m])) if m.any() else np.nan
        for n, ls in ((1, "--"), (2, ":")):
            axs[0].plot(Eg, g0 * (Eg / a.E0) ** n, color="k" if lab == "old" else "0.45", ls=ls, lw=2.2, zorder=5,
                        label=rf"{lab}: $\propto E^{n}$ (from {a.E0:g} meV)")
    axs[0].legend(fontsize=8, markerscale=2, loc="upper left")
    axs[0].set_ylim(top=3e3)
    axs[0].set_ylabel(r"scattering rate $\gamma$ (ps$^{-1}$)")
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "lifetimes_old_vs_new.png"), dpi=250)
print(f"\n -> {os.path.join(a.out_dir, 'gmp_old_vs_new.png')}\n -> {os.path.join(a.out_dir, 'lifetimes_old_vs_new.png')}")
