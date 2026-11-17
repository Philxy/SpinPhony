"""
Bare and hybrid lifetimes of bcc Fe next to the reference (literature) magnon and phonon lifetimes, and G_mp from each (SI, printed).

    python python/plot_lifetimes_Fe_ref.py --dir Outputs/dyn/<SCAN>/bccFe_hybrid_Tm300_Tp300_g20 [--T 300]

The folder needs path_lifetimes.csv (bare) and hybrid_path_lifetimes.csv of an equal-temperature run. The reference rates and the G_mp helpers
(dn/dT, literature binning) are taken from python/plot-hybrid-lifetimesbccFe.py, so the numbers follow the same conventions as that script:
    G_mp = (1/N) sum_modes w E gamma dn_BE/dT ,   per cell -> W m^-3 K^-1 with the primitive-cell volume
  bare       : phonon rows of path_lifetimes.csv, w = 1
  hybrid     : all hybrid modes, w = phonon character
  reference  : literature phonon rate gamma(E) (energy-binned, interpolated) on the bare phonon energies (w = 1) and on the hybrid modes (w = phonon
               character); arithmetic- and geometric-mean binning (the geometric one is a lower bound). Energies outside the literature range get 0.
Output lifetimes_Fe_vs_reference.png in --dir.
"""
import argparse, importlib.util, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("fe_ref", os.path.join(here, "plot-hybrid-lifetimesbccFe.py"))
ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True); ap.add_argument("--T", type=float, default=None)
ap.add_argument("--a", type=float, default=2.8665); ap.add_argument("--cell_volume", type=float, default=None)
a = ap.parse_args()

b = pd.read_csv(os.path.join(a.dir, "path_lifetimes.csv"), comment="#")
h = ref.load_spinphony_csv(os.path.join(a.dir, "hybrid_path_lifetimes.csv"))
lit_m = pd.read_csv(ref.io.StringIO(ref.literature_scattering_rates), names=["E", "g"]); lit_p = pd.read_csv(ref.io.StringIO(ref.literature_phonon_scatt_rates_per_ps), names=["E", "g"])
T = a.T
if T is None:
    pf = os.path.join(a.dir, "params.txt")
    if os.path.exists(pf):
        for line in open(pf):
            if line.startswith("T_mag="): T = float(line.split("=")[1])
    T = T or ref.infer_temperature(h) or 300.0
V_A3 = a.cell_volume if a.cell_volume else 0.5 * a.a ** 3; V_m3 = V_A3 * ref.ANG3_TO_M3
N = b.q_idx.nunique()

bt = pd.to_numeric(b.tau_ps, errors="coerce"); ht = pd.to_numeric(h.tau_ps, errors="coerce")
bok = np.isfinite(bt) & (bt > 0) & (b.energy_meV > 0.2); hok = np.isfinite(ht) & (ht > 0) & (h.energy_meV > 0.2)
fig, ax = plt.subplots(figsize=(9.5, 6.4))
for part, mk, col, lab in (("magnon", "^", "#d62728", "bare magnons"), ("phonon", "s", "#1f77b4", "bare phonons")):
    s = bok & (b.particle == part)
    ax.scatter(b.energy_meV[s], 4 * bt[s], marker=mk, s=16, facecolors="none", edgecolors=col, lw=0.6, alpha=0.5, label=lab, zorder=2)
sc = ax.scatter(h.energy_meV[hok], ht[hok], c=(h.mag_character - h.phon_character)[hok], cmap="coolwarm", vmin=-1, vmax=1, s=4, alpha=0.7, zorder=3,
                label="hybrid modes")
fig.colorbar(sc, ax=ax).set_label("hybrid character (blue = phonon, red = magnon)")
ax.scatter(lit_m.E, 1 / lit_m.g, marker="^", s=45, facecolors="none", edgecolors="k", lw=1.2, label="reference: magnons", zorder=5)
ax.scatter(lit_p.E, 1 / lit_p.g, marker="s", s=45, facecolors="none", edgecolors="k", lw=1.2, label="reference: phonons", zorder=5)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("energy (meV)"); ax.set_ylabel(r"lifetime $\tau$ (ps)")
ax.set_title(f"bcc Fe, T = {T:g} K: bare and hybrid vs reference lifetimes"); ax.grid(True, which="both", ls=":", alpha=0.4); ax.legend(fontsize=8, loc="upper right")
fig.tight_layout(); out = os.path.join(a.dir, "lifetimes_Fe_vs_reference.png"); fig.savefig(out, dpi=250); print(f" -> {out}\n")

# ---------------- G_mp in SI
bp = b[b.particle == "phonon"]; Eb = bp.energy_meV.to_numpy(float); gb = bp["gamma_ps-1"].to_numpy(float); kb = Eb > 0.2
Eh = h.energy_meV.to_numpy(float); gh = pd.to_numeric(h["gamma_ps-1"], errors="coerce").to_numpy(float); wh = np.clip(h.phon_character.to_numpy(float), 0, 1); kh = Eh > 0.2
Gb = ref.compute_Gmp(Eb[kb], gb[kb], np.ones(kb.sum()), T, N, V_m3)
Gh = ref.compute_Gmp(Eh[kh], gh[kh], wh[kh], T, N, V_m3)
Elit, glit = lit_p.E.to_numpy(float), lit_p.g.to_numpy(float)
res = {}
for tag, E, w in (("bare phonon energies, w=1", Eb[kb], np.ones(kb.sum())), ("hybrid modes, w=phonon character", Eh[kh], wh[kh])):
    ga, inside = ref.literature_gamma_of_E(Elit, glit, E, mean="arith"); gg, _ = ref.literature_gamma_of_E(Elit, glit, E, mean="geo")
    res[tag] = (ref.compute_Gmp(E, ga, w, T, N, V_m3), ref.compute_Gmp(E, gg, w, T, N, V_m3),
                w[~inside].sum() / max(w.sum(), 1e-30))
print("=" * 78); print(f" bcc Fe  T = {T:g} K   N = {N} q-points   V_cell = {V_A3:.4f} A^3")
print("-" * 78); print(f" {'G_mp from':44s} {'W m^-3 K^-1':>14s}   {'meV/(K ps)/cell':>16s}")
print(f" {'bare lifetimes (phonon rows)':44s} {Gb[1]:14.4e}   {Gb[0]:16.4e}")
print(f" {'hybrid lifetimes (phonon-character weighted)':44s} {Gh[1]:14.4e}   {Gh[0]:16.4e}")
for tag, ((a_c, a_s), (g_c, g_s), out_w) in res.items():
    print(f" {'reference, arithmetic, ' + tag:44s} {a_s:14.4e}   {a_c:16.4e}")
    print(f" {'reference, geometric,  ' + tag:44s} {g_s:14.4e}   {g_c:16.4e}   (lower bound; {100*out_w:.1f}% of weight outside the reference energy range [{Elit.min():.1f}, {Elit.max():.1f}] meV)")
print("-" * 78)
print(f" ratios: hybrid/bare = {Gh[1]/Gb[1]:.3g};  bare/reference(arith, bare energies) = {Gb[1]/list(res.values())[0][0][1]:.3g};  hybrid/reference(arith, hybrid modes) = {Gh[1]/list(res.values())[1][0][1]:.3g}")
print("=" * 78)
