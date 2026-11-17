"""
Bare versus hybrid lifetimes of one run folder (needs path_lifetimes.csv and hybrid_path_lifetimes.csv, written by a run whose
lifetimes were not skipped - i.e. the hybrid runs of dyn_scan.sh; use an equal-temperature run, T_mag = T_phon, so both bases use
the same occupations).

    python python/plot_lifetimes_basis.py --dir Outputs/dyn/<SCAN>/bccFe_hybrid_Tm300_Tp300_g20

Figure lifetimes_bare_vs_hybrid.png
  left   : tau vs energy; bare magnons / bare phonons (open markers) and hybrid modes coloured by magnon weight
  right  : gamma_hybrid / gamma_bare for modes matched per q-point by energy order inside the same class
           (magnon-like = mag_character > 0.5, phonon-like otherwise); median and ratio of sums are printed
"""
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--out", default=None)
a = ap.parse_args()

b = pd.read_csv(os.path.join(a.dir, "path_lifetimes.csv"), comment="#")
h = pd.read_csv(os.path.join(a.dir, "hybrid_path_lifetimes.csv"), comment="#")
bt = pd.to_numeric(b["tau_ps"], errors="coerce"); ht = pd.to_numeric(h["tau_ps"], errors="coerce")
bok = np.isfinite(bt) & (bt > 0) & (b.energy_meV > 0.2); hok = np.isfinite(ht) & (ht > 0) & (h.energy_meV > 0.2)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.4))
for part, mk, col in (("magnon", "^", "#d62728"), ("phonon", "s", "#1f77b4")):
    s = bok & (b.particle == part)
    ax.scatter(b.energy_meV[s], bt[s], marker=mk, s=22, facecolors="none", edgecolors=col, lw=0.7, alpha=0.6, label=f"bare {part}", zorder=3)
sc = ax.scatter(h.energy_meV[hok], ht[hok], c=h.mag_character[hok], cmap="coolwarm_r", vmin=0, vmax=1, s=5, alpha=0.7, zorder=2,
                label="hybrid")
fig.colorbar(sc, ax=ax).set_label("magnon weight (hybrid modes)")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("energy (meV)"); ax.set_ylabel(r"lifetime $\tau$ (ps)")
ax.legend(fontsize=8); ax.grid(True, which="both", ls=":", alpha=0.4); ax.set_title("Lifetimes: bare vs hybrid basis")

summary = []
for name, bsel, hsel, col in (("magnon-like", b.particle == "magnon", h.mag_character > 0.5, "#d62728"),
                              ("phonon-like", b.particle == "phonon", h.mag_character <= 0.5, "#1f77b4")):
    B = b[bsel].sort_values(["q_idx", "energy_meV"]); H = h[hsel].sort_values(["q_idx", "energy_meV"])
    if len(B) != len(H):                              # classes can differ in size at some q: match only equal-size q
        cb, ch = B.groupby("q_idx").size(), H.groupby("q_idx").size()
        ok_q = cb.index[(cb.reindex(cb.index) == ch.reindex(cb.index)).fillna(False)]
        B, H = B[B.q_idx.isin(ok_q)], H[H.q_idx.isin(ok_q)]
    gb, gh = B["gamma_ps-1"].to_numpy(float), H["gamma_ps-1"].to_numpy(float)
    E = B.energy_meV.to_numpy(float)
    m = (gb > 1e-10) & (gh > 1e-10) & (E > 0.2)
    ax2.scatter(E[m], gh[m] / gb[m], s=4, alpha=0.4, color=col, label=name)
    summary.append(f"{name:12s}: {m.sum():6d} matched modes; median gamma_hyb/gamma_bare = {np.median(gh[m]/gb[m]):.3g}; "
                   f"ratio of summed rates = {gh.sum()/gb.sum():.3g}")
ax2.axhline(1.0, color="k", lw=0.8)
ax2.set_xscale("log"); ax2.set_yscale("log"); ax2.set_xlabel("energy (meV)"); ax2.set_ylabel(r"$\gamma_{hyb}/\gamma_{bare}$")
ax2.legend(fontsize=8); ax2.grid(True, which="both", ls=":", alpha=0.4); ax2.set_title("Rate ratio per matched mode (1 = same)")
fig.tight_layout()
out = a.out or os.path.join(a.dir, "lifetimes_bare_vs_hybrid.png")
fig.savefig(out, dpi=250); print(f" -> {out}")
print("\n".join(summary))
