import argparse
import numpy as np
import pandas as pd

# Bare-limit consistency test between the two rate implementations.
#
# Run SpinPhony once with the hybridization switched off and magnons/phonons at the
# SAME temperature, on the full BZ, so both files cover the same q-points:
#
#     python python/SpinPhony.py --material CrI3 --full_bz --no_slc --tmag 300 --tphon 300 \
#            --out_dir Outputs/bare_check
#
#   hybrid_path_lifetimes.csv  - hybrid vertex + phase_lifetime_hybrid_path (all modes at T_mag)
#   equilibrium_lifetimes.csv  - bare unified vertex + phase_lifetime (what the dynamics linearises to)
#
# With --no_slc the hybrid modes ARE the bare modes, so gamma_hybrid / gamma_bare must be 1 for
# every mode. The phonons are the sharpest probe: their rate comes only from the coalescence term.

p = argparse.ArgumentParser()
p.add_argument("folder")
p.add_argument("--emin", type=float, default=1.0, help="ignore modes below this energy (meV)")
a = p.parse_args()

h = pd.read_csv(f"{a.folder}/hybrid_path_lifetimes.csv", comment="#")
b = pd.read_csv(f"{a.folder}/equilibrium_lifetimes.csv")

for name, bsel, hsel in [("magnon", b.particle == "magnon", h.mag_character > 0.5),
                         ("phonon", b.particle == "phonon", h.phon_character > 0.5)]:
    B, H = b[bsel], h[hsel]
    rows = []
    for q, bq in B.groupby("q_idx"):
        hq = H[H.q_idx == q]
        if len(bq) != len(hq):
            continue
        bq, hq = bq.sort_values("energy_meV"), hq.sort_values("energy_meV")
        rows.append(pd.DataFrame({"E": bq.energy_meV.values, "dE": hq.energy_meV.values - bq.energy_meV.values,
                                  "gb": bq["gamma_ps-1"].values, "gh": hq["gamma_ps-1"].values}))
    if not rows:
        print(f"{name}: no q-point with matching mode counts"); continue
    d = pd.concat(rows)
    print(f"\n{name}: {len(d):,} modes, energy mismatch |dE|<1e-3 meV for {(d.dE.abs() < 1e-3).mean():.1%}")
    d = d[(d.E > a.emin) & (d.gb > 1e-8) & (d.gh > 1e-8) & (d.dE.abs() < 1e-3)]
    if d.empty:
        print("  nothing left after cuts (is --no_slc set? is equilibrium_lifetimes.csv from phase_lifetime?)"); continue
    r = d.gh / d.gb
    print(f"  gamma_hyb/gamma_bare: median {r.median():.3f}   quartiles {r.quantile(.25):.3f} .. {r.quantile(.75):.3f}"
          f"   ratio of sums {d.gh.sum() / d.gb.sum():.3f}")
    d["bin"] = pd.cut(d.E, 6)
    print(d.groupby("bin", observed=True).apply(lambda g: pd.Series({"n": len(g), "median ratio": (g.gh / g.gb).median()}),
                                                include_groups=False).to_string())
