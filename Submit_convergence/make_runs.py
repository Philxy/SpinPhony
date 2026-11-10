#!/usr/bin/env python3
"""
Writes the run list of a convergence scan of the bare lifetimes (grid size N x Gaussian broadening) and prints the sbatch command.

    python3 Submit_convergence/make_runs.py --tag conv_$(date +%Y%m%d) [--materials bccFe fccNi] [--smearings 0.05 0.1 0.2 0.3 0.5]
                                            [--grids bccFe=12,20,40 fccNi=20,32,40] [--T 300] [--max_parallel 6]

Output: Outputs/convergence/<tag>/runs.csv  (one line per run, index = SLURM array task id, read by Submit_convergence/submit.sh)
        and later  Outputs/convergence/<tag>/<run>/{config.toml, params.txt, path_lifetimes.csv, run.log, status.txt}.
Post-processing (python/convergence_lifetimes.py, python/convergence_Gmp.py) takes the scan directory.
Pure standard library: runs on the login node.
"""
import argparse
import csv
import os

MESHES = {
    "bccFe": {12: "Inputs/bccFe/combined_band_12x12x12.h5", 20: "Inputs/bccFe/combined_band_20x20x20.h5", 40: "Inputs/bccFe/grid_40x40x40.h5"},
    "fccNi": {20: "Inputs/fccNi/combined_20x20x20_fccNi.h5", 32: "Inputs/fccNi/combined_32x32x32_fccNi.h5", 40: "Inputs/fccNi/combined_40x40x40_fccNi.h5"},
}
PH3_FC3 = "Inputs/bccFe/PH3/fc3.hdf5"
# starting point of anticipated_fraction (channel-buffer share of N^2 n_mag^2 n_ph 3) at smearing 0.1; scaled with the smearing and
# raised automatically by submit.sh when the scan reports an overflow
FRAC0 = {12: 0.20, 20: 0.06, 32: 0.03, 40: 0.02}
MEM_BUDGET_GB = 60.0      # channel buffer: 36 bytes per slot, N^2 * 9 slots per unit of the fraction

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--materials", nargs="+", default=["bccFe", "fccNi"])
ap.add_argument("--smearings", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.3, 0.5])
ap.add_argument("--grids", nargs="*", default=[], help="override per material, e.g. bccFe=12,20,40 fccNi=20,32")
ap.add_argument("--T", type=float, default=300.0, help="magnon = phonon temperature (K)")
ap.add_argument("--no_phph", action="store_true", help="no three-phonon scattering for bccFe (it is always off for fccNi)")
ap.add_argument("--max_parallel", type=int, default=6)
ap.add_argument("--root", default="Outputs/convergence")
a = ap.parse_args()

grids = {m: sorted(MESHES[m]) for m in a.materials}
for g in a.grids:
    m, v = g.split("=")
    grids[m] = [int(x) for x in v.split(",")]

rows = []
for m in a.materials:
    for n in grids[m]:
        for s in a.smearings:
            frac = min(1.0, MEM_BUDGET_GB * 1e9 / (36.0 * 9.0 * float(n) ** 6), FRAC0.get(n, 0.03) * max(1.0, s / 0.1))
            rows.append(dict(id=len(rows), run=f"{m}_N{n}_s{s:g}", material=m, N=n, mesh=MESHES[m][n], smearing=s, frac0=round(frac, 5),
                             phph=int(m == "bccFe" and not a.no_phph), fc3=PH3_FC3 if m == "bccFe" else "", T=a.T))
root = os.path.join(a.root, a.tag)
os.makedirs(root, exist_ok=True)
with open(os.path.join(root, "runs.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(f"{len(rows)} runs -> {root}/runs.csv")
print(f"submit:  mkdir -p {a.root}/logs && sbatch --array=0-{len(rows) - 1}%{a.max_parallel} Submit_convergence/submit.sh {a.tag}")
