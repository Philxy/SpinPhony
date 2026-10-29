#!/usr/bin/env python3
"""
Magnon-phonon coupling constant G_mp from SpinPhony's hybrid_path_lifetimes.csv
(run with --full_bz, so the "path" is the whole Brillouin-zone grid):

    G_mp = (1/N) * sum_{q,lambda} w_{q lambda} * E_{q lambda} * gamma_{q lambda} * dn_BE(E_{q lambda}, T)/dT

    dn_BE/dT = E / (k_B T^2) * n (n + 1),      n = 1 / (exp(E / k_B T) - 1)

w_{q lambda} restricts the sum to phonons. Choose with --weight:
    char   : w = phonon character            (default; smooth for hybrid modes)
    char2  : w = (phonon character)^2
    filter : w = 1 if phonon character > 0.5 else 0

Units
-----
E [meV] * gamma [1/ps] * dn/dT [1/K]   ->  meV / (ps K)  per primitive cell
1 meV/ps = 1.602176634e-22 J / 1e-12 s = 1.602176634e-10 W
  -> per cell     :  W / K
  -> volumetric   :  W / (m^3 K)   after dividing by the primitive-cell volume

Usage
-----
python calc_Gmp.py Outputs/bccFe_reprod/hybrid_path_lifetimes.csv
python calc_Gmp.py run.csv --T 300 --lattice bcc --a 2.8665
python calc_Gmp.py run.csv --cell_volume 11.777          # Angstrom^3, any lattice
"""
import argparse
import sys

import numpy as np
import pandas as pd

KB_MEV = 0.08617333262          # meV / K
MEV_PER_PS_TO_W = 1.602176634e-10
ANG3_TO_M3 = 1e-30

PRIMITIVE_VOLUME_FACTOR = {"sc": 1.0, "bcc": 0.5, "fcc": 0.25}


def infer_temperature(E, n):
    """Invert n = 1/(exp(E/kT)-1) on well-conditioned modes to recover the T the run used."""
    ok = (E > 0.5) & (n > 1e-8) & np.isfinite(n)
    if not ok.any():
        return None
    T = E[ok] / (KB_MEV * np.log1p(1.0 / n[ok]))
    return float(np.median(T)), float(np.std(T))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="hybrid_path_lifetimes.csv from a --full_bz run")
    ap.add_argument("--T", type=float, default=None,
                    help="Temperature (K). Default: inferred from the n_occ column, i.e. the T "
                         "the scattering rates were computed at.")
    ap.add_argument("--weight", choices=["char", "char2", "filter"], default="char",
                    help="How the phonon character restricts the sum (default: char)")
    ap.add_argument("--lattice", choices=list(PRIMITIVE_VOLUME_FACTOR), default="bcc")
    ap.add_argument("--a", type=float, default=2.8665, help="Conventional lattice constant (Angstrom)")
    ap.add_argument("--cell_volume", type=float, default=None,
                    help="Primitive-cell volume in Angstrom^3 (overrides --lattice/--a)")
    ap.add_argument("--min_energy", type=float, default=0.0,
                    help="Exclude modes with E <= this (meV); acoustic modes at Gamma are excluded anyway")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, comment="#")
    need = ["q_idx", "energy_meV", "gamma_ps-1", "phon_character"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit(f"Error: missing column(s) {missing}. Found: {list(df.columns)}")

    E = df["energy_meV"].to_numpy(float)
    gamma = pd.to_numeric(df["gamma_ps-1"], errors="coerce").to_numpy(float)
    pc = df["phon_character"].to_numpy(float)

    # N = number of q-points in the BZ sum (each q appears once per branch)
    N = df["q_idx"].nunique()
    if N < 1000:
        print(f"WARNING: only {N} q-points. G_mp is a BZ sum - was this run with --full_bz?")

    # Temperature
    if args.T is None:
        if "n_occ" not in df.columns:
            sys.exit("Error: no n_occ column to infer T from; pass --T explicitly.")
        res = infer_temperature(E, df["n_occ"].to_numpy(float))
        if res is None:
            sys.exit("Error: could not infer T from n_occ; pass --T explicitly.")
        T, T_spread = res
        print(f"Temperature inferred from n_occ: T = {T:.2f} K (spread {T_spread:.2e} K)")
    else:
        T = args.T
        if "n_occ" in df.columns:
            res = infer_temperature(E, df["n_occ"].to_numpy(float))
            if res and abs(res[0] - T) > 1.0:
                print(f"WARNING: rates were computed at T = {res[0]:.1f} K, you asked for {T:.1f} K. "
                      "gamma itself depends on T, so this mixes temperatures.")

    # Phonon weight
    if args.weight == "char":
        w = pc
    elif args.weight == "char2":
        w = pc ** 2
    else:
        w = (pc > 0.5).astype(float)
    w = np.clip(w, 0.0, 1.0)

    ok = np.isfinite(gamma) & np.isfinite(E) & (E > max(args.min_energy, 1e-6))
    n_dropped = int((~ok).sum())

    # dn/dT, overflow-safe: n(n+1) = 1 / (4 sinh^2(x/2)),  x = E/kT
    x = E[ok] / (KB_MEV * T)
    with np.errstate(over="ignore"):
        nn1 = np.where(x < 700, 1.0 / (4.0 * np.sinh(0.5 * np.minimum(x, 700)) ** 2), 0.0)
    dndT = x / T * nn1                     # = E/(kB T^2) * n(n+1)   [1/K]

    terms = w[ok] * E[ok] * gamma[ok] * dndT        # meV / (ps K) per mode
    G_cell_meV = terms.sum() / N                    # meV / (ps K) per primitive cell

    # Units
    G_cell_W = G_cell_meV * MEV_PER_PS_TO_W         # W / K per primitive cell
    if args.cell_volume is not None:
        V_A3 = args.cell_volume
        vol_src = "given"
    else:
        V_A3 = PRIMITIVE_VOLUME_FACTOR[args.lattice] * args.a ** 3
        vol_src = f"{args.lattice}, a = {args.a} A"
    V_m3 = V_A3 * ANG3_TO_M3
    G_SI = G_cell_W / V_m3                          # W / (m^3 K)

    # Reference: phonon heat capacity with the same weighting (for a 2TM time scale)
    C_cell_meV = (w[ok] * E[ok] * dndT).sum() / N   # meV / K per cell
    C_SI = C_cell_meV * 1.602176634e-22 / V_m3      # J / (m^3 K)

    print()
    print("=" * 62)
    print(f" File            : {args.csv}")
    print(f" q-points (N)    : {N:,}   modes used: {ok.sum():,}   dropped: {n_dropped:,}")
    print(f" T               : {T:.2f} K")
    print(f" Phonon weight   : {args.weight}   (sum of weights / N = {w[ok].sum() / N:.3f})")
    print(f" Cell volume     : {V_A3:.4f} A^3  ({vol_src})")
    print("-" * 62)
    print(f" G_mp            : {G_cell_meV:.6e} meV/(ps K) per cell")
    print(f"                 : {G_cell_W:.6e} W/K per cell")
    print(f" G_mp (SI)       : {G_SI:.6e} W m^-3 K^-1")
    print("-" * 62)
    print(f" C_ph (weighted) : {C_SI:.6e} J m^-3 K^-1")
    if G_SI > 0:
        print(f" C_ph / G_mp     : {C_SI / G_SI * 1e12:.4f} ps")
    print("=" * 62)


if __name__ == "__main__":
    main()