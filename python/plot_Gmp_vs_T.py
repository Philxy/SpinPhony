import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# G_mp versus temperature for all materials in one plot.
#
# Input: scan folders written by temp_submit.sh, i.e.
#     <scan>/<material>_<mode>_T<T>K/{params.txt, hybrid_path_lifetimes.csv}
# Several scan folders (or shell globs) may be given; every run folder found below
# them is used. G_mp is evaluated from each run's lifetimes,
#     G_mp = (1/N) sum_{q,lam} w * E * gamma * dn_BE/dT |_T ,   w = phonon character,
# with T the magnon temperature of the run (params.txt; fallback: inferred from n_occ).
# Only full-BZ runs are a BZ sample, so path runs are skipped (use --allow_path to keep them).

KB_MEV = 0.08617333262             # meV/K
MEV_PER_PS_TO_W = 1.602176634e-10  # 1 meV/ps in W
ANG3_TO_M3 = 1e-30

# Primitive-cell volumes in A^3 (override with --volume MATERIAL=V)
CELL_VOLUME = {
    'CrI3': 281.794,                       # bulk rhombohedral cell, 8 atoms (Inputs/CrI3/band.h5)
    'fccNi': 3.516822176 ** 3 / 4.0,       # fcc primitive, a from the h5 files
    'bccFe': 2.8665 ** 3 / 2.0,            # bcc primitive
}

COLUMN_ALIASES = {'mag_char': 'mag_character', 'phon_char': 'phon_character'}
CSV_NAMES = ('hybrid_path_lifetimes.csv', 'hybrid_equilibrium_lifetimes.csv')


def dn_dT(E, T):
    """dn_BE/dT = E/(kB T^2) n(n+1), overflow-safe via n(n+1) = 1/(4 sinh^2(x/2))."""
    x = np.asarray(E, float) / (KB_MEV * T)
    out = np.zeros_like(x)
    ok = (x > 0) & (x < 700)
    out[ok] = (x[ok] / T) / (4.0 * np.sinh(0.5 * x[ok]) ** 2)
    return out


def infer_temperature(df):
    if 'n_occ' not in df.columns:
        return None
    E, n = df['energy_meV'].to_numpy(float), df['n_occ'].to_numpy(float)
    ok = (E > 0.5) & (n > 1e-8) & np.isfinite(n)
    if not ok.any():
        return None
    return float(np.median(E[ok] / (KB_MEV * np.log1p(1.0 / n[ok]))))


def read_params(path):
    p = {}
    with open(path) as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=', 1)
                p[k] = v
    return p


def gmp_of_run(csv, T, V_A3):
    df = pd.read_csv(csv, comment='#')
    df = df.rename(columns={o: n for o, n in COLUMN_ALIASES.items() if o in df.columns})
    E = df['energy_meV'].to_numpy(float)
    g = np.where(np.isfinite(pd.to_numeric(df['gamma_ps-1'], errors='coerce')),
                 pd.to_numeric(df['gamma_ps-1'], errors='coerce'), 0.0)
    w = np.clip(df['phon_character'].to_numpy(float), 0.0, 1.0)
    N = df['q_idx'].nunique()
    keep = E > 0
    G_cell = np.sum(w[keep] * E[keep] * g[keep] * dn_dT(E[keep], T)) / N
    return G_cell, G_cell * MEV_PER_PS_TO_W / (V_A3 * ANG3_TO_M3), T, N, df


def find_runs(roots):
    runs = []
    for root in roots:
        for params in glob.glob(os.path.join(root, '**', 'params.txt'), recursive=True):
            d = os.path.dirname(params)
            csv = next((os.path.join(d, n) for n in CSV_NAMES if os.path.exists(os.path.join(d, n))), None)
            if csv is None:
                print(f"no lifetimes csv yet in {d} (job unfinished or failed? "
                      f"{'.done present' if os.path.exists(os.path.join(d, '.done')) else 'no .done marker'})")
                continue
            runs.append((d, params, csv))
    return runs


def main():
    ap = argparse.ArgumentParser(description="Plot G_mp vs temperature for all materials")
    ap.add_argument("scans", nargs='+', help="scan folders (or the parent Outputs/scans); globs allowed")
    ap.add_argument("--out", default="Gmp_vs_T.png", help="output figure (default: ./Gmp_vs_T.png)")
    ap.add_argument("--csv", default=None, help="summary csv (default: <out>.csv)")
    ap.add_argument("--per_cell", action="store_true", help="plot meV/(ps K) per cell instead of W m^-3 K^-1")
    ap.add_argument("--linear_y", action="store_true", help="linear instead of log y axis")
    ap.add_argument("--volume", action="append", default=[], metavar="MAT=V",
                    help="primitive-cell volume in A^3 for a material, e.g. --volume CrI3=80.2")
    ap.add_argument("--allow_path", action="store_true",
                    help="also use path-mode runs (G_mp is then NOT a BZ integral)")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    vols = dict(CELL_VOLUME)
    for item in args.volume:
        k, v = item.split('=')
        vols[k] = float(v)

    roots = [r for pat in args.scans for r in (glob.glob(pat) or [pat])]
    runs = find_runs(roots)
    if not runs:
        raise SystemExit("No run folders (params.txt + lifetimes csv) found below: " + ", ".join(args.scans))

    rows = {}
    for d, params, csv in runs:
        p = read_params(params)
        mat = p.get('material', '?')
        mode = p.get('mode', '?')
        if mode != 'fullbz' and not args.allow_path:
            print(f"skip {d}: mode={mode} (use --allow_path to include)")
            continue
        if mat not in vols:
            print(f"skip {d}: no cell volume for '{mat}' (use --volume {mat}=V)")
            continue
        T = float(p['T_mag']) if 'T_mag' in p else None
        try:
            tmp = pd.read_csv(csv, comment='#', usecols=['energy_meV', 'n_occ'])
        except ValueError:
            tmp = None
        if T is None:
            T = infer_temperature(tmp) if tmp is not None else None
            if T is None:
                print(f"skip {d}: no temperature (no T_mag in params.txt, no n_occ column)")
                continue
        else:
            T_inf = infer_temperature(tmp) if tmp is not None else None
            if T_inf is not None and abs(T_inf - T) > 0.05 * T:
                print(f"WARNING {d}: params T_mag={T:g} K but n_occ implies {T_inf:.1f} K")
        G_cell, G_vol, T, N, _ = gmp_of_run(csv, T, vols[mat])
        key = (mat, p.get('scan', '?'), T)
        mtime = os.path.getmtime(csv)
        if key not in rows or mtime > rows[key]['mtime']:       # newest run wins for duplicates
            rows[key] = dict(material=mat, scan=p.get('scan', '?'), T_K=T, N_q=N, V_A3=vols[mat],
                             Gmp_W_m3K=G_vol, Gmp_meV_psK_cell=G_cell, mtime=mtime, run_dir=d)
            print(f"{mat:6s} T={T:6.1f} K  N={N:6d}  G_mp={G_vol:.4e} W m^-3 K^-1   [{d}]")

    if not rows:
        raise SystemExit("Nothing to plot.")
    res = pd.DataFrame(rows.values()).sort_values(['material', 'scan', 'T_K']).drop(columns='mtime')
    csv_out = args.csv or os.path.splitext(args.out)[0] + '.csv'
    res.to_csv(csv_out, index=False)
    print(f" -> Saved {csv_out}")

    ycol = 'Gmp_meV_psK_cell' if args.per_cell else 'Gmp_W_m3K'
    ylabel = (r'$G_{mp}$ (meV ps$^{-1}$ K$^{-1}$ / cell)' if args.per_cell
              else r'$G_{mp}$ (W m$^{-3}$ K$^{-1}$)')
    fig, ax = plt.subplots(figsize=(8, 6))
    multi_scan = res.groupby('material')['scan'].nunique().max() > 1
    for (mat, scan), g in res.groupby(['material', 'scan']):
        ax.plot(g['T_K'], g[ycol], 'o-', ms=6, label=f"{mat} ({scan})" if multi_scan else mat)
    ax.set_xlabel('Temperature (K)', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    if not args.linear_y:
        ax.set_yscale('log')
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_title(r'$G_{mp}$ vs temperature', fontsize=14)
    fig.tight_layout()
    fig.savefig(args.out, dpi=300)
    print(f" -> Saved {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
