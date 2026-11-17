"""
Acoustic sum rule (translation invariance) of the three-site spin-lattice coupling tensor J^{ab,mu}_{ij,k} of bcc Fe
(Mankovsky et al., PRL 129, 067202), read from Fe_full_tensor_ij-uk_{x,y,z}_displacement.csv.

    python python/check_slc_asr.py [--dir Inputs/bccFe] [--out_dir Outputs/slc_asr] [--project uniform|self|weighted] [--write]

Moving all atoms rigidly must not change any exchange tensor J^{ab}_{ij}, hence for every pair (i,j), displacement
direction mu and tensor component (a,b):   sum_k J^{ab,mu}_{ij,k} = 0    (k runs over all atoms incl. k = i and k = j).
Checked separately for the isotropic part (J_xx+J_yy)/2, the DMI-like part D^x,D^y,D^z (antisymmetric off-diagonals)
and every raw tensor component. Printed: max / rms of the sum over all pairs, the same relative to the pair's sum |J_k|, the
residual of the nearest-neighbour pair, and the partial sum versus the k cutoff.

--project: removes the violation per (i,j,mu,component) from the tensor entries (the corrected set satisfies the rule exactly)
    uniform   subtract sum/n_k from every k          (minimal L2 change)
    weighted  distribute the residual over k in proportion to |J_k|   (keeps small entries small)
    self      put the whole residual on the k = i term (the usual 'self' term fixed by the sum rule)
--write   writes the corrected tensors to --out_dir as <name>_asr.csv (same columns as the input; the input is never modified)
          and a summary csv of the sums.
Units are those of the input files (positions in units of a).
"""
import argparse
import os

import numpy as np
import pandas as pd

COMP = ["J_xx", "J_xy", "J_xz", "J_yx", "J_yy", "J_yz", "J_zx", "J_zy", "J_zz"]
KEY = ["x_ij", "y_ij", "z_ij"]
POS = ["x_ij", "y_ij", "z_ij", "x_ik", "y_ik", "z_ik"]


def derived(d):
    d = d.copy()
    d["iso"] = (d.J_xx + d.J_yy) / 2
    d["Dx"] = (d.J_yz - d.J_zy) / 2
    d["Dy"] = (d.J_zx - d.J_xz) / 2
    d["Dz"] = (d.J_xy - d.J_yx) / 2
    return d


def project(d, method):
    d = d.copy()
    g = d.groupby(KEY)
    for c in COMP:
        S = g[c].transform("sum")
        if method == "uniform":
            d[c] = d[c] - S / g[c].transform("size")
        elif method == "weighted":
            w = d[c].abs()
            W = w.groupby([d[k] for k in KEY]).transform("sum").replace(0, np.nan)
            d[c] = d[c] - (S * w / W).fillna(0.0)
        elif method == "self":
            self_k = (d.x_ik == 0) & (d.y_ik == 0) & (d.z_ik == 0)
            d.loc[self_k, c] = d.loc[self_k, c] - S[self_k]
    return d


def report(d, name, cols):
    g = d.groupby(KEY)
    S = g[cols].sum()
    A = g[cols].agg(lambda x: x.abs().sum())
    rel = (S.abs() / A.replace(0, np.nan)).fillna(0.0)
    out = pd.DataFrame({"max|sum_k|": S.abs().max(), "rms sum_k": np.sqrt((S ** 2).mean()), "max |J_k|": d[cols].abs().max(),
                        "max sum|J_k|": A.max(), "median rel": rel.median(), "max rel": rel.max()})
    print(f"\n=== displacement along {name}:  sum_k over {len(S)} pairs (i,j)")
    print(out.round(5).to_string())
    return S, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="Inputs/bccFe")
    ap.add_argument("--out_dir", default="Outputs/slc_asr")
    ap.add_argument("--project", choices=("uniform", "weighted", "self"), default=None)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--pair", nargs=3, type=float, default=(0.5, -0.5, -0.5), help="(i,j) vector (units of a) for the cutoff-convergence printout")
    a = ap.parse_args()
    cols = COMP + ["iso", "Dx", "Dy", "Dz"]
    summary = []
    for m in "xyz":
        fn = os.path.join(a.dir, f"Fe_full_tensor_ij-uk_{m}_displacement.csv")
        d = pd.read_csv(fn)
        for c in POS:
            d[c] = d[c].round(4)
        d = derived(d)
        S, out = report(d, m, cols)
        out.insert(0, "u", m); out.insert(1, "state", "input")
        summary.append(out.reset_index().rename(columns={"index": "component"}))
        pr = tuple(round(x, 4) for x in a.pair)
        if pr in S.index:
            s = d[(d.x_ij == pr[0]) & (d.y_ij == pr[1]) & (d.z_ij == pr[2])].copy()
            s["rik"] = np.sqrt(s.x_ik ** 2 + s.y_ik ** 2 + s.z_ik ** 2)
            s = s.sort_values("rik")
            print(f"  pair {pr}: sum_k iso={S.loc[pr, 'iso']:.4f} Dx={S.loc[pr, 'Dx']:.4f} Dy={S.loc[pr, 'Dy']:.4f} Dz={S.loc[pr, 'Dz']:.4f};  partial sums vs r_ik cutoff:")
            for R in (0.9, 1.5, 2.0, 2.5, 3.0, 3.5, 4.01):
                t = s[s.rik <= R]
                print(f"     r_ik<={R:<4} n={len(t):4d}  iso={t.iso.sum():9.4f}  Dx={t.Dx.sum():8.4f}  Dy={t.Dy.sum():8.4f}  Dz={t.Dz.sum():8.4f}")
        if a.project:
            p = derived(project(d[POS + COMP], a.project))
            S2, out2 = report(p, f"{m} (after '{a.project}' projection)", cols)
            ch = (p[COMP] - d[COMP]).abs()
            print(f"  change of the tensor entries: max {ch.values.max():.4f}, rms {np.sqrt((ch.values ** 2).mean()):.5f}; "
                  f"iso/D after projection: max|iso change| {(p.iso - d.iso).abs().max():.4f}, max|D change| {max((p[k] - d[k]).abs().max() for k in ('Dx', 'Dy', 'Dz')):.4f}")
            out2.insert(0, "u", m); out2.insert(1, "state", a.project)
            summary.append(out2.reset_index().rename(columns={"index": "component"}))
            if a.write:
                os.makedirs(a.out_dir, exist_ok=True)
                full = pd.read_csv(fn)
                full[COMP] = p[COMP].to_numpy()
                fo = os.path.join(a.out_dir, os.path.basename(fn).replace(".csv", "_asr.csv"))
                full.to_csv(fo, index=False, float_format="%.6f")
                print(f" -> {fo}")
    if a.write:
        os.makedirs(a.out_dir, exist_ok=True)
        pd.concat(summary).to_csv(os.path.join(a.out_dir, "asr_summary.csv"), index=False)
        print(f" -> {os.path.join(a.out_dir, 'asr_summary.csv')}")
    elif a.project:
        print("\n(dry run: nothing written; add --write to save the corrected tensors)")


if __name__ == "__main__":
    main()
