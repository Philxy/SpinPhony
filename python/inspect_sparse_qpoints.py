"""
Pure diagnostic - no remapping, no plotting. Dumps the sparse (lifetime)
file's raw qx,qy,qz across its full row range, and its |q| (Euclidean norm,
fractional units), so we can see directly whether points nominally in the
K-M / M-Gamma segments actually move away from Gamma or stay clustered near
it (which would point to a bug upstream of this script, in how the sparse
path's q-points were generated - e.g. mesh_to_QPOINTS.py / combine_gc.py /
yaml_to_hdf5_band.py - rather than anything fixable in post-processing).

Usage:
    python inspect_sparse_qpoints.py <lifetime_csv> [segment_nqpoint ...]

segment_nqpoint (optional): the per-segment point counts straight from the
band.yaml (e.g. 11 6 16 for this run), used only to print which *nominal*
segment (by row index, not by any distance/label) each printed point belongs
to - purely informational.
"""
import sys
import numpy as np
import pandas as pd


def load_qxyz(path):
    with open(path) as f:
        first_line = f.readline().strip()
    if first_line.startswith("# path_labels:"):
        df = pd.read_csv(path, skiprows=1)
    else:
        df = pd.read_csv(path)
    return df


def nominal_segment(q_idx, segment_nqpoint):
    if not segment_nqpoint:
        return "?"
    cum = np.cumsum(segment_nqpoint)
    for i, c in enumerate(cum):
        if q_idx < c:
            return i
    return len(segment_nqpoint) - 1


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_sparse_qpoints.py <lifetime_csv> [seg1 seg2 ...]")
        sys.exit(1)

    path = sys.argv[1]
    segment_nqpoint = [int(x) for x in sys.argv[2:]]

    df = load_qxyz(path)
    q_cols = ["qx", "qy", "qz"]

    # One row per unique q_idx (file has one row per branch at each q_idx).
    unique_q = df.drop_duplicates(subset=["q_idx"]).sort_values("q_idx")

    print(f"{'q_idx':>6} {'seg':>4} {'qx':>10} {'qy':>10} {'qz':>10} {'|q|':>10}")
    for _, row in unique_q.iterrows():
        q_idx = int(row["q_idx"])
        qvec = row[q_cols].to_numpy(dtype=float)
        norm = np.linalg.norm(qvec)
        seg = nominal_segment(q_idx, segment_nqpoint)
        print(f"{q_idx:>6} {seg:>4} {qvec[0]:>10.5f} {qvec[1]:>10.5f} {qvec[2]:>10.5f} {norm:>10.5f}")


if __name__ == "__main__":
    main()
