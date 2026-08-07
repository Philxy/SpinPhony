"""
Prints the dense file's own qx,qy,qz at each high-symmetry label (using its
trusted path_labels comment + path_dist column), so it can be compared
directly against the sparse file's raw q-vectors at the same nominal labels
(e.g. from inspect_sparse_qpoints.py's output: G=(0,0,0), K~(0.656,0.312,
0.312), M=(0.5,0,0), G=(0,0,0)).

If the dense file's M doesn't match (0.5, 0, 0) (up to snapping precision),
that's the bug: dense and sparse are using different, symmetry-equivalent
but differently-located high-symmetry points, so any geometric reference
line built from dense's labels for K-M/M-Gamma won't be near sparse's real
K-M/M-Gamma points at all.

Usage:
    python inspect_dense_label_anchors.py <dense_csv>
"""
import sys
import numpy as np
import pandas as pd


def load_path_csv(path):
    with open(path) as f:
        first_line = f.readline().strip()
    labels = []
    if first_line.startswith("# path_labels:"):
        payload = first_line[len("# path_labels:"):].strip()
        if payload and payload != "unavailable":
            for pair in payload.split(","):
                name, val = pair.split("=")
                labels.append((name, float(val)))
        df = pd.read_csv(path, skiprows=1)
    else:
        df = pd.read_csv(path)
    return df, labels


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_dense_label_anchors.py <dense_csv>")
        sys.exit(1)

    df, labels = load_path_csv(sys.argv[1])
    if not labels:
        print("No path_labels found in this file.")
        sys.exit(1)

    q_cols = ["qx", "qy", "qz"]
    native_dist = df["path_dist"].to_numpy(dtype=float)

    print(f"{'label':>6} {'path_dist':>10} {'qx':>10} {'qy':>10} {'qz':>10}")
    for name, dist in labels:
        idx = np.argmin(np.abs(native_dist - dist))
        row = df.iloc[idx]
        print(f"{name:>6} {dist:>10.6f} {row['qx']:>10.5f} {row['qy']:>10.5f} {row['qz']:>10.5f}")


if __name__ == "__main__":
    main()
