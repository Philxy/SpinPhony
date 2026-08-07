import sys
import h5py


def check(path):
    with h5py.File(path, 'r') as f:
        print(f"\n{path}")
        print("  keys:", list(f.keys()))
        if 'distances' in f:
            d = f['distances'][:]
            print(f"  distances: shape={d.shape}")
            print(f"  first 5: {d[:5]}")
            print(f"  last 5:  {d[-5:]}")
        else:
            print("  NO 'distances' dataset present")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_h5_distances.py <file1.h5> [file2.h5 ...]")
        sys.exit(1)
    for p in sys.argv[1:]:
        check(p)
