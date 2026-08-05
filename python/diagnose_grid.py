import sys
import numpy as np
import h5py


def diagnose_grid(hdf5_path):
    with h5py.File(hdf5_path, 'r') as f:
        mesh = f['mesh'][:]
        N = int(f['nqpoint'][()])
        q_frac = f['q_positions'][:]

    mesh = mesh.astype(np.int64)
    expected = int(mesh[0]) * int(mesh[1]) * int(mesh[2])
    print(f"mesh = {tuple(mesh)}  ->  expected full-grid points = {expected}")
    print(f"nqpoint in file       = {N}")

    grid_map = np.full((mesh[0], mesh[1], mesh[2]), -1, dtype=np.int64)
    collisions = []

    for q_idx in range(N):
        pos = np.round(q_frac[q_idx] * mesh).astype(np.int64) % mesh
        cell = tuple(pos)
        if grid_map[cell] != -1:
            collisions.append((q_idx, int(grid_map[cell]), cell, q_frac[q_idx].tolist()))
        grid_map[cell] = q_idx

    missing_cells = np.argwhere(grid_map == -1)
    print(f"\nFilled cells:   {expected - len(missing_cells)} / {expected}")
    print(f"Missing cells:  {len(missing_cells)}")
    print(f"Colliding q-points (mapped onto an already-filled cell): {len(collisions)}")

    if len(missing_cells) > 0:
        print("\nFirst 20 missing grid cells (integer indices -> fractional coords):")
        for cell in missing_cells[:20]:
            frac = cell / mesh
            print(f"  {tuple(int(c) for c in cell)}  ->  frac = {frac}")
        if len(missing_cells) > 20:
            print(f"  ... and {len(missing_cells) - 20} more")

    if collisions:
        print("\nFirst 20 colliding q-points (new_idx, existing_idx_at_that_cell, cell, new_q_frac):")
        for c in collisions[:20]:
            print(f"  q_idx={c[0]} collides with q_idx={c[1]} at cell {c[2]}, frac={c[3]}")
        if len(collisions) > 20:
            print(f"  ... and {len(collisions) - 20} more")

    if len(missing_cells) == 0 and not collisions:
        print("\nGrid is complete and non-degenerate - no issue found here.")

    return missing_cells, collisions


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python diagnose_grid.py <grid.h5>")
        sys.exit(1)
    diagnose_grid(sys.argv[1])
