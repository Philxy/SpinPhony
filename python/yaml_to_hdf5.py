import yaml
import h5py
import numpy as np
import time
import sys

def convert_yaml_to_hdf5(yaml_file, hdf5_file):
    print(f"Loading YAML file: {yaml_file}")
    print("This will take a few minutes (parsing text to dictionaries)...")
    
    t0 = time.time()
    with open(yaml_file, 'r') as f:
        # Use CSafeLoader which is implemented in C for a massive speedup over standard safe_load
        try:
            from yaml import CSafeLoader as Loader
        except ImportError:
            print("Warning: CSafeLoader not found. Falling back to slow SafeLoader.")
            from yaml import SafeLoader as Loader
        
        config = yaml.load(f, Loader=Loader)
        
    print(f"YAML parsed in {time.time() - t0:.2f} seconds.")

    print("Extracting global parameters...")
    mesh = np.array(config['mesh'], dtype=np.int32)
    nqpoint = config['nqpoint']
    natom = config['natom']
    reciprocal_lattice = np.array(config['reciprocal_lattice'], dtype=np.float64)

    # Extract atoms
    atom_masses = np.zeros(natom, dtype=np.float64)
    mag_moments = np.zeros(natom, dtype=np.float64)
    
    for idx, point in enumerate(config['points']):
        atom_masses[idx] = point['mass']
        mag = point['magnetic_moment']
        if isinstance(mag, list):
            mag_moments[idx] = np.linalg.norm(mag)
        else:
            mag_moments[idx] = float(mag)

    print("Extracting q-points and slicing dynamical matrices...")
    phon_branches = 3 * natom
    q_positions = np.zeros((nqpoint, 3), dtype=np.float64)
    dyn_mat_complex = np.zeros((nqpoint, phon_branches, phon_branches), dtype=np.complex128)

    for q_idx, p_node in enumerate(config['phonon']):
        q_positions[q_idx] = p_node['q-position']

        if 'dynamical_matrix' in p_node:
            # Convert flat lists to a float64 array, then slice directly into complex128
            dm_raw = np.array(p_node['dynamical_matrix'], dtype=np.float64)
            dyn_mat_complex[q_idx] = dm_raw[:, 0::2] + 1j * dm_raw[:, 1::2]

    print(f"Writing binary arrays to HDF5: {hdf5_file}...")
    with h5py.File(hdf5_file, 'w') as f:
        f.create_dataset('mesh', data=mesh)
        f.create_dataset('nqpoint', data=nqpoint)
        f.create_dataset('natom', data=natom)
        f.create_dataset('reciprocal_lattice', data=reciprocal_lattice)
        f.create_dataset('atom_masses', data=atom_masses)
        f.create_dataset('mag_moments', data=mag_moments)
        f.create_dataset('q_positions', data=q_positions)
        
        # We apply gzip compression to the large complex matrices to save disk space
        f.create_dataset('dynamical_matrices', data=dyn_mat_complex, compression="gzip")

    print(f"Success! HDF5 file saved in {time.time() - t0:.2f} seconds total.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_hdf5.py <input.yaml> <output.h5>")
        sys.exit(1)
        
    convert_yaml_to_hdf5(sys.argv[1], sys.argv[2])