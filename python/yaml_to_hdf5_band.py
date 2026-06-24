import yaml
import h5py
import numpy as np
import time
import sys
import json

def convert_band_yaml_to_hdf5(yaml_file, hdf5_file):
    print(f"Loading YAML file: {yaml_file}")
    print("This will take a few minutes (parsing text to dictionaries)...")

    t0 = time.time()
    with open(yaml_file, 'r') as f:
        # Use CLoader for massive speedup over standard loaders
        try:
            from yaml import CLoader as Loader
        except ImportError:
            print("Warning: CLoader not found. Falling back to slow SafeLoader.")
            from yaml import SafeLoader as Loader
        
        config = yaml.load(f, Loader=Loader)

    print(f"YAML parsed in {time.time() - t0:.2f} seconds.")

    print("Extracting global parameters...")
    natom = config['natom']
    nqpoint = config['nqpoint']
    npath = config.get('npath', 0)
    
    # Structural parameters
    lattice = np.array(config.get('lattice', []), dtype=np.float64)
    reciprocal_lattice = np.array(config.get('reciprocal_lattice', []), dtype=np.float64)
    segment_nqpoint = np.array(config.get('segment_nqpoint', []), dtype=np.int32)
    
    # Store labels as a JSON string to easily handle the ragged list of lists (e.g. [['G', 'K'], ...])
    labels_json = json.dumps(config.get('labels', []))

    print("Extracting atom data...")
    atom_masses = np.zeros(natom, dtype=np.float64)
    mag_moments = np.zeros(natom, dtype=np.float64)
    coordinates = np.zeros((natom, 3), dtype=np.float64)
    symbols = []

    for idx, point in enumerate(config['points']):
        atom_masses[idx] = point['mass']
        coordinates[idx] = point['coordinates']
        symbols.append(point['symbol'])
        
        mag = point.get('magnetic_moment', 0.0)
        if isinstance(mag, list):
            mag_moments[idx] = np.linalg.norm(mag)
        else:
            mag_moments[idx] = float(mag)

    print("Extracting phonon bands, frequencies, and eigenvectors...")
    # Dynamically determine number of bands from the first q-point
    phon_bands = len(config['phonon'][0]['band'])
    
    q_positions = np.zeros((nqpoint, 3), dtype=np.float64)
    frequencies = np.zeros((nqpoint, phon_bands), dtype=np.float64)
    
    # Check what optional band data is available
    has_gv = 'group_velocity' in config['phonon'][0]['band'][0]
    has_evec = 'eigenvector' in config['phonon'][0]['band'][0]

    if has_gv:
        group_velocities = np.zeros((nqpoint, phon_bands, 3), dtype=np.float64)
    if has_evec:
        eigenvectors = np.zeros((nqpoint, phon_bands, natom, 3), dtype=np.complex128)

    for q_idx, p_node in enumerate(config['phonon']):
        if 'q-position' in p_node:
            q_positions[q_idx] = p_node['q-position']
            
        for b_idx, band in enumerate(p_node['band']):
            frequencies[q_idx, b_idx] = band['frequency']
            
            if has_gv and 'group_velocity' in band:
                group_velocities[q_idx, b_idx] = band['group_velocity']
                
            if has_evec and 'eigenvector' in band:
                # Raw eigenvector is shape (natom, 3, 2) where 2 is [real, imaginary]
                evec_raw = np.array(band['eigenvector'], dtype=np.float64)
                eigenvectors[q_idx, b_idx] = evec_raw[:, :, 0] + 1j * evec_raw[:, :, 1]

    print(f"Writing binary arrays to HDF5: {hdf5_file}...")
    with h5py.File(hdf5_file, 'w') as f:
        # Globals
        f.create_dataset('natom', data=natom)
        f.create_dataset('nqpoint', data=nqpoint)
        f.create_dataset('npath', data=npath)
        f.create_dataset('labels_json', data=labels_json)
        
        if lattice.size > 0:
            f.create_dataset('lattice', data=lattice)
        if reciprocal_lattice.size > 0:
            f.create_dataset('reciprocal_lattice', data=reciprocal_lattice)
        if segment_nqpoint.size > 0:
            f.create_dataset('segment_nqpoint', data=segment_nqpoint)

        # Points
        f.create_dataset('atom_masses', data=atom_masses)
        f.create_dataset('mag_moments', data=mag_moments)
        f.create_dataset('coordinates', data=coordinates)
        
        # Save string array for symbols
        dt_str = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('symbols', data=np.array(symbols, dtype=object), dtype=dt_str)

        # Phonon Data
        f.create_dataset('q_positions', data=q_positions)
        f.create_dataset('frequencies', data=frequencies, compression="gzip")
        
        if has_gv:
            f.create_dataset('group_velocities', data=group_velocities, compression="gzip")
        if has_evec:
            f.create_dataset('eigenvectors', data=eigenvectors, compression="gzip")

    print(f"Success! HDF5 file saved in {time.time() - t0:.2f} seconds total.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_band_to_hdf5.py <input_band.yaml> <output.h5>")
        sys.exit(1)

    convert_band_yaml_to_hdf5(sys.argv[1], sys.argv[2])