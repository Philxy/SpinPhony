import numpy as np
import yaml
import os
from numba import cuda
import cmath
import math
import h5py


class CrystalDataSoA:
    def __init__(self, hdf5_filename, exchange_filename, slc_files=None, lattice_constant=1.0, anisotropy=0.01):
        # 1. Parse Jij Data
        self.jij_interactions = np.loadtxt(exchange_filename, delimiter=',', skiprows=1)

        # 2. Instant Binary Load via HDF5
        with h5py.File(hdf5_filename, 'r') as f:
            self.mesh = f['mesh'][:]
            self.N = f['nqpoint'][()]
            self.l_atoms = f['natom'][()]
            self.reciprocal_lattice = f['reciprocal_lattice'][:]
            self.atom_masses = f['atom_masses'][:]
            self.mag_moments = f['mag_moments'][:]
            q_frac_positions = f['q_positions'][:]
            self.dyn_mat_phon = f['dynamical_matrices'][:]

        self.phon_branches = 3 * self.l_atoms
        self.mag_indices = np.where(np.abs(self.mag_moments) > 1e-2)[0]
        self.n_mag_branches = len(self.mag_indices)

        # 3. Reconstruct Grid Maps and Phonon Eigendecomposition
        self.q_grid = np.zeros((self.N, 3), dtype=np.int32)
        self.grid_map = np.full((self.mesh[0], self.mesh[1], self.mesh[2]), -1, dtype=np.int32)
        
        self.w_phon = np.zeros((self.N, self.phon_branches), dtype=np.float64)
        self.eig_phon = np.zeros((self.N, self.phon_branches, self.l_atoms, 3), dtype=np.complex128)

        # Process the q-grid mapping
        for q_idx in range(self.N):
            grid_pos = np.round(q_frac_positions[q_idx] * self.mesh).astype(np.int32) % self.mesh
            self.q_grid[q_idx] = grid_pos
            self.grid_map[grid_pos[0], grid_pos[1], grid_pos[2]] = q_idx
            
            # Diagonalize the pre-sliced complex dynamical matrix
            dm_complex = self.dyn_mat_phon[q_idx]
            eigenvalues, eigenvectors = np.linalg.eigh(dm_complex)
            
            for b in range(self.phon_branches):
                ev = eigenvalues[b]
                self.w_phon[q_idx, b] = (np.sqrt(ev) if ev > 0 else -np.sqrt(-ev)) * 4.135667696 * 15.633302
                for atom in range(self.l_atoms):
                    self.eig_phon[q_idx, b, atom, 0] = eigenvectors[3*atom + 0, b]
                    self.eig_phon[q_idx, b, atom, 1] = eigenvectors[3*atom + 1, b]
                    self.eig_phon[q_idx, b, atom, 2] = eigenvectors[3*atom + 2, b]

        if np.any(self.grid_map == -1):
            raise ValueError("Grid map initialization failed: Incomplete q-point mesh.")

        # Allocate and Compute Magnons
        self.w_mag = np.zeros((self.N, self.n_mag_branches), dtype=np.float64)
        self._compute_magnon_dispersions_test(K_anisotropy=anisotropy, lattice_constant=lattice_constant)

        # Cartesian conversion
        q_frac_array = self.q_grid / self.mesh
        self.q_grid_cart = np.dot(q_frac_array, self.reciprocal_lattice * 2.0 * np.pi)
        
        # 4. Parse SLC Tensors
        if slc_files and len(slc_files) == 3:
            self._parse_slc_tensors(slc_files[0], slc_files[1], slc_files[2], lattice_constant)

    def print_summary(self):
        """Prints a verification summary of the loaded SoA data."""
        print("\n" + "="*50)
        print(" CrystalDataSoA Initialization Summary")
        print("="*50)
        print(f" Grid Mesh         : {self.mesh[0]}x{self.mesh[1]}x{self.mesh[2]} ({self.N} points)")
        print(f" Atoms             : {self.l_atoms} total, {self.n_mag_branches} magnetic")
        print(f" Phonon Branches   : {self.phon_branches}")
        print(f" Magnon Branches   : {self.n_mag_branches}")
        print(f" Jij Elements      : {self.jij_interactions.shape[0]}")
        
        if hasattr(self, 'slc_axis'):
            print(f" SLC Tensors       : {self.slc_axis.shape[0]} interactions loaded")
            print(f"   -> X disps      : {np.sum(self.slc_axis == 0)}")
            print(f"   -> Y disps      : {np.sum(self.slc_axis == 1)}")
            print(f"   -> Z disps      : {np.sum(self.slc_axis == 2)}")
            
        print(" Memory Allocation : Contiguous NumPy arrays ready for GPU")
        print("="*50 + "\n")

    def _parse_slc_tensors(self, file_x, file_y, file_z, lattice_constant):
        BOHR_TO_ANGSTROM = 0.529177210903
        
        temp_axis, temp_rij, temp_rik, temp_J, temp_types = [], [], [], [], []
        
        def process_file(filepath, axis_code):
            if not os.path.exists(filepath):
                raise RuntimeError(f"Error: Could not open SLC file {filepath}")
                
            with open(filepath, 'r') as f:
                lines = f.readlines()[1:] # Skip header
                
            for line in lines:
                # Replace commas with spaces and split
                parts = line.replace(',', ' ').split()
                if len(parts) < 16:
                    continue
                
                # 1. Coordinates
                rij = [float(parts[0]), float(parts[1]), float(parts[2])]
                rik = [float(parts[3]), float(parts[4]), float(parts[5])]
                
                # 2. J Tensor (3x3)
                J = [
                    [float(parts[6]), float(parts[7]), float(parts[8])],
                    [float(parts[9]), float(parts[10]), float(parts[11])],
                    [float(parts[12]), float(parts[13]), float(parts[14])]
                ]
                
                # 3. Adaptive Type Parsing
                val1 = int(parts[15])
                if len(parts) >= 18:
                    # New Format
                    type_i, type_j, displaced_type = val1, int(parts[16]), int(parts[17])
                else:
                    # Old Format
                    type_i, type_j, displaced_type = 1, 1, val1
                
                # Append to temp lists
                temp_axis.append(axis_code)
                temp_rij.append(rij)
                temp_rik.append(rik)
                temp_J.append(J)
                temp_types.append([type_i, type_j, displaced_type])

        # Process all three files
        process_file(file_x, 0) # X axis
        process_file(file_y, 1) # Y axis
        process_file(file_z, 2) # Z axis
        
        # Convert to strict NumPy arrays and apply unit conversions
        self.slc_axis = np.array(temp_axis, dtype=np.int32)
        self.slc_rij = np.array(temp_rij, dtype=np.float64) * lattice_constant
        self.slc_rik = np.array(temp_rik, dtype=np.float64) * lattice_constant
        self.slc_J = np.array(temp_J, dtype=np.float64) / BOHR_TO_ANGSTROM
        self.slc_types = np.array(temp_types, dtype=np.int32)

    def _parse_phonons(self, phonon_list):
        for q_idx, p_node in enumerate(phonon_list):
            q_frac = np.array(p_node['q-position'], dtype=np.float64)
            grid_pos = np.round(q_frac * self.mesh).astype(np.int32) % self.mesh
            
            self.q_grid[q_idx] = grid_pos
            self.grid_map[grid_pos[0], grid_pos[1], grid_pos[2]] = q_idx
            
            dim = self.phon_branches
            if 'dynamical_matrix' in p_node:
                dm_raw = p_node['dynamical_matrix']
                dm_complex = np.zeros((dim, dim), dtype=np.complex128)
                for r in range(dim):
                    for c in range(dim):
                        dm_complex[r, c] = complex(dm_raw[r][2 * c], dm_raw[r][2 * c + 1]) 
                
                self.dyn_mat_phon[q_idx] = dm_complex

                # Conversion Factors
                # 1 VASP Unit = (eV / (Ang^2 * AMU))
                # Sqrt(1 VASP Unit) -> THz:  Factor = 15.633302
                # THz -> meV:                Factor = 4.135667696
                #const double VASP_TO_THz = 15.633302;
                #const double THz_TO_meV = 4.135667696;

                eigenvalues, eigenvectors = np.linalg.eigh(dm_complex)
                for b in range(dim):
                    ev = eigenvalues[b]
                    self.w_phon[q_idx, b] = (np.sqrt(ev) if ev > 0 else -np.sqrt(-ev)) * 4.135667696 * 15.633302
                    for atom in range(self.l_atoms):
                        self.eig_phon[q_idx, b, atom, 0] = eigenvectors[3*atom + 0, b]
                        self.eig_phon[q_idx, b, atom, 1] = eigenvectors[3*atom + 1, b]
                        self.eig_phon[q_idx, b, atom, 2] = eigenvectors[3*atom + 2, b]
            
            """
            elif 'band' in p_node:
                for b_idx, band in enumerate(p_node['band']):
                    self.w_phon[q_idx, b_idx] = band['frequency'] * 4.135667696
                    if 'eigenvector' in band:
                        for atom, eig_vec in enumerate(band['eigenvector']):
                            self.eig_phon[q_idx, b_idx, atom, 0] = complex(eig_vec[0][0], eig_vec[0][1])
                            self.eig_phon[q_idx, b_idx, atom, 1] = complex(eig_vec[1][0], eig_vec[1][1])
                            self.eig_phon[q_idx, b_idx, atom, 2] = complex(eig_vec[2][0], eig_vec[2][1])
            """

        if np.any(self.grid_map == -1):
            raise ValueError("Grid map initialization failed: Incomplete q-point mesh.")
            
    def push_to_gpu(self):
        from numba import cuda
        
        # Dictionary to store pointers
        gpu_buffers = {}
        total_vram_used = 0
        
        # Helper to push and track
        def track_and_push(name, arr):
            nonlocal total_vram_used
            # Calculate bytes (size * itemsize)
            bytes_size = arr.nbytes
            total_vram_used += bytes_size
            print(f"Pushing {name:15s} | Size: {arr.shape} | Footprint: {bytes_size/1e6:8.2f} MB")
            return cuda.to_device(arr)

        gpu_buffers["mesh"] = track_and_push("mesh", self.mesh)
        gpu_buffers["q_grid"] = track_and_push("q_grid", self.q_grid)
        gpu_buffers["q_grid_cart"] = track_and_push("q_grid_cart", self.q_grid_cart)
        gpu_buffers["grid_map"] = track_and_push("grid_map", self.grid_map)
        gpu_buffers["w_phon"] = track_and_push("w_phon", self.w_phon)
        gpu_buffers["eig_phon"] = track_and_push("eig_phon", self.eig_phon)
        gpu_buffers["eig_mag"] = track_and_push("eig_mag", self.eig_mag)
        gpu_buffers["dyn_mat_phon"] = track_and_push("dyn_mat_phon", self.dyn_mat_phon)
        gpu_buffers["w_mag"] = track_and_push("w_mag", self.w_mag)
        gpu_buffers["jij"] = track_and_push("jij", self.jij_interactions)
        gpu_buffers["atom_masses"] = track_and_push("atom_masses", self.atom_masses)
        gpu_buffers["mag_moments"] = track_and_push("mag_moments", self.mag_moments)
        
        if hasattr(self, 'slc_axis'):
            gpu_buffers["slc_axis"] = track_and_push("slc_axis", self.slc_axis)
            gpu_buffers["slc_rij"] = track_and_push("slc_rij", self.slc_rij)
            gpu_buffers["slc_rik"] = track_and_push("slc_rik", self.slc_rik)
            gpu_buffers["slc_J"] = track_and_push("slc_J", self.slc_J)
            gpu_buffers["slc_types"] = track_and_push("slc_types", self.slc_types)

        print(f"-------------------------------------------")
        print(f"TOTAL VRAM ALLOCATED: {total_vram_used/1e6:.2f} MB")
        
        return gpu_buffers
    

    def _compute_magnon_dispersions(self, K_anisotropy=0.01, lattice_constant=1.0):
        """
        Calculates exact magnon energies matching the Ferromagnetic C++ implementation.
        """
        self.eig_mag = np.zeros((self.N, 2*self.n_mag_branches, 2*self.n_mag_branches), dtype=np.complex128)
        
        atom_to_mag = np.full(self.l_atoms, -1, dtype=np.int32)
        for i, m_idx in enumerate(self.mag_indices):
            atom_to_mag[m_idx] = i

        S_eff = np.abs(self.mag_moments[self.mag_indices]) / 2.0
        
        # Precompute real J(0)
        J_0 = np.zeros((self.n_mag_branches, self.n_mag_branches), dtype=np.float64)
        for row in self.jij_interactions:
            i, j = int(row[4]) - 1, int(row[5]) - 1
            mag_i, mag_j = atom_to_mag[i], atom_to_mag[j]
            if mag_i != -1 and mag_j != -1:
                J_0[mag_i, mag_j] += row[3]
                
        for q_idx in range(self.N):
            # 1. Recover fractional coordinates from integer grid
            q_frac = self.q_grid[q_idx] / self.mesh
            
            # 2. Convert to Cartesian wavevector (includes 2*pi factor)
            # Units: 1 / Angstrom (assuming reciprocal_lattice is in 1/A)
            q_cart = np.dot(q_frac, self.reciprocal_lattice * 2.0 * np.pi)
            
            J_q = np.zeros((self.n_mag_branches, self.n_mag_branches), dtype=np.complex128)
            for row in self.jij_interactions:
                i, j = int(row[4]) - 1, int(row[5]) - 1
                mag_i, mag_j = atom_to_mag[i], atom_to_mag[j]
                if mag_i == -1 or mag_j == -1: 
                    continue
                
                # 3. Extract Cartesian connection vector and scale by lattice constant
                # Units: Angstroms
                r_cart = np.array([row[0], row[1], row[2]]) * lattice_constant
                
                # 4. Cartesian dot product yields a dimensionless phase
                phase = np.dot(q_cart, r_cart)
                J_q[mag_i, mag_j] += row[3] * cmath.exp(1j * phase)
                
            # Ferromagnetic Hamiltonian: Omega = S * (J_k - sum(J_0))
            Omega_k = np.zeros((self.n_mag_branches, self.n_mag_branches), dtype=np.complex128)
            for n in range(self.n_mag_branches):
                sum_J_0 = np.sum(J_0[n, :])
                for m in range(self.n_mag_branches):
                    if n == m:
                        Omega_k[n, n] = S_eff[n] * (J_q[n, n] - sum_J_0)
                    else:
                        Omega_k[n, m] = S_eff[n] * J_q[n, m]
                    
            # Build Bogoliubov-de Gennes Matrix 
            H_BdG = np.zeros((2*self.n_mag_branches, 2*self.n_mag_branches), dtype=np.complex128)
            for n in range(self.n_mag_branches):
                for m in range(self.n_mag_branches):
                    # Applying the -1 flip for ferromagnets as done in C++
                    val = -Omega_k[n, m] 
                    if n == m:
                        val += K_anisotropy
                    H_BdG[n, m] = val
                    H_BdG[n + self.n_mag_branches, m + self.n_mag_branches] = np.conj(val)
            
            try:
                energies, para_unitary = diagonalize_bosonic_hamiltonian(H_BdG)
                self.w_mag[q_idx] = energies
                self.eig_mag[q_idx] = para_unitary
            except RuntimeError as e:
                print(f"Warning at q_idx {q_idx}: {e}")
                self.w_mag[q_idx] = np.zeros(self.n_mag_branches)


    def _compute_magnon_dispersions_test(self, K_anisotropy=0.5, lattice_constant=1.0):
            self.eig_mag = np.zeros((self.N, 2*self.n_mag_branches, 2*self.n_mag_branches), dtype=np.complex128)
            
            atom_to_mag = np.full(self.l_atoms, -1, dtype=np.int32)
            for i, m_idx in enumerate(self.mag_indices):
                atom_to_mag[m_idx] = i

            moments = self.mag_moments[self.mag_indices]
            S_eff = np.abs(moments) / 2.0
            sigma = np.sign(moments) 
            
            # 1. Pre-extract all valid Jij connections
            valid_bonds = []
            J_0 = np.zeros((self.n_mag_branches, self.n_mag_branches), dtype=np.float64)
            for row in self.jij_interactions:
                i, j = int(row[4]) - 1, int(row[5]) - 1
                mag_i, mag_j = atom_to_mag[i], atom_to_mag[j]
                if mag_i != -1 and mag_j != -1:
                    J_0[mag_i, mag_j] += row[3]
                    valid_bonds.append((mag_i, mag_j, row[0], row[1], row[2], row[3]))

            J_0 = (J_0 + J_0.T) / 2.0 # Force strict symmetry
            
            # Convert to fast NumPy arrays
            mag_i_arr = np.array([b[0] for b in valid_bonds])
            mag_j_arr = np.array([b[1] for b in valid_bonds])
            r_cart_arr = np.array([b[2:5] for b in valid_bonds]) * lattice_constant
            J_val_arr = np.array([b[5] for b in valid_bonds])

            # 2. Compute q_cart for ALL q-points at once (Shape: N_points x 3)
            q_frac = self.q_grid / self.mesh
            q_cart_all = np.dot(q_frac, self.reciprocal_lattice * 2.0 * np.pi)

            # 3. Vectorized Phase Calculation
            # np.dot( (N x 3), (3 x N_bonds) ) -> phases is (N x N_bonds)
            phases = np.dot(q_cart_all, r_cart_arr.T)
            
            # Compute exp(i * phase) * J for all q-points and bonds simultaneously
            exp_phases = np.exp(1j * phases) * J_val_arr

            # 4. Accumulate J_q for all q-points
            J_q_all = np.zeros((self.N, self.n_mag_branches, self.n_mag_branches), dtype=np.complex128)
            for b_idx in range(len(valid_bonds)):
                mi, mj = mag_i_arr[b_idx], mag_j_arr[b_idx]
                J_q_all[:, mi, mj] += exp_phases[:, b_idx]
                
            # Symmetrize all J_q matrices simultaneously to kill floating point noise
            J_q_all = (J_q_all + np.transpose(J_q_all.conj(), axes=(0, 2, 1))) / 2.0

            # 5. Build BdG and Diagonalize (Now the loop only handles small 4x4 matrices)
            for q_idx in range(self.N):
                J_q = J_q_all[q_idx]
                
                A_mat = np.zeros((self.n_mag_branches, self.n_mag_branches), dtype=np.complex128)
                B_mat = np.zeros((self.n_mag_branches, self.n_mag_branches), dtype=np.complex128)
                    
                for n in range(self.n_mag_branches):
                    sum_J_0 = np.sum([J_0[n, m] * S_eff[m] * (sigma[n] * sigma[m]) for m in range(self.n_mag_branches)])
                    for m in range(self.n_mag_branches):
                        if n == m:
                            A_mat[n, n] = sum_J_0 - S_eff[n] * J_q[n, n] + K_anisotropy
                        else:
                            if sigma[n] == sigma[m]:
                                A_mat[n, m] = -np.sqrt(S_eff[n] * S_eff[m]) * J_q[n, m]
                            else:
                                B_mat[n, m] = -np.sqrt(S_eff[n] * S_eff[m]) * J_q[n, m]
                                
                H_BdG = np.zeros((2*self.n_mag_branches, 2*self.n_mag_branches), dtype=np.complex128)
                H_BdG[:self.n_mag_branches, :self.n_mag_branches] = A_mat
                H_BdG[self.n_mag_branches:, self.n_mag_branches:] = np.conj(A_mat)
                H_BdG[:self.n_mag_branches, self.n_mag_branches:] = B_mat
                H_BdG[self.n_mag_branches:, :self.n_mag_branches] = np.conj(B_mat.T)

                # Positive Definiteness Enforcement
                min_eig = np.min(np.linalg.eigvalsh(H_BdG))
                if min_eig <= 1e-8:
                    np.fill_diagonal(H_BdG, H_BdG.diagonal() + np.abs(min_eig) + 1e-5)

                try:
                    energies, para_unitary = diagonalize_bosonic_hamiltonian(H_BdG)
                    self.w_mag[q_idx] = energies
                    self.eig_mag[q_idx] = para_unitary
                except RuntimeError as e:
                    print(f"Warning at q_idx {q_idx}: {e}")
                    self.w_mag[q_idx] = np.zeros(self.n_mag_branches)


    def load_and_evaluate_path_hdf5(self, hdf5_path_file, K_anisotropy=0.01, lattice_constant=1.0):
        """
        Loads pre-calculated phonon band structures from an HDF5 file and 
        computes exact magnon energies along the same high-symmetry path.
        """
        import json
        
        print(f"\nLoading explicit high-symmetry path from HDF5: {hdf5_path_file}")
        
        # 1. Instant Binary Load of Phonon Path Data
        with h5py.File(hdf5_path_file, 'r') as f:
            self.N_path = f['nqpoint'][()]
            
            self.path_q_frac = f['q_positions'][:]
            # Convert to Cartesian wavevectors
            self.path_q_cart = np.dot(self.path_q_frac, self.reciprocal_lattice * 2.0 * np.pi)
            
            # Phonopy frequencies are in THz. Convert to meV.
            # 1 THz = 4.135667696 meV
            raw_frequencies = f['frequencies'][:]
            self.path_w_phon = raw_frequencies * 4.135667696
            
            if 'eigenvectors' in f:
                # Shape is exactly what the GPU expects: (N_path, phon_branches, l_atoms, 3) complex128
                self.path_eig_phon = f['eigenvectors'][:]
            else:
                raise ValueError("Eigenvectors missing from HDF5. Make sure 'EIGENVECTORS = .TRUE.' was used.")
            
            # Optional: Keep track of symmetry labels for plotting later
            if 'labels_json' in f:
                self.path_labels = json.loads(f['labels_json'][()])
            if 'segment_nqpoint' in f:
                self.path_segments = f['segment_nqpoint'][:]

        # 2. Allocate Path Arrays for Magnons
        self.path_w_mag = np.zeros((self.N_path, self.n_mag_branches), dtype=np.float64)
        self.path_eig_mag = np.zeros((self.N_path, 2*self.n_mag_branches, 2*self.n_mag_branches), dtype=np.complex128)

        # 3. Compute Exact Magnons for the Path
        atom_to_mag = np.full(self.l_atoms, -1, dtype=np.int32)
        for i, m_idx in enumerate(self.mag_indices):
            atom_to_mag[m_idx] = i

        S_eff = np.abs(self.mag_moments[self.mag_indices]) / 2.0
        
        # Precompute real J(0)
        J_0 = np.zeros((self.n_mag_branches, self.n_mag_branches), dtype=np.float64)
        for row in self.jij_interactions:
            i, j = int(row[4]) - 1, int(row[5]) - 1
            mag_i, mag_j = atom_to_mag[i], atom_to_mag[j]
            if mag_i != -1 and mag_j != -1:
                J_0[mag_i, mag_j] += row[3]
                
        for q_idx in range(self.N_path):
            q_cart = self.path_q_cart[q_idx]
            J_q = np.zeros((self.n_mag_branches, self.n_mag_branches), dtype=np.complex128)
            
            for row in self.jij_interactions:
                i, j = int(row[4]) - 1, int(row[5]) - 1
                mag_i, mag_j = atom_to_mag[i], atom_to_mag[j]
                if mag_i == -1 or mag_j == -1: 
                    continue
                
                r_cart = np.array([row[0], row[1], row[2]]) * lattice_constant
                phase = np.dot(q_cart, r_cart)
                J_q[mag_i, mag_j] += row[3] * cmath.exp(1j * phase)
                
            Omega_k = np.zeros((self.n_mag_branches, self.n_mag_branches), dtype=np.complex128)
            for n in range(self.n_mag_branches):
                sum_J_0 = np.sum(J_0[n, :])
                for m in range(self.n_mag_branches):
                    if n == m:
                        Omega_k[n, n] = S_eff[n] * (J_q[n, n] - sum_J_0)
                    else:
                        Omega_k[n, m] = S_eff[n] * J_q[n, m]
                    
            H_BdG = np.zeros((2*self.n_mag_branches, 2*self.n_mag_branches), dtype=np.complex128)
            for n in range(self.n_mag_branches):
                for m in range(self.n_mag_branches):
                    val = -Omega_k[n, m] 
                    if n == m:
                        val += K_anisotropy
                    H_BdG[n, m] = val
                    H_BdG[n + self.n_mag_branches, m + self.n_mag_branches] = np.conj(val)
            
            energies, para_unitary = diagonalize_bosonic_hamiltonian(H_BdG)
            self.path_w_mag[q_idx] = energies
            self.path_eig_mag[q_idx] = para_unitary
            
        print(f"-> Evaluated {self.N_path} exact path points.")


    @cuda.jit(device=True)
    def calc_fourier_transform_vec(kpx, kpy, kpz, qx, qy, qz, slc_axis, slc_rij, slc_rik, slc_J, slc_types, n_type, m_type, l_type, mu_type, J_tilde_out):
        """Computes the FT tensor explicitly passing scalar Cartesian coordinates."""
        for a in range(3):
            for b in range(3):
                J_tilde_out[a, b] = 0.0 + 0.0j

        for i in range(slc_axis.shape[0]):
            if slc_axis[i] == mu_type:
                if slc_types[i, 0] == n_type and slc_types[i, 1] == m_type and slc_types[i, 2] == l_type:
                    phase_val = (kpx * slc_rij[i, 0] + kpy * slc_rij[i, 1] + kpz * slc_rij[i, 2]) + \
                                (qx * slc_rik[i, 0] + qy * slc_rik[i, 1] + qz * slc_rik[i, 2])
                    phase_factor = cmath.exp(1j * phase_val)
                    for a in range(3):
                        for b in range(3):
                            J_tilde_out[a, b] += slc_J[i, a, b] * phase_factor

    @cuda.jit(device=True)
    def calc_vertex_V_path(kpx, kpy, kpz, qx, qy, qz, gammax, gammay, gammaz, lambda_phon, n, m, 
                        slc_axis, slc_rij, slc_rik, slc_J, slc_types, 
                        eig_phon_q, omega, atom_masses, mag_moments):
        """Calculates the scattering vertex specifically for explicitly projected wavevectors."""
        if omega < 1e-1: return 0.0
        
        hbar = 0.6582119569 # meV * ps
        DALTON_TO_meV_PS2_PER_A2 = 0.10364269
        
        S_n = math.fabs(mag_moments[n] / 2.0 ) 
        S_m = math.fabs(mag_moments[m] / 2.0 )
        sigma_n = math.copysign(1.0, mag_moments[n]) if S_n > 0 else 0.0
        sigma_m = math.copysign(1.0, mag_moments[m]) if S_m > 0 else 0.0

        J_tilde_dyn = cuda.local.array((3, 3), dtype=np.complex128)
        J_tilde_stat = cuda.local.array((3, 3), dtype=np.complex128)
        V_complex = 0.0 + 0.0j
        
        num_atoms = atom_masses.shape[0]
        num_mag_branches = mag_moments.shape[0]

        for l in range(num_atoms):
            mass_l = atom_masses[l] * DALTON_TO_meV_PS2_PER_A2
            disp_amp = math.sqrt(hbar*hbar / (2.0 * mass_l * omega))
            
            for mu in range(3):
                e_mu = eig_phon_q[lambda_phon, l, mu]
                
                calc_fourier_transform_vec(kpx, kpy, kpz, qx, qy, qz, slc_axis, slc_rij, slc_rik, slc_J, slc_types, n + 1, m + 1, l + 1, mu, J_tilde_dyn)
                W_dynamic = (J_tilde_dyn[0, 0] + (sigma_n * sigma_m) * J_tilde_dyn[1, 1] - 1j * sigma_m * J_tilde_dyn[0, 1] + 1j * sigma_n * J_tilde_dyn[1, 0]) / math.sqrt(S_n * S_m)
                
                W_static = 0.0 + 0.0j
                if n == m: 
                    for mp in range(num_mag_branches):
                        if math.fabs(mag_moments[mp]) > 1e-2:
                            sigma_mp = math.copysign(1.0, mag_moments[mp])
                            calc_fourier_transform_vec(gammax, gammay, gammaz, qx, qy, qz, slc_axis, slc_rij, slc_rik, slc_J, slc_types, n + 1, mp + 1, l + 1, mu, J_tilde_stat)
                            W_static += (2.0 / S_n) * (sigma_n * sigma_mp) * J_tilde_stat[2, 2]
                            
                V_complex += disp_amp * e_mu * (W_dynamic - W_static)
                
        return (V_complex.real**2 + V_complex.imag**2)

    @cuda.jit
    def phase_1_scan_path(mesh, grid_q_frac, grid_q_cart, grid_map, 
                        path_q_frac, path_q_cart, path_w_phon, path_w_mag, path_eig_phon,
                        w_phon_grid, w_mag_grid, eig_phon_grid,
                        slc_axis, slc_rij, slc_rik, slc_J, slc_types, 
                        smearing, chan_indices, chan_weights, channel_count, 
                        atom_masses, mag_moments, gamma_idx):
        """Scans phase space integrating exact path points against the BZ grid."""
        path_idx, k_idx = cuda.grid(2)
        
        if path_idx >= path_q_frac.shape[0] or k_idx >= grid_q_frac.shape[0]: 
            return
            
        n_mag = path_w_mag.shape[1]
        n_phon = path_w_phon.shape[1]
        gaussian_norm = 1.0 / (smearing * 2.50662827463)
        cutoff = 3.0 * smearing
        
        # 1. Kinematics mapping: q(path) - k(grid) -> p(grid nearest neighbor)
        px_int = int(math.floor(((path_q_frac[path_idx, 0] - grid_q_frac[k_idx, 0]) % 1.0) * mesh[0] + 0.5)) % mesh[0]
        py_int = int(math.floor(((path_q_frac[path_idx, 1] - grid_q_frac[k_idx, 1]) % 1.0) * mesh[1] + 0.5)) % mesh[1]
        pz_int = int(math.floor(((path_q_frac[path_idx, 2] - grid_q_frac[k_idx, 2]) % 1.0) * mesh[2] + 0.5)) % mesh[2]
        idx_qmink = grid_map[px_int, py_int, pz_int]

        # 2. Kinematics mapping: k(grid) - q(path) -> p(grid nearest neighbor)
        px2_int = int(math.floor(((grid_q_frac[k_idx, 0] - path_q_frac[path_idx, 0]) % 1.0) * mesh[0] + 0.5)) % mesh[0]
        py2_int = int(math.floor(((grid_q_frac[k_idx, 1] - path_q_frac[path_idx, 1]) % 1.0) * mesh[1] + 0.5)) % mesh[1]
        pz2_int = int(math.floor(((grid_q_frac[k_idx, 2] - path_q_frac[path_idx, 2]) % 1.0) * mesh[2] + 0.5)) % mesh[2]
        idx_kminq = grid_map[px2_int, py2_int, pz2_int]

        gammax, gammay, gammaz = grid_q_cart[gamma_idx, 0], grid_q_cart[gamma_idx, 1], grid_q_cart[gamma_idx, 2]

        for n in range(n_mag):
            for m in range(n_mag):
                for lam in range(n_phon):
                    
                    # 0: Magnon Emission (Magnon on path -> Magnon + Phonon on grid)
                    dE = path_w_mag[path_idx, n] - w_mag_grid[k_idx, m] - w_phon_grid[idx_qmink, lam]
                    if abs(dE) < cutoff:
                        weight = gaussian_norm * math.exp(-0.5 * (dE*dE) / (smearing*smearing))
                        kpx, kpy, kpz = path_q_cart[path_idx, 0], path_q_cart[path_idx, 1], path_q_cart[path_idx, 2]
                        qx, qy, qz = grid_q_cart[idx_qmink, 0], grid_q_cart[idx_qmink, 1], grid_q_cart[idx_qmink, 2]
                        
                        V_sq = calc_vertex_V_path(kpx, kpy, kpz, qx, qy, qz, gammax, gammay, gammaz, lam, n, m,
                                                slc_axis, slc_rij, slc_rik, slc_J, slc_types, 
                                                eig_phon_grid[idx_qmink], w_phon_grid[idx_qmink, lam], atom_masses, mag_moments)
                        c_idx = cuda.atomic.add(channel_count, 0, 1)
                        if c_idx < chan_indices.shape[1]:
                            chan_indices[0, c_idx] = 0; chan_indices[1, c_idx] = path_idx; chan_indices[2, c_idx] = k_idx
                            chan_indices[3, c_idx] = idx_qmink; chan_indices[4, c_idx] = n; chan_indices[5, c_idx] = m; chan_indices[6, c_idx] = lam
                            chan_weights[c_idx] = V_sq * weight

                    # 1: Magnon Absorption (Magnon on path + Phonon on grid -> Magnon on grid)
                    dE = path_w_mag[path_idx, n] - w_mag_grid[k_idx, m] + w_phon_grid[idx_kminq, lam]
                    if abs(dE) < cutoff:
                        weight = gaussian_norm * math.exp(-0.5 * (dE*dE) / (smearing*smearing))
                        kpx, kpy, kpz = path_q_cart[path_idx, 0], path_q_cart[path_idx, 1], path_q_cart[path_idx, 2]
                        qx, qy, qz = grid_q_cart[idx_kminq, 0], grid_q_cart[idx_kminq, 1], grid_q_cart[idx_kminq, 2]
                        
                        V_sq = calc_vertex_V_path(kpx, kpy, kpz, qx, qy, qz, gammax, gammay, gammaz, lam, m, n,
                                                slc_axis, slc_rij, slc_rik, slc_J, slc_types, 
                                                eig_phon_grid[idx_kminq], w_phon_grid[idx_kminq, lam], atom_masses, mag_moments)
                        c_idx = cuda.atomic.add(channel_count, 0, 1)
                        if c_idx < chan_indices.shape[1]:
                            chan_indices[0, c_idx] = 1; chan_indices[1, c_idx] = path_idx; chan_indices[2, c_idx] = k_idx
                            chan_indices[3, c_idx] = idx_kminq; chan_indices[4, c_idx] = n; chan_indices[5, c_idx] = m; chan_indices[6, c_idx] = lam
                            chan_weights[c_idx] = V_sq * weight

                    # 2: Phonon Scattering (Phonon on path + Magnon on grid -> Magnon on grid)
                    dE = path_w_phon[path_idx, lam] + w_mag_grid[idx_kminq, m] - w_mag_grid[k_idx, n]
                    if abs(dE) < cutoff:
                        weight = gaussian_norm * math.exp(-0.5 * (dE*dE) / (smearing*smearing))
                        kpx, kpy, kpz = grid_q_cart[k_idx, 0], grid_q_cart[k_idx, 1], grid_q_cart[k_idx, 2]
                        qx, qy, qz = -path_q_cart[path_idx, 0], -path_q_cart[path_idx, 1], -path_q_cart[path_idx, 2]
                        
                        V_sq = calc_vertex_V_path(kpx, kpy, kpz, qx, qy, qz, gammax, gammay, gammaz, lam, n, m,
                                                slc_axis, slc_rij, slc_rik, slc_J, slc_types, 
                                                path_eig_phon[path_idx], path_w_phon[path_idx, lam], atom_masses, mag_moments)
                        c_idx = cuda.atomic.add(channel_count, 0, 1)
                        if c_idx < chan_indices.shape[1]:
                            chan_indices[0, c_idx] = 2; chan_indices[1, c_idx] = path_idx; chan_indices[2, c_idx] = k_idx
                            chan_indices[3, c_idx] = idx_kminq; chan_indices[4, c_idx] = n; chan_indices[5, c_idx] = m; chan_indices[6, c_idx] = lam
                            chan_weights[c_idx] = V_sq * weight

    @cuda.jit
    def phase_lifetime_path(chan_indices, chan_weights, num_channels, n_mag_grid, n_phon_grid, gamma_mag_path, gamma_phon_path, N_grid_points):
        """Calculates lifetimes for the path array by evaluating the thermal distributions on the regular grid."""
        idx = cuda.grid(1)
        if idx >= num_channels[0] or idx >= chan_weights.shape[0]: 
            return
            
        c_type = chan_indices[0, idx]
        path_idx = chan_indices[1, idx] 
        k_idx  = chan_indices[2, idx] 
        p_idx  = chan_indices[3, idx] 
        n      = chan_indices[4, idx]
        m      = chan_indices[5, idx]
        lam    = chan_indices[6, idx]
        V_sq   = chan_weights[idx]
        
        hbar = 0.6582119569 # meV * ps
        # Prefactor keeps the 1/N BZ normalization from the regular grid
        fgr_prefactor = (2.0 * math.pi / hbar) / N_grid_points
        
        num_mag_branches = n_mag_grid.shape[1]
        num_phon_branches = n_phon_grid.shape[1]
        
        if c_type == 0: 
            nk_mag = n_mag_grid[k_idx, m]
            n_qmink_ph = n_phon_grid[p_idx, lam]
            rate = fgr_prefactor * V_sq * (1.0 + n_qmink_ph + nk_mag)
            cuda.atomic.add(gamma_mag_path, path_idx * num_mag_branches + n, rate)
            
        elif c_type == 1:
            nk_mag = n_mag_grid[k_idx, m]
            n_kminq_ph = n_phon_grid[p_idx, lam]
            rate = fgr_prefactor * V_sq * (n_kminq_ph - nk_mag)
            cuda.atomic.add(gamma_mag_path, path_idx * num_mag_branches + n, rate)
            
        elif c_type == 2:
            nk_mag = n_mag_grid[k_idx, n]
            n_kminq_mag = n_mag_grid[p_idx, m]
            rate = fgr_prefactor * V_sq * (n_kminq_mag - nk_mag)
            cuda.atomic.add(gamma_phon_path, path_idx * num_phon_branches + lam, rate)

    def save_path_dispersions(self, output_filename="Outputs/path_dispersions.csv"):
        """
        Saves the exact high-symmetry path dispersions (magnons and phonons) to a CSV file.
        """
        import os
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        
        print(f"\nWriting high-resolution path dispersions to {output_filename}...")
        with open(output_filename, 'w') as f:
            # Generate Header
            header = ["q_idx", "qx", "qy", "qz"]
            for b in range(self.phon_branches):
                header.append(f"w_phon_{b}_meV")
            for b in range(self.n_mag_branches):
                header.append(f"w_mag_{b}_meV")
            f.write(",".join(header) + "\n")

            # Write Rows
            for q_idx in range(self.N_path):
                qx, qy, qz = self.path_q_frac[q_idx]
                row = [f"{q_idx}", f"{qx:.6f}", f"{qy:.6f}", f"{qz:.6f}"]
                
                for b in range(self.phon_branches):
                    row.append(f"{self.path_w_phon[q_idx, b]:.6f}")
                
                for b in range(self.n_mag_branches):
                    row.append(f"{self.path_w_mag[q_idx, b]:.6f}")
                    
                f.write(",".join(row) + "\n")
        print("-> Done!")


    def plot_path_dispersions(self, filename="dispersion_verification.png"):
        """
        Plots the exact high-resolution path dispersions using the loaded HDF5 data.
        """
        import matplotlib.pyplot as plt
        
        if not hasattr(self, 'path_w_mag'):
            print("Error: Must call load_and_evaluate_path_hdf5() before plotting.")
            return

        # Calculate cumulative Cartesian distances for the x-axis
        k_distances = np.zeros(self.N_path)
        for i in range(1, self.N_path):
            dq_frac = self.path_q_frac[i] - self.path_q_frac[i-1]
            dq_cart = np.dot(dq_frac, self.reciprocal_lattice * 2.0 * np.pi)
            k_distances[i] = k_distances[i-1] + np.linalg.norm(dq_cart)

        fig, ax = plt.subplots(figsize=(10/2.52, 12/2.52))
        
        # Plot Phonons
        for b in range(self.phon_branches):
            label = 'Phonons' if b == 0 else ""
            ax.plot(k_distances, self.path_w_phon[:, b], color='#1f77b4', lw=2, label=label)
            
        # Plot Magnons
        for b in range(self.n_mag_branches):
            label = 'Magnons' if b == 0 else ""
            ax.plot(k_distances, self.path_w_mag[:, b], color='#d62728', lw=2, linestyle='--', label=label)

        # Formatting
        ax.set_ylabel('Energy (meV)', fontsize=14, fontweight='bold')
        ax.set_xlim(0, k_distances[-1])
        ax.set_ylim(bottom=0)
        ax.grid(True, axis='y', linestyle=':', color='gray', alpha=0.5)
        ax.legend(loc='upper right', fontsize=12, framealpha=1.0)
        
        # Add symmetry labels if they exist in the HDF5 file
        if hasattr(self, 'path_labels') and hasattr(self, 'path_segments'):
            tick_locs = [k_distances[0]]
            tick_labels = [self.path_labels[0][0]]
            
            idx = 0
            for i, seg_len in enumerate(self.path_segments):
                idx += seg_len
                tick_locs.append(k_distances[idx - 1])
                tick_labels.append(self.path_labels[i][1])
                
            ax.set_xticks(tick_locs)
            ax.set_xticklabels(tick_labels, fontsize=14)
            ax.grid(True, axis='x', linestyle='-', color='gray', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"-> Saved true path dispersion plot to '{filename}'")


    def plot_dispersions(self):
            """
            Extracts high-symmetry lines from the random SoA q-grid and plots 
            the Magnon and Phonon dispersions to verify energy scales.
            """
            import matplotlib.pyplot as plt
            
            # 1. Define high-symmetry path for Hexagonal lattice (CrI3)
            # Fractional coordinates [x, y, z]
            sym_points = {
                'Γ': [0.0, 0.0, 0.0],
                'M': [0.5, 0.0, 0.0],
                'K': [1/3, 1/3, 0.0],
                'A': [0.0, 0.0, 0.5],
                'L': [0.5, 0.0, 0.5],
                'H': [1/3, 1/3, 0.5]
            }
            
            # Standard path for 3D hexagonal systems
            path = ['Γ', 'M', 'K', 'Γ', 'A', 'L', 'H', 'A']
            
            # 2. Reconstruct path matching the grid
            k_path_indices = []
            k_distances = []
            tick_locs = []
            tick_labels = []
            
            current_dist = 0.0
            
            for i in range(len(path) - 1):
                p1 = np.array(sym_points[path[i]])
                p2 = np.array(sym_points[path[i+1]])
                
                # Number of steps based on grid resolution
                steps = np.max(self.mesh) 
                
                tick_locs.append(current_dist)
                tick_labels.append(path[i])
                
                for step in range(steps):
                    frac = p1 + (p2 - p1) * (step / steps)
                    # Map to grid integers
                    grid_pos = np.round(frac * self.mesh).astype(np.int32) % self.mesh
                    q_idx = self.grid_map[grid_pos[0], grid_pos[1], grid_pos[2]]
                    
                    if q_idx != -1 and (len(k_path_indices) == 0 or q_idx != k_path_indices[-1]):
                        k_path_indices.append(q_idx)
                        k_distances.append(current_dist)
                        
                        if step > 0:
                            dp = (p2 - p1) / steps
                            # Dist in Cartesian space
                            cart_dp = np.dot(dp, self.reciprocal_lattice * 2.0 * np.pi)
                            current_dist += np.linalg.norm(cart_dp)

            # Append final point
            tick_locs.append(current_dist)
            tick_labels.append(path[-1])
            
            # 3. Extract energies along the path
            w_mag_path = self.w_mag[k_path_indices]
            w_phon_path = self.w_phon[k_path_indices]
            
            # 4. Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot Phonons
            for b in range(self.phon_branches):
                label = 'Phonons' if b == 0 else ""
                ax.plot(k_distances, w_phon_path[:, b], color='#1f77b4', lw=2, label=label)
                
            # Plot Magnons
            for b in range(self.n_mag_branches):
                label = 'Magnons' if b == 0 else ""
                ax.plot(k_distances, w_mag_path[:, b], color='#d62728', lw=2, linestyle='--', label=label)

            # Formatting
            ax.set_ylabel('Energy (meV)', fontsize=14, fontweight='bold')
            ax.set_xlim(0, current_dist)
            ax.set_ylim(bottom=0)
            ax.set_xticks(tick_locs)
            ax.set_xticklabels(tick_labels, fontsize=14)
            ax.grid(True, axis='x', linestyle='-', color='gray', alpha=0.5)
            ax.grid(True, axis='y', linestyle=':', color='gray', alpha=0.5)
            ax.legend(loc='upper right', fontsize=12, framealpha=1.0)
            ax.set_title('CrI3 Hexagonal Dispersion Verification', fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('dispersion_verification.png', dpi=300)
            print("-> Saved dispersion plot to 'dispersion_verification.png'")



def diagonalize_bosonic_hamiltonian(H_matrix):
    """
    Implements Colpa's Cholesky-based algorithm for diagonalizing a Bosonic 
    Bogoliubov-de Gennes Hamiltonian.
    
    Translates the C++ implementation exactly 1-to-1.
    """
    dim = H_matrix.shape[0]
    if dim % 2 != 0:
        raise ValueError("Matrix dimension must be even (2m x 2m).")
    
    m = dim // 2

    # 2. Cholesky Decomposition: H = L * L^dagger (Lower triangular)
    # Colpa requires H = K^dagger * K, so K = L^dagger
    try:
        # np.linalg.cholesky returns the lower triangular factor L
        L_factor = np.linalg.cholesky(H_matrix)
    except np.linalg.LinAlgError:

        print( H_matrix )


        raise RuntimeError("Cholesky decomposition failed. Matrix is not positive definite.")
        
    K = L_factor.conj().T

    # 3. Construct the Commutation Metric Matrix J (Sigma_z)
    J = np.diag(np.concatenate([np.ones(m), -np.ones(m)]))

    # 4. Construct auxiliary Hermitian matrix W = K * J * K^dagger
    W = K @ J @ K.conj().T

    # 5. Unitary Diagonalization of W
    # eigh is for Hermitian matrices. Returns eigenvalues in ascending order.
    evals_unsorted, V_unsorted = np.linalg.eigh(W)

    # We sort so that bosonic commutation relations are preserved:
    # Move the second half (Positive, indices m to 2m) to the front
    # Move the first half (Negative, indices 0 to m) to the back
    evals_D = np.concatenate([evals_unsorted[m:], evals_unsorted[:m]])
    V = np.hstack([V_unsorted[:, m:], V_unsorted[:, :m]])

    # QR decomposition of V = Q*R
    Q, R = np.linalg.qr(V)

    # L = R * D * R^-1
    L_mat = R @ np.diag(evals_D) @ np.linalg.inv(R)

    # Final eigenvalues: sqrt(|diag(J * L)|)
    # The C++ code takes absolute value then sqrt
    final_evals_squared = np.abs(np.diag(J @ L_mat))
    final_evals = np.sqrt(final_evals_squared)

    # Para-unitary matrix Q_final (T) = K^-1 * Q * sqrt(D)
    Q_final = np.linalg.inv(K) @ Q @ np.diag(final_evals)

    # We only return the physical, positive modes (first m elements)


    return final_evals_squared[:m], Q_final



# ==========================================
# 1. GPU Kernels: Math Helpers
# ==========================================
@cuda.jit(device=True)
def calc_fourier_transform(kp_idx, q_idx, grid_cart, slc_axis, slc_rij, slc_rik, slc_J, slc_types, n_type, m_type, l_type, mu_type, J_tilde_out):
    """
    Computes the FT of the SLC tensor using Cartesian coordinate dot products.
    """
    for a in range(3):
        for b in range(3):
            J_tilde_out[a, b] = 0.0 + 0.0j

    # Pull directly from the precomputed Cartesian grid
    kp_vec_x = grid_cart[kp_idx, 0]
    kp_vec_y = grid_cart[kp_idx, 1]
    kp_vec_z = grid_cart[kp_idx, 2]
    
    q_vec_x = grid_cart[q_idx, 0]
    q_vec_y = grid_cart[q_idx, 1]
    q_vec_z = grid_cart[q_idx, 2]

    for i in range(slc_axis.shape[0]):
        if slc_axis[i] == mu_type:
            t_i = slc_types[i, 0]
            t_j = slc_types[i, 1]
            t_l = slc_types[i, 2]
            
            if t_i == n_type and t_j == m_type and t_l == l_type:
                phase_val = (kp_vec_x * slc_rij[i, 0] + kp_vec_y * slc_rij[i, 1] + kp_vec_z * slc_rij[i, 2]) + \
                            (q_vec_x * slc_rik[i, 0] + q_vec_y * slc_rik[i, 1] + q_vec_z * slc_rik[i, 2])
                            
                # The 2*pi is omitted here because it was multiplied into q_grid_cart
                phase_factor = cmath.exp(1j * phase_val)
                
                for a in range(3):
                    for b in range(3):
                        J_tilde_out[a, b] += slc_J[i, a, b] * phase_factor



@cuda.jit(device=True)
def calc_vertex_V(kp_idx, q_idx, lambda_phon, n, m, q_grid_cart, grid_map, slc_axis, slc_rij, slc_rik, slc_J, slc_types, eig_phon, w_phon, atom_masses, mag_moments):
    """
    Calculates the full scattering vertex V^{+-} combining the FT tensor and phonon eigenvectors.
    """

    # Find the Gamma point (0,0,0) index
    gamma_idx = grid_map[0, 0, 0]



    omega = w_phon[q_idx, lambda_phon]
    
    if omega < 1e-1:
        return 0.0

    # this is crucian. Now we check for the energy instead
    #if kp_idx == gamma_idx or q_idx == gamma_idx: #maybe this is not necessary as we skip it when calling the kernel
    #    return 0.0
    

    omega = w_phon[q_idx, lambda_phon]

    # --- Physical Constants & Spin Factors ---
    hbar = 0.6582119569 # meV * ps
    DALTON_TO_meV_PS2_PER_A2 = 0.10364269 # Equivalent to 1.0 / 9.6485
    
    S_n = math.fabs(mag_moments[n] / 2.0 ) 
    S_m = math.fabs(mag_moments[m] / 2.0 )
    
    # Sign of the moment (up/down for antiferromagnets, just 1 for FM)
    sigma_n = math.copysign(1.0, mag_moments[n]) if S_n > 0 else 0.0
    sigma_m = math.copysign(1.0, mag_moments[m]) if S_m > 0 else 0.0

    # 1. Allocate thread-local 3x3 complex tensors
    # J_tilde_dyn holds \tilde{J}(k-q, q)
    # J_tilde_stat holds \tilde{J}(0, q) for the delta term
    J_tilde_dyn = cuda.local.array((3, 3), dtype=np.complex128)
    J_tilde_stat = cuda.local.array((3, 3), dtype=np.complex128)
    
    V_complex = 0.0 + 0.0j
    num_atoms = atom_masses.shape[0]
    num_mag_branches = mag_moments.shape[0]

    # --- Sum over all atoms 'l' in the unit cell ---
    for l in range(num_atoms):
        
        # Calculate quantum displacement amplitude: \sqrt{\hbar / (2 M_l \omega)}
        mass_l = atom_masses[l] * DALTON_TO_meV_PS2_PER_A2
        disp_amp = math.sqrt(hbar*hbar / (2.0 * mass_l * omega))
        
        # Sum over Cartesian directions \mu \in {x=0, y=1, z=2}
        for mu in range(3):
            # Phonon eigenvector e^\mu_{l\lambda}(q)
            e_mu = eig_phon[q_idx, lambda_phon, l, mu]
            
            # --- Evaluate W^{+-,\mu}_{nml} ---
            
            # A) The Dynamic Term: \tilde{J}(k-q, q)
            # Pass (n+1, m+1, l+1) to respect the 1-based indexing in the CSVs.
            calc_fourier_transform(kp_idx, q_idx, q_grid_cart, slc_axis, slc_rij, slc_rik, slc_J, slc_types, n + 1, m + 1, l + 1, mu, J_tilde_dyn)

            # Extract components for W^{+-} dynamic part
            J_xx = J_tilde_dyn[0, 0]
            J_yy = J_tilde_dyn[1, 1]
            J_xy = J_tilde_dyn[0, 1]
            J_yx = J_tilde_dyn[1, 0]
            
            W_dynamic = (J_xx + 
                         (sigma_n * sigma_m) * J_yy - 
                         1j * sigma_m * J_xy + 
                         1j * sigma_n * J_yx) / math.sqrt(S_n * S_m)
            
            # B) The Static Term (Acoustic Sum Rule): \tilde{J}(0, q)
            W_static = 0.0 + 0.0j
            
            if n == m: 
                for mp in range(num_mag_branches):
                    if math.fabs(mag_moments[mp]) > 1e-2:
                        sigma_mp = math.copysign(1.0, mag_moments[mp])
                        calc_fourier_transform(gamma_idx, q_idx, q_grid_cart, slc_axis, slc_rij, slc_rik, slc_J, slc_types, n + 1, mp + 1, l + 1, mu, J_tilde_stat)
                        
                        W_static += (2.0 / S_n) * (sigma_n * sigma_mp) * J_tilde_stat[2, 2] # J^{zz}
            
            # C) Combine terms into the full W
            W_tot = W_dynamic - W_static
            
            # D) Add to total Vertex
            V_complex += disp_amp * e_mu * W_tot

    # Note: The 1/\sqrt{N} prefactor is handled elsewhere 
    
    # Return |V|^2
    return (V_complex.real**2 + V_complex.imag**2)


# ==========================================
# 2. GPU Kernels: The Main Phases
# ==========================================
@cuda.jit
def phase_1_scan(mesh, q_grid, q_grid_cart, grid_map, w_phon, w_mag, eig_phon, slc_axis, slc_rij, slc_rik, slc_J, slc_types, smearing, chan_indices, chan_weights, channel_count, atom_masses, mag_moments, gamma_idx):
    """
    Scans phase space for energy conservation using a Gaussian delta function.
    """
    q_idx, k_idx = cuda.grid(2)
    N = q_grid.shape[0]

    # Guard against out-of-bounds
    if q_idx >= N or k_idx >= N: 
        return

    # We allow this for now:
    #if q_idx == k_idx or q_idx == gamma_idx or k_idx == gamma_idx:
    #    return
        
    n_mag = w_mag.shape[1]
    n_phon = w_phon.shape[1]
    
    gaussian_norm = 1.0 / (smearing * 2.50662827463)
    cutoff = 3.0 * smearing
    
    # --- Kinematic Mappings ---
    qx_qmink = (q_grid[q_idx, 0] - q_grid[k_idx, 0] + mesh[0]) % mesh[0]
    qy_qmink = (q_grid[q_idx, 1] - q_grid[k_idx, 1] + mesh[1]) % mesh[1]
    qz_qmink = (q_grid[q_idx, 2] - q_grid[k_idx, 2] + mesh[2]) % mesh[2]
    idx_qmink = grid_map[qx_qmink, qy_qmink, qz_qmink]
    
    qx_kminq = (q_grid[k_idx, 0] - q_grid[q_idx, 0] + mesh[0]) % mesh[0]
    qy_kminq = (q_grid[k_idx, 1] - q_grid[q_idx, 1] + mesh[1]) % mesh[1]
    qz_kminq = (q_grid[k_idx, 2] - q_grid[q_idx, 2] + mesh[2]) % mesh[2]
    idx_kminq = grid_map[qx_kminq, qy_kminq, qz_kminq]

    for n in range(n_mag):
        for m in range(n_mag):
            for lam in range(n_phon):
                
                # ---------------------------------------------------------
                # Process 0: Magnon Emission
                # ---------------------------------------------------------
                dE_mag_emit = w_mag[q_idx, n] - w_mag[k_idx, m] - w_phon[idx_qmink, lam]
                if abs(dE_mag_emit) < cutoff:
                    delta_weight = gaussian_norm * math.exp(-0.5 * (dE_mag_emit * dE_mag_emit) / (smearing * smearing))
                    V_sq = calc_vertex_V(q_idx, idx_qmink, lam, n, m, q_grid_cart, grid_map, slc_axis, slc_rij, slc_rik, slc_J, slc_types, eig_phon, w_phon, atom_masses, mag_moments)

                    c_idx = cuda.atomic.add(channel_count, 0, 1)
                    if c_idx < chan_indices.shape[1]:
                        # Write contiguously down the columns
                        chan_indices[0, c_idx] = 0          # c_type
                        chan_indices[1, c_idx] = q_idx
                        chan_indices[2, c_idx] = k_idx
                        chan_indices[3, c_idx] = idx_qmink
                        chan_indices[4, c_idx] = n
                        chan_indices[5, c_idx] = m
                        chan_indices[6, c_idx] = lam
                        chan_weights[c_idx] = V_sq * delta_weight
                        
                # ---------------------------------------------------------
                # Process 1: Magnon Absorption
                # ---------------------------------------------------------
                dE_mag_abs = w_mag[q_idx, n] - w_mag[k_idx, m] + w_phon[idx_kminq, lam]
                if abs(dE_mag_abs) < cutoff:
                    delta_weight = gaussian_norm * math.exp(-0.5 * (dE_mag_abs * dE_mag_abs) / (smearing * smearing))
                    V_sq = calc_vertex_V(q_idx, idx_kminq, lam, m, n, q_grid_cart, grid_map, slc_axis, slc_rij, slc_rik, slc_J, slc_types, eig_phon, w_phon, atom_masses, mag_moments)

                    c_idx = cuda.atomic.add(channel_count, 0, 1)
                    if c_idx < chan_indices.shape[1]:
                        # Write contiguously down the columns
                        chan_indices[0, c_idx] = 1          # c_type
                        chan_indices[1, c_idx] = q_idx
                        chan_indices[2, c_idx] = k_idx
                        chan_indices[3, c_idx] = idx_kminq
                        chan_indices[4, c_idx] = n
                        chan_indices[5, c_idx] = m
                        chan_indices[6, c_idx] = lam
                        chan_weights[c_idx] = V_sq * delta_weight
                        
                # ---------------------------------------------------------
                # Process 2: Phonon Emission
                # ---------------------------------------------------------
                dE_phon_emit = w_phon[q_idx, lam] + w_mag[idx_kminq, m] - w_mag[k_idx, n]
                if abs(dE_phon_emit) < cutoff:
                    delta_weight = gaussian_norm * math.exp(-0.5 * (dE_phon_emit * dE_phon_emit) / (smearing * smearing))

                    # Find the index of -q for the vertex calculation
                    qx_minq = (-q_grid[q_idx, 0] + mesh[0]) % mesh[0]
                    qy_minq = (-q_grid[q_idx, 1] + mesh[1]) % mesh[1]
                    qz_minq = (-q_grid[q_idx, 2] + mesh[2]) % mesh[2]
                    idx_minus_q = grid_map[qx_minq, qy_minq, qz_minq]

                    V_sq = calc_vertex_V(k_idx, idx_minus_q, lam, n, m, q_grid_cart, grid_map, slc_axis, slc_rij, slc_rik, slc_J, slc_types, eig_phon, w_phon, atom_masses, mag_moments)

                    c_idx = cuda.atomic.add(channel_count, 0, 1)
                    if c_idx < chan_indices.shape[1]:
                        # Write contiguously down the columns
                        chan_indices[0, c_idx] = 2          # c_type
                        chan_indices[1, c_idx] = q_idx
                        chan_indices[2, c_idx] = k_idx
                        chan_indices[3, c_idx] = idx_kminq
                        chan_indices[4, c_idx] = n
                        chan_indices[5, c_idx] = m
                        chan_indices[6, c_idx] = lam
                        chan_weights[c_idx] = V_sq * delta_weight

@cuda.jit
def phase_2_time_step(chan_indices, chan_weights, num_channels, n_mag, n_phon, dn_mag, dn_phon, N_points, smearing):
    idx = cuda.grid(1)
    if idx >= num_channels[0] or idx >= chan_weights.shape[0]: 
        return
        
    # Perfectly coalesced memory reads
    c_type = chan_indices[0, idx]
    q_idx  = chan_indices[1, idx] 
    k_idx  = chan_indices[2, idx] 
    p_idx  = chan_indices[3, idx] 
    n      = chan_indices[4, idx]
    m      = chan_indices[5, idx]
    lam    = chan_indices[6, idx]
    V_sq   = chan_weights[idx]
    
    num_mag_branches = n_mag.shape[1]
    num_phon_branches = n_phon.shape[1]
    
    hbar = 0.6582119569 # meV * ps
    fgr_prefactor = (2.0 * math.pi / hbar) / N_points
    
    if c_type == 0: 
        # ---------------------------------------------------------
        # Magnon 1: q_idx = q, k_idx = k, p_idx = q-k
        # ---------------------------------------------------------
        nk_mag = n_mag[k_idx, m]
        nq_mag = n_mag[q_idx, n]
        n_qmink_ph = n_phon[p_idx, lam]
        
        rate = fgr_prefactor * V_sq * ((nq_mag + 1.0) * nk_mag * n_qmink_ph - nq_mag * (nk_mag + 1.0) * (n_qmink_ph + 1.0))

        idx_update = q_idx * num_mag_branches + n
        cuda.atomic.add(dn_mag, idx_update, rate)
        
    elif c_type == 1:
        # ---------------------------------------------------------
        # Magnon 2: q_idx = q, k_idx = k, p_idx = k-q
        # ---------------------------------------------------------
        nk_mag = n_mag[k_idx, m]
        nq_mag = n_mag[q_idx, n]
        n_kminq_ph = n_phon[p_idx, lam]
        
        rate = fgr_prefactor * V_sq * ((nq_mag + 1.0) * nk_mag * (n_kminq_ph + 1.0) - nq_mag * (nk_mag + 1.0) * n_kminq_ph)

        idx_update = q_idx * num_mag_branches + n
        cuda.atomic.add(dn_mag, idx_update, rate)
        
    elif c_type == 2:
        # ---------------------------------------------------------
        # Phonon: q_idx = q (phonon), k_idx = k, p_idx = k-q
        # ---------------------------------------------------------
        nk_mag = n_mag[k_idx, n]
        n_kminq_mag = n_mag[p_idx, m]
        nq_ph = n_phon[q_idx, lam]
        
        rate = fgr_prefactor * V_sq * ((nq_ph + 1.0) * (n_kminq_mag + 1.0) * nk_mag - nq_ph * n_kminq_mag * (nk_mag + 1.0))

        idx_update = q_idx * num_phon_branches + lam
        cuda.atomic.add(dn_phon, idx_update, rate)


@cuda.jit
def phase_lifetime(chan_indices, chan_weights, num_channels, n_mag, n_phon, gamma_mag, gamma_phon, N_points):
    idx = cuda.grid(1)
    if idx >= num_channels[0] or idx >= chan_weights.shape[0]: 
        return
        
    c_type = chan_indices[0, idx]
    q_idx  = chan_indices[1, idx] 
    k_idx  = chan_indices[2, idx] 
    p_idx  = chan_indices[3, idx] 
    n      = chan_indices[4, idx]
    m      = chan_indices[5, idx]
    lam    = chan_indices[6, idx]
    V_sq   = chan_weights[idx]
    
    num_mag_branches = n_mag.shape[1]
    num_phon_branches = n_phon.shape[1]
    
    hbar = 0.6582119569 # meV * ps
    fgr_prefactor = (2.0 * math.pi / hbar) / N_points
    
    if c_type == 0: 
        # ---------------------------------------------------------
        # Magnon Emission: q_idx = q, k_idx = k, p_idx = q-k
        # Term: [1 + n_ph(q-k) + n_mag(k)]
        # ---------------------------------------------------------
        nk_mag = n_mag[k_idx, m]
        n_qmink_ph = n_phon[p_idx, lam]
        
        scattering_rate = fgr_prefactor * V_sq * (1.0 + n_qmink_ph + nk_mag)

        idx_update = q_idx * num_mag_branches + n
        cuda.atomic.add(gamma_mag, idx_update, scattering_rate)
        
    elif c_type == 1:
        # ---------------------------------------------------------
        # Magnon Absorption: q_idx = q, k_idx = k, p_idx = k-q
        # Term: [n_ph(k-q) - n_mag(k)]
        # ---------------------------------------------------------
        nk_mag = n_mag[k_idx, m]
        n_kminq_ph = n_phon[p_idx, lam]
        
        scattering_rate = fgr_prefactor * V_sq * (n_kminq_ph - nk_mag)

        idx_update = q_idx * num_mag_branches + n
        cuda.atomic.add(gamma_mag, idx_update, scattering_rate)
        
    elif c_type == 2:
        # ---------------------------------------------------------
        # Phonon Scattering: q_idx = q (phonon), k_idx = k, p_idx = k-q
        # ---------------------------------------------------------
        nk_mag = n_mag[k_idx, n]
        n_kminq_mag = n_mag[p_idx, m]
        
        scattering_rate = fgr_prefactor * V_sq * (n_kminq_mag - nk_mag)

        idx_update = q_idx * num_phon_branches + lam
        cuda.atomic.add(gamma_phon, idx_update, scattering_rate)



def compute_and_write_observables(step, current_time, n_mag, n_phon, w_mag, w_phon, file_handle):
    """
    Calculates macroscopic observables (Energy, Particles, T_eff) 
    and appends them to the output file.
    """
    # Boltzmann constant in meV/K (aligning with the 4.135 meV/THz scaling)
    kB = 0.08617333262 
    
    # 1. Particle Numbers
    N_mag = np.sum(n_mag)
    N_phon = np.sum(n_phon)
    
    # 2. Subsystem Energies
    E_mag = np.sum(n_mag * w_mag)
    E_phon = np.sum(n_phon * w_phon)
    E_tot = E_mag + E_phon
    
    # 3. Effective Temperatures
    def calc_subsystem_temp(n_dist, w_dist):
        # Mask out zero-energy modes (e.g., acoustic modes at Gamma) 
        # and unpopulated modes to prevent NaNs and infinities
        valid_modes = (n_dist > 1e-10) & (w_dist > 1e-5)
        if not np.any(valid_modes):
            return 0.0
            
        n_valid = n_dist[valid_modes]
        w_valid = w_dist[valid_modes]
        
        # Invert Bose-Einstein distribution for mode temperature
        mode_temps = w_valid / (kB * np.log(1.0 + 1.0 / n_valid))
        
        # Energy-weighted average temperature
        weights = n_valid * w_valid
        return np.average(mode_temps, weights=weights)

    T_eff_mag = calc_subsystem_temp(n_mag, w_mag)
    T_eff_phon = calc_subsystem_temp(n_phon, w_phon)
    
    # 4. Write to file
    line = f"{step}\t{current_time:.6e}\t{E_tot:.6e}\t{E_mag:.6e}\t{E_phon:.6e}\t{N_mag:.6e}\t{N_phon:.6e}\t{T_eff_mag:.2f}\t{T_eff_phon:.2f}\n"
    file_handle.write(line)
    file_handle.flush() # Flush buffer to ensure data is saved during long runs


@cuda.jit
def apply_euler_and_reset(n_mag, n_phon, dn_mag, dn_phon, dt):
    """
    Applies the explicit Euler step, clips negative populations, 
    and zeros out the derivative buffers entirely on the GPU.
    """
    idx = cuda.grid(1)
    
    # 1. Update Magnons
    if idx < dn_mag.shape[0]:
        q_idx = idx // n_mag.shape[1]
        b = idx % n_mag.shape[1]
        
        new_n = n_mag[q_idx, b] + dn_mag[idx] * dt

        n_mag[q_idx, b] = new_n if new_n > 1e-15 else 1e-15
        dn_mag[idx] = 0.0  # Reset for next step
        
    # 2. Update Phonons
    if idx < dn_phon.shape[0]:
        q_idx = idx // n_phon.shape[1]
        b = idx % n_phon.shape[1]
        
        new_n = n_phon[q_idx, b] + dn_phon[idx] * dt
        n_phon[q_idx, b] = new_n if new_n > 1e-15 else 1e-15
        dn_phon[idx] = 0.0 # Reset for next step


def init_bose_einstein(w_distribution, temperature_K):
    """
    Generates a Bose-Einstein occupation array for given energies (meV)
    and a target temperature (Kelvin).
    """
    if temperature_K <= 0.0:
        return np.zeros_like(w_distribution, dtype=np.float64)
        
    kB = 0.08617333262  # meV/K
    
    # Suppress runtime warnings for dividing by zero at the Gamma point
    with np.errstate(divide='ignore', invalid='ignore'):
        occ = 1.0 / (np.exp(w_distribution / (kB * temperature_K)) - 1.0)
        
    # Cleanly catch numerical singularities (inf/nan from zero-energy modes) 
    # and safely regularize them to zero occupation
    occ[~np.isfinite(occ)] = 0.0
    return occ


if __name__ == "__main__":
    slc_files_CrSb = ['Inputs/CrSb/transformed_SLC_tensor_x_scaled.csv', 'Inputs/CrSb/transformed_SLC_tensor_y_scaled.csv', 'Inputs/CrSb/transformed_SLC_tensor_z_scaled.csv']
    slc_files_bccFe = ['Inputs/bccFe/Fe_full_tensor_ij-uk_x_displacement.csv', 'Inputs/bccFe/Fe_full_tensor_ij-uk_y_displacement.csv', 'Inputs/bccFe/Fe_full_tensor_ij-uk_z_displacement.csv']
    slc_files_CrI3 = ['Inputs/CrI3/transformed_SLC_tensor_x_filtered.csv', 'Inputs/CrI3/transformed_SLC_tensor_y_filtered.csv', 'Inputs/CrI3/transformed_SLC_tensor_z_filtered.csv']
    mesh_bccFe = "Inputs/bccFe/combined_band_20x20x20.h5"
    mesh_CrI3 = "Inputs/CrI3/combined_band_12x12x12.h5"
    Jijs_bccFe = "Inputs/bccFe/Fe_Jij_scaled.csv"
    Jijs_CrI3 = "Inputs/CrI3/JijCrI3.dat"

    lattice_constant_bccFe = 2.8665  # in Angstroms
    lattice_constant_CrI3 = 7.006660421592247  # in Angstroms

    anisotropy_bccFe = 0.01 
    anisotropy_CrI3 = 0.49

    anisotropy = anisotropy_CrI3     

    lattice_constant = lattice_constant_CrI3    
    mesh = mesh_CrI3
    Jijs = Jijs_CrI3
    slc_files = slc_files_CrI3

    smearing = 0.005
    
    crystal_data = CrystalDataSoA(
        mesh, 
        Jijs,
        slc_files=slc_files,
        lattice_constant=lattice_constant,
        anisotropy=anisotropy,  # Set to zero for CrI3 to match the DFT inputs
    )
    
    crystal_data.print_summary()

    crystal_data.plot_dispersions()

    gpu_data = crystal_data.push_to_gpu()

    gamma_idx = int(crystal_data.grid_map[0, 0, 0])


    # Load the high-symmetry path from the HDF5 band file
    crystal_data.load_and_evaluate_path_hdf5("Inputs/CrI3/band.h5", K_anisotropy=anisotropy, lattice_constant=lattice_constant)
    # Save the exact path energies to a CSV
    crystal_data.save_path_dispersions("Outputs/path_dispersions.csv")
    # Plot the exact path
    crystal_data.plot_path_dispersions("Outputs/exact_path_dispersions.png")
    # Push path data to GPU for scanning kernels
    d_path_q_frac = cuda.to_device(crystal_data.path_q_frac)
    d_path_q_cart = cuda.to_device(crystal_data.path_q_cart)
    d_path_w_phon = cuda.to_device(crystal_data.path_w_phon)
    d_path_w_mag = cuda.to_device(crystal_data.path_w_mag)
    d_path_eig_phon = cuda.to_device(crystal_data.path_eig_phon)


    # 2. Setup Phase 1 memory
    N_points = crystal_data.N
    
    anticipated_fraction = 0.01
    total_loops = N_points**2 * crystal_data.n_mag_branches**2 * crystal_data.phon_branches * 3
    max_channels = int(total_loops * anticipated_fraction)
    
    # --- SoA ALLOCATION ---
    # Shape is (7, max_channels) so the last axis is contiguous.
    # Row 0: c_type | Row 1: q_idx | Row 2: k_idx | Row 3: p_idx
    # Row 4: n      | Row 5: m     | Row 6: lam
    d_chan_indices = cuda.device_array((7, max_channels), dtype=np.int32)
    d_chan_weights = cuda.device_array(max_channels, dtype=np.float64)
    d_channel_count = cuda.to_device(np.zeros(1, dtype=np.int32))
    
    threads_per_block = 256
    blocks_per_grid = math.ceil(N_points / threads_per_block)


    # 3. Execute Phase 1
    print("\nStarting Phase 1: Scanning Phase Space and computing FT Vertices...")

    threads_per_block_2d = (16, 16) # 256 threads total per block
    blocks_x = math.ceil(N_points / threads_per_block_2d[0])
    blocks_y = math.ceil(N_points / threads_per_block_2d[1])
    blocks_per_grid_2d = (blocks_x, blocks_y)


    phase_1_scan[blocks_per_grid_2d, threads_per_block_2d](
        gpu_data["mesh"], 
        gpu_data["q_grid"], 
        gpu_data["q_grid_cart"],
        gpu_data["grid_map"], 
        gpu_data["w_phon"], 
        gpu_data["w_mag"], 
        gpu_data["eig_phon"],
        gpu_data["slc_axis"], 
        gpu_data["slc_rij"], 
        gpu_data["slc_rik"], 
        gpu_data["slc_J"], 
        gpu_data["slc_types"], 
        smearing, 
        d_chan_indices,   
        d_chan_weights, 
        d_channel_count,
        gpu_data["atom_masses"], 
        gpu_data["mag_moments"],
        gamma_idx
    )

    # Wait for the GPU to finish the scan
    cuda.synchronize()
    
    # Extract the final count BEFORE slicing
    num_channels = d_channel_count.copy_to_host()[0]
    print(f"Allowed Channels found: {num_channels:,}")
    print(f" -> Percentage of phase space allowed: {num_channels / total_loops:.2%}")

    # Slice the device arrays so Phase 2 ONLY iterates over valid channels
    # Slicing along the 2nd axis preserves the C-contiguous layout
    d_chan_indices_active = d_chan_indices[:, :num_channels]
    d_chan_weights_active = d_chan_weights[:num_channels]
    
    blocks_eval = math.ceil(num_channels / threads_per_block)


    # 4. Setup Phase 2 memory
    T_mag_init = 25
    T_phon_init = 20
    
    print(f"\nInitializing populations at thermal equilibrium:")
    print(f" -> Magnons: {T_mag_init} K")
    print(f" -> Phonons: {T_phon_init} K")

    # Generate population profiles matching the actual branch dispersions
    n_mag_cpu = init_bose_einstein(crystal_data.w_mag, T_mag_init)
    n_phon_cpu = init_bose_einstein(crystal_data.w_phon, T_phon_init)
    
    # Set Gamma point occupations to zero to avoid singularities
    n_mag_cpu[gamma_idx, :] = 0.0
    n_phon_cpu[gamma_idx, :] = 0.0

    # Push initial states
    d_n_mag = cuda.to_device(n_mag_cpu)
    d_n_phon = cuda.to_device(n_phon_cpu)
    
    # STRIP FIX: Initialize derivatives to strict zeros ONCE before the loop
    d_dn_mag = cuda.to_device(np.zeros(N_points * crystal_data.n_mag_branches, dtype=np.float64))
    d_dn_phon = cuda.to_device(np.zeros(N_points * crystal_data.phon_branches, dtype=np.float64))

    # ========================== Path Lifetime Evaluation ==========================
    print("\nStarting Path Lifetime Evaluation...")
    
    # 1. Setup fractional grid mapping
    grid_q_frac_cpu = crystal_data.q_grid.astype(np.float64) / crystal_data.mesh
    d_grid_q_frac = cuda.to_device(grid_q_frac_cpu)

    # 2. Setup Phase 1 memory for path scanning
    N_path = crystal_data.N_path
    total_path_loops = N_path * N_points * crystal_data.n_mag_branches**2 * crystal_data.phon_branches * 3
    max_path_channels = int(total_path_loops * anticipated_fraction)
    
    d_path_chan_indices = cuda.device_array((7, max_path_channels), dtype=np.int32)
    d_path_chan_weights = cuda.device_array(max_path_channels, dtype=np.float64)
    d_path_channel_count = cuda.to_device(np.zeros(1, dtype=np.int32))

    threads_2d_path = (16, 16)
    blocks_x_path = math.ceil(N_path / threads_2d_path[0])
    blocks_y_path = math.ceil(N_points / threads_2d_path[1])

    # 3. Execute Path Scan
    print(" -> Scanning Phase Space (Path x Grid)...")
    phase_1_scan_path[(blocks_x_path, blocks_y_path), threads_2d_path](
        gpu_data["mesh"], d_grid_q_frac, gpu_data["q_grid_cart"], gpu_data["grid_map"],
        d_path_q_frac, d_path_q_cart, d_path_w_phon, d_path_w_mag, d_path_eig_phon,
        gpu_data["w_phon"], gpu_data["w_mag"], gpu_data["eig_phon"],
        gpu_data["slc_axis"], gpu_data["slc_rij"], gpu_data["slc_rik"], gpu_data["slc_J"], gpu_data["slc_types"], 
        smearing, d_path_chan_indices, d_path_chan_weights, d_path_channel_count,
        gpu_data["atom_masses"], gpu_data["mag_moments"], gamma_idx
    )
    cuda.synchronize()
    
    path_num_channels = d_path_channel_count.copy_to_host()[0]
    print(f" -> Path Channels found: {path_num_channels:,}")

    # Slice active channels
    d_path_chan_indices_active = d_path_chan_indices[:, :path_num_channels]
    d_path_chan_weights_active = d_path_chan_weights[:path_num_channels]

    # 4. Allocate Path Scattering Rate Arrays
    d_gamma_mag_path = cuda.to_device(np.zeros(N_path * crystal_data.n_mag_branches, dtype=np.float64))
    d_gamma_phon_path = cuda.to_device(np.zeros(N_path * crystal_data.phon_branches, dtype=np.float64))

    # 5. Execute Path Lifetimes (using evaluated d_n_mag and d_n_phon from the grid)
    print(" -> Calculating explicit path lifetimes...")
    blocks_eval_path = math.ceil(path_num_channels / threads_per_block)
    
    phase_lifetime_path[blocks_eval_path, threads_per_block](
        d_path_chan_indices_active,
        d_path_chan_weights_active,
        d_path_channel_count, 
        d_n_mag, 
        d_n_phon, 
        d_gamma_mag_path, 
        d_gamma_phon_path, 
        N_points  # Density of states normalization relies on BZ grid resolution
    )
    cuda.synchronize()

    # 6. Extract and Save
    gamma_mag_path_cpu = d_gamma_mag_path.copy_to_host().reshape((N_path, crystal_data.n_mag_branches))
    gamma_phon_path_cpu = d_gamma_phon_path.copy_to_host().reshape((N_path, crystal_data.phon_branches))

    with open("Outputs/path_lifetimes.csv", "w") as f:
        f.write("q_idx,qx,qy,qz,particle,branch,energy_meV,gamma_ps-1,tau_ps\n")
        
        # Magnons
        for i in range(N_path):
            qx, qy, qz = crystal_data.path_q_frac[i]
            for b in range(crystal_data.n_mag_branches):
                energy = crystal_data.path_w_mag[i, b]
                gamma = gamma_mag_path_cpu[i, b]
                tau = 1.0 / gamma if gamma > 1e-12 else float('inf')
                f.write(f"{i},{qx:.6f},{qy:.6f},{qz:.6f},magnon,{b},{energy:.6f},{gamma:.6e},{tau:.6e}\n")
                
        # Phonons
        for i in range(N_path):
            qx, qy, qz = crystal_data.path_q_frac[i]
            for b in range(crystal_data.phon_branches):
                energy = crystal_data.path_w_phon[i, b]
                gamma = gamma_phon_path_cpu[i, b]
                tau = 1.0 / gamma if gamma > 1e-12 else float('inf')
                f.write(f"{i},{qx:.6f},{qy:.6f},{qz:.6f},phonon,{b},{energy:.6f},{gamma:.6e},{tau:.6e}\n")
                
    print("-> Saved true path lifetimes to Outputs/path_lifetimes.csv")


    # ====== Lifetime and scattering rate phase ======

    # Allocate dedicated arrays for the scattering rates
    # Shape flattened to match the atomic add logic: (N_points * branches)
    d_gamma_mag = cuda.to_device(np.zeros(N_points * crystal_data.n_mag_branches, dtype=np.float64))
    d_gamma_phon = cuda.to_device(np.zeros(N_points * crystal_data.phon_branches, dtype=np.float64))

    # Evaluate grid geometry
    blocks_eval = math.ceil(num_channels / threads_per_block)

    # Launch Kernel
    print("\nCalculating equilibrium scattering rates and lifetimes...")
    phase_lifetime[blocks_eval, threads_per_block](
        d_chan_indices_active,
        d_chan_weights_active,
        d_channel_count, 
        d_n_mag,     # Evaluated at thermal equilibrium T_0
        d_n_phon,    # Evaluated at thermal equilibrium T_0
        d_gamma_mag, 
        d_gamma_phon, 
        N_points
    )
    cuda.synchronize()

    # 1. Pull back and reshape
    gamma_mag_cpu = d_gamma_mag.copy_to_host().reshape((N_points, crystal_data.n_mag_branches))
    gamma_phon_cpu = d_gamma_phon.copy_to_host().reshape((N_points, crystal_data.phon_branches))

    # 2. Write to CSV
    os.makedirs("Outputs", exist_ok=True)
    with open("Outputs/equilibrium_lifetimes.csv", "w") as f:
        f.write("q_idx,qx,qy,qz,particle,branch,energy_meV,gamma_ps-1,tau_ps\n")
        
        # Write Magnon Lifetimes
        for q_idx in range(N_points):
            qx, qy, qz = crystal_data.q_grid[q_idx]
            for branch in range(crystal_data.n_mag_branches):
                energy = crystal_data.w_mag[q_idx, branch]
                gamma = gamma_mag_cpu[q_idx, branch]
                
                # Protect against divide-by-zero for non-scattering modes (or Gamma point)
                tau = 1.0 / gamma if gamma > 1e-12 else float('inf')
                
                f.write(f"{q_idx},{qx},{qy},{qz},magnon,{branch},{energy:.6f},{gamma:.6e},{tau:.6e}\n")
                
        # Write Phonon Lifetimes
        for q_idx in range(N_points):
            qx, qy, qz = crystal_data.q_grid[q_idx]
            for branch in range(crystal_data.phon_branches):
                energy = crystal_data.w_phon[q_idx, branch]
                gamma = gamma_phon_cpu[q_idx, branch]
                
                tau = 1.0 / gamma if gamma > 1e-12 else float('inf')
                
                f.write(f"{q_idx},{qx},{qy},{qz},phonon,{branch},{energy:.6f},{gamma:.6e},{tau:.6e}\n")

    print("-> Saved lifetimes to Outputs/equilibrium_lifetimes.csv")


    # ========================== Time-evolution Phase ==========================
    steps = 10000000
    dt = 5E-5  # ps
    
    # Grid sizes for both kernels
    blocks_eval = math.ceil(num_channels / threads_per_block)
    max_elements = max(N_points * crystal_data.n_mag_branches, N_points * crystal_data.phon_branches)
    blocks_euler = math.ceil(max_elements / threads_per_block)
    
    obs_file = open("Outputs/observables_dynamics.txt", "w")
    obs_file.write("Step\tTime_ps\tE_tot_meV\tE_mag_meV\tE_phon_meV\tN_mag\tN_phon\tT_eff_mag_K\tT_eff_phon_K\n")
    
    print(f"\nStarting Phase 2: Time Integration ({steps} steps)...")

    for step in range(steps):
        
        # CPU Interaction: Only pull data across the PCIe bus every 100 steps
        if step % 1000 == 0:
            compute_and_write_observables(
                step=step,
                current_time=step * dt,
                n_mag=d_n_mag.copy_to_host(),       # Natively returns the 2D array
                n_phon=d_n_phon.copy_to_host(),
                w_mag=crystal_data.w_mag,   
                w_phon=crystal_data.w_phon,
                file_handle=obs_file
            )

        phase_2_time_step[blocks_eval, threads_per_block](
            d_chan_indices_active,
            d_chan_weights_active,
            d_channel_count, 
            d_n_mag, 
            d_n_phon, 
            d_dn_mag, 
            d_dn_phon, 
            N_points, 
            smearing
        )
        
        # 2. Apply Euler integration, clip negatives, and reset derivatives to 0.0
        apply_euler_and_reset[blocks_euler, threads_per_block](
            d_n_mag, d_n_phon, d_dn_mag, d_dn_phon, dt
        )

    obs_file.close()
    print("Simulation Complete.")