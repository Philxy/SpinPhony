import numpy as np
import os
from numba import cuda
import cmath
import math
import h5py
import argparse
import sys

try:
    import tomllib
except ImportError:
    import tomli as tomllib


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

            # Print the missing q-points for debugging
            missing_q_points = np.argwhere(self.grid_map == -1)
            print(f"Missing q-points (grid positions): {missing_q_points.shape[0]}")
            for pos in missing_q_points:
                print(f"  -> {pos[0]}, {pos[1]}, {pos[2]}")

            raise ValueError("Grid map initialization failed: Incomplete q-point mesh.")

        # Allocate and Compute Magnons
        self.w_mag = np.zeros((self.N, self.n_mag_branches), dtype=np.float64)
        self._compute_magnon_dispersions(K_anisotropy=anisotropy, lattice_constant=lattice_constant)

        #self._compute_group_velocities() # temporarily disabled for performance; can be re-enabled if needed
        self.grad_f_phon = np.zeros((self.N, self.phon_branches, 3), dtype=np.float64)


        # Cartesian conversion
        q_frac_array = self.q_grid / self.mesh
        # Fold fractional coordinates strictly to [-0.5, 0.5) before Cartesian mapping
        q_frac_centered = q_frac_array - np.round(q_frac_array)
        self.q_grid_cart = np.dot(q_frac_centered, self.reciprocal_lattice * 2.0 * np.pi)
        
        # 4. Parse SLC Tensors
        if slc_files and len(slc_files) == 3:
            self._parse_slc_tensors(slc_files[0], slc_files[1], slc_files[2], lattice_constant)

        self.w_hyb, self.Qmatrix, self.H_BdG_pre_diagonalized, self.H_BdG_diagonalized = self._calculate_coupled_hamiltonian(
            q_cart_array=self.q_grid_cart,
            dyn_mat=self.dyn_mat_phon,
            K_anisotropy=anisotropy,
            lattice_constant=lattice_constant,
            return_full_matrices=True
        )

        self._compute_hybrid_group_velocities()
        #self.grad_f_hyb = np.zeros((self.N, self.phon_branches + self.n_mag_branches, 3), dtype=np.float64)


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

            # Print the missing q-points for debugging
            missing_q_points = np.argwhere(self.grid_map == -1)
            print(f"Missing q-points (grid positions): {missing_q_points.shape[0]}")
            for pos in missing_q_points:
                print(f"  -> {pos[0]}, {pos[1]}, {pos[2]}")

            raise ValueError("Grid map initialization failed: Incomplete q-point mesh.")
            
    def push_to_gpu(self):
        
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

        gpu_buffers["grad_f_phon"] = track_and_push("grad_f_phon", self.grad_f_phon)
        gpu_buffers["grad_f_mag"] = track_and_push("grad_f_mag", self.grad_f_mag)


        gpu_buffers["jij"] = track_and_push("jij", self.jij_interactions)
        gpu_buffers["atom_masses"] = track_and_push("atom_masses", self.atom_masses)
        gpu_buffers["mag_moments"] = track_and_push("mag_moments", self.mag_moments)

        # NEW:
        #gpu_buffers["mag_moments"] = track_and_push("mag_moments", self.mag_moments[self.mag_indices])   # length n_mag_branches

        gpu_buffers["w_hyb"] = track_and_push("w_hyb", self.w_hyb)
        gpu_buffers["Qmatrix"] = track_and_push("Qmatrix", self.Qmatrix)
        gpu_buffers["grad_f_hyb"] = track_and_push("grad_f_hyb", self.grad_f_hyb)
        
        if hasattr(self, 'slc_axis'):
            gpu_buffers["slc_axis"] = track_and_push("slc_axis", self.slc_axis)
            gpu_buffers["slc_rij"] = track_and_push("slc_rij", self.slc_rij)
            gpu_buffers["slc_rik"] = track_and_push("slc_rik", self.slc_rik)
            gpu_buffers["slc_J"] = track_and_push("slc_J", self.slc_J)
            gpu_buffers["slc_types"] = track_and_push("slc_types", self.slc_types)

        print(f"-------------------------------------------")
        print(f"TOTAL VRAM ALLOCATED: {total_vram_used/1e6:.2f} MB")
        
        return gpu_buffers
    

    def _compute_magnon_dispersions(self, K_anisotropy=0.5, lattice_constant=1.0):
        self.eig_mag = np.zeros((self.N, 2*self.n_mag_branches, 2*self.n_mag_branches), dtype=np.complex128)
        
        self.grad_f_mag = np.zeros((self.N, self.n_mag_branches, 3), dtype=np.float64)
        atom_to_mag = np.full(self.l_atoms, -1, dtype=np.int32)
        for i, m_idx in enumerate(self.mag_indices):
            atom_to_mag[m_idx] = i

        moments = self.mag_moments[self.mag_indices]
        S_eff = np.abs(moments) / 2.0
        S_val = S_eff[0] if len(S_eff) > 0 else 1.0 
        anisotropy_term = S_val * 2.0 * K_anisotropy
        
        # 1. Pre-extract all valid Jij connections
        valid_bonds = []
        J_0 = np.zeros((self.n_mag_branches, self.n_mag_branches), dtype=np.float64)
        for row in self.jij_interactions:
            i, j = int(row[4]) - 1, int(row[5]) - 1
            mag_i, mag_j = atom_to_mag[i], atom_to_mag[j]
            if mag_i != -1 and mag_j != -1:
                J_val_scaled = row[3]
                J_0[mag_i, mag_j] += J_val_scaled
                valid_bonds.append((mag_i, mag_j, row[0], row[1], row[2], J_val_scaled))

        sum_J0_row = np.sum(J_0, axis=1)
        
        # Convert to fast NumPy arrays
        mag_i_arr = np.array([b[0] for b in valid_bonds])
        mag_j_arr = np.array([b[1] for b in valid_bonds])
        r_cart_arr = np.array([b[2:5] for b in valid_bonds]) * lattice_constant
        J_val_arr = np.array([b[5] for b in valid_bonds])

        # 2. Compute q_cart for ALL q-points at once (Shape: N_points x 3)
        q_frac = self.q_grid / self.mesh
        q_cart_all = np.dot(q_frac, self.reciprocal_lattice * 2.0 * np.pi)

        # 3. Vectorized Phase Calculation
        phases = np.dot(q_cart_all, r_cart_arr.T)
        
        # Compute exp(+-i * phase) * J for all q-points and bonds simultaneously
        exp_phases_k = np.exp(1j * phases) * J_val_arr
        exp_phases_m_k = np.exp(-1j * phases) * J_val_arr

        # Prepare base matrices and analytical derivative matrices
        B = self.reciprocal_lattice * 2.0 * np.pi
        B_dot_r = np.dot(B, r_cart_arr.T) # Shape: (3, num_bonds)
        

        # 4. Accumulate J_k and J_m_k for all q-points
        J_k_all = np.zeros((self.N, self.n_mag_branches, self.n_mag_branches), dtype=np.complex128)
        J_m_k_all = np.zeros((self.N, self.n_mag_branches, self.n_mag_branches), dtype=np.complex128)
        dJ_k_all = np.zeros((3, self.N, self.n_mag_branches, self.n_mag_branches), dtype=np.complex128)
        dJ_m_k_all = np.zeros((3, self.N, self.n_mag_branches, self.n_mag_branches), dtype=np.complex128)
        
        for b_idx in range(len(valid_bonds)):
            mi, mj = mag_i_arr[b_idx], mag_j_arr[b_idx]
            J_k_all[:, mi, mj] += exp_phases_k[:, b_idx]
            J_m_k_all[:, mi, mj] += exp_phases_m_k[:, b_idx]
            
            # ANALYTICAL GRADIENT FILLING:
            # You must multiply the phase factor by (i * B_dot_r) for each Cartesian axis alpha
            for alpha in range(3):
                # The derivative of exp(i * q * R) with respect to fractional index f_alpha
                # is (i * B_dot_r[alpha]) * exp(...)
                dJ_k_all[alpha, :, mi, mj] += 1j * B_dot_r[alpha, b_idx] * exp_phases_k[:, b_idx]
                dJ_m_k_all[alpha, :, mi, mj] += -1j * B_dot_r[alpha, b_idx] * exp_phases_m_k[:, b_idx]

        # 5. Build BdG and Diagonalize 
        for q_idx in range(self.N):
            J_k = J_k_all[q_idx]
            J_m_k = J_m_k_all[q_idx]
            
            Omega_k = np.zeros((self.n_mag_branches, self.n_mag_branches), dtype=np.complex128)
            Omega_m_k = np.zeros((self.n_mag_branches, self.n_mag_branches), dtype=np.complex128)
                
            for i in range(self.n_mag_branches):
                for j in range(self.n_mag_branches):
                    if i == j:
                        Omega_k[i, i] = S_val * (J_k[i, i] - sum_J0_row[i]) 
                        Omega_m_k[i, i] = S_val * (J_m_k[i, i] - sum_J0_row[i])
                    else:
                        Omega_k[i, j] = S_val * J_k[i, j]
                        Omega_m_k[i, j] = S_val * J_m_k[i, j]
                        
            H_BdG = np.zeros((2*self.n_mag_branches, 2*self.n_mag_branches), dtype=np.complex128)
            
            for m in range(self.n_mag_branches):
                for n in range(self.n_mag_branches):
                    val_n = -Omega_k[m, n]
                    if m == n:
                        val_n += anisotropy_term
                    H_BdG[m, n] = val_n

                    val_h = -Omega_m_k[n, m]
                    if m == n:
                        val_h += anisotropy_term
                    H_BdG[self.n_mag_branches + m, self.n_mag_branches + n] = val_h

            # Positive Definiteness Enforcement
            min_eig = np.min(np.linalg.eigvalsh(H_BdG))
            if min_eig <= 1e-8:
                np.fill_diagonal(H_BdG, H_BdG.diagonal() + np.abs(min_eig) + 1e-5)

            try:
                energies, para_unitary = diagonalize_bosonic_hamiltonian(H_BdG)
                self.w_mag[q_idx] = energies
                self.eig_mag[q_idx] = para_unitary

                # --- Hellmann-Feynman Analytical Gradient ---
                for alpha in range(3):
                    dOmega_k = np.zeros((self.n_mag_branches, self.n_mag_branches), dtype=np.complex128)
                    dOmega_m_k = np.zeros((self.n_mag_branches, self.n_mag_branches), dtype=np.complex128)
                    for i in range(self.n_mag_branches):
                        for j in range(self.n_mag_branches):
                            dOmega_k[i, j] = S_val * dJ_k_all[alpha, q_idx, i, j]
                            dOmega_m_k[i, j] = S_val * dJ_m_k_all[alpha, q_idx, i, j]
                            
                    dH_BdG = np.zeros((2*self.n_mag_branches, 2*self.n_mag_branches), dtype=np.complex128)
                    for m in range(self.n_mag_branches):
                        for n in range(self.n_mag_branches):
                            dH_BdG[m, n] = -dOmega_k[m, n]
                            dH_BdG[self.n_mag_branches + m, self.n_mag_branches + n] = -dOmega_m_k[n, m]
                    
                    # Project exact operator derivative onto unperturbed eigenvectors
                    # Removed J_metric to enforce true para-unitary projection
                    grad_w = np.diag(para_unitary.conj().T @ dH_BdG @ para_unitary).real
                    self.grad_f_mag[q_idx, :, alpha] = grad_w[:self.n_mag_branches]

            except RuntimeError as e:
                print(f"Warning at q_idx {q_idx}: {e}")
                self.w_mag[q_idx] = np.zeros(self.n_mag_branches)
    

    def _calculate_coupled_hamiltonian(self, q_cart_array, dyn_mat, K_anisotropy, lattice_constant, ref_omega=5.0, is_FM=True, return_full_matrices=False):
        print(f" -> Constructing Joint Magnon-Phonon BdG Matrix (Strict C++ Mapping)...")
        N_pts = q_cart_array.shape[0]
        num_phon = self.phon_branches
        num_mag = self.n_mag_branches
        dim = 2 * (num_phon + num_mag)

        w_hyb = np.zeros((N_pts, num_phon + num_mag), dtype=np.float64)
        eig_hyb = np.zeros((N_pts, dim, dim), dtype=np.complex128)
        
        # Prepare optional storage arrays
        if return_full_matrices:
            H_pre = np.zeros((N_pts, dim, dim), dtype=np.complex128)
            H_diag = np.zeros((N_pts, dim, dim), dtype=np.complex128)
        
        off_ph_p = 0
        off_mag_p = num_phon
        off_ph_h = num_phon + num_mag
        off_mag_h = 2 * num_phon + num_mag

        I_phon = np.eye(num_phon, dtype=np.complex128)

        CONV_FACTOR = 15.633302 * 4.135667696

        # Choose grid vs path data based on N_pts
        if N_pts == self.N:
            w_phon_source = self.w_phon
            eig_phon_source = self.eig_phon
        else:
            w_phon_source = self.path_w_phon
            eig_phon_source = self.path_eig_phon

        # --- Constants & Conversions ---
        hbar = 0.6582119569 # meV * ps
        DALTON_TO_meV_PS2_PER_A2 = 0.10364269

        moments = self.mag_moments[self.mag_indices]
        S_eff = np.abs(moments) / 2.0
        S_val = S_eff[0] if len(S_eff) > 0 else 1.0 
        print(f"Using S_val: {S_val}")
        anisotropy_term = S_val * 2.0 * K_anisotropy

        # ==========================================
        # 1. Precalculate Magnon Tensors
        # ==========================================
        atom_to_mag = np.full(self.l_atoms, -1, dtype=np.int32)
        for i, m_idx in enumerate(self.mag_indices):
            atom_to_mag[m_idx] = i

        valid_bonds_mag = []
        J_0 = np.zeros((num_mag, num_mag), dtype=np.float64)
        for row in self.jij_interactions:
            i, j = int(row[4]) - 1, int(row[5]) - 1
            mag_i, mag_j = atom_to_mag[i], atom_to_mag[j]
            if mag_i != -1 and mag_j != -1:
                J_val_scaled = row[3]
                J_0[mag_i, mag_j] += J_val_scaled
                valid_bonds_mag.append((mag_i, mag_j, row[0], row[1], row[2], J_val_scaled))

        sum_J0_row = np.sum(J_0, axis=1)

        J_k_all = np.zeros((N_pts, num_mag, num_mag), dtype=np.complex128) 
        J_m_k_all = np.zeros((N_pts, num_mag, num_mag), dtype=np.complex128) 
        
        if valid_bonds_mag:
            mag_i_arr = np.array([b[0] for b in valid_bonds_mag])
            mag_j_arr = np.array([b[1] for b in valid_bonds_mag])
            r_cart_arr = np.array([b[2:5] for b in valid_bonds_mag]) * lattice_constant
            J_val_arr = np.array([b[5] for b in valid_bonds_mag])

            phases_mag = np.dot(q_cart_array, r_cart_arr.T)
            
            exp_phases_k = np.exp(1j * phases_mag) * J_val_arr
            exp_phases_m_k = np.exp(-1j * phases_mag) * J_val_arr

            for b_idx in range(len(valid_bonds_mag)):
                mi, mj = mag_i_arr[b_idx], mag_j_arr[b_idx]
                J_k_all[:, mi, mj] += exp_phases_k[:, b_idx]
                J_m_k_all[:, mi, mj] += exp_phases_m_k[:, b_idx]

        # ==========================================
        # 1.5 Precalculate SLC Tensors (FM Case)
        # ==========================================
        V_plus_all = np.zeros((N_pts, num_mag, num_phon), dtype=np.complex128)
        V_minus_all = np.zeros((N_pts, num_mag, num_phon), dtype=np.complex128)

        if hasattr(self, 'slc_axis') and is_FM:
            for i in range(self.slc_axis.shape[0]):
                type_i = self.slc_types[i, 0] - 1
                type_j = self.slc_types[i, 1] - 1
                disp_atom_idx = self.slc_types[i, 2] - 1
                
                n_mag_i = atom_to_mag[type_i]
                n_mag_j = atom_to_mag[type_j]

                if n_mag_i == -1 or n_mag_j == -1:
                    continue

                mu = self.slc_axis[i]
                p_idx = 3 * disp_atom_idx + mu

                Jxz = self.slc_J[i, 0, 2]  # X=0, Z=2
                Jyz = self.slc_J[i, 1, 2]  # Y=1, Z=2

                phases_slc = np.dot(q_cart_array, self.slc_rik[i])
                phase_factor = np.exp(1j * phases_slc)

                V_plus_all[:, n_mag_i, p_idx] += (Jxz + 1j * Jyz) * phase_factor
                V_minus_all[:, n_mag_i, p_idx] += (Jxz - 1j * Jyz) * phase_factor

            for p in range(num_phon):
                atom_l = p // 3
                mass = self.atom_masses[atom_l]
                prefactor = np.sqrt((hbar * hbar) / (S_val * mass * DALTON_TO_meV_PS2_PER_A2 * ref_omega))
                
                V_plus_all[:, :, p] *= prefactor
                V_minus_all[:, :, p] *= prefactor




        # ==========================================
        # 2. HAMILTONIAN ASSEMBLY
        # ==========================================
        for q_idx in range(N_pts):
            H_BdG = np.zeros((dim, dim), dtype=np.complex128)
            
            """
            # --- Phonon Blocks from the Dynamical Matrix ---
            D_complex = dyn_mat[q_idx]
            D_meV2 = D_complex * (CONV_FACTOR ** 2)

            evals_D, evecs_D = np.linalg.eigh(D_meV2)
            if np.any(evals_D < 1e-5):
                evals_clamped = np.maximum(evals_D, 1e-5)
                D_meV2 = evecs_D @ np.diag(evals_clamped) @ evecs_D.conj().T

            A_phon = 0.5 * (D_meV2 / ref_omega + ref_omega * I_phon)
            B_phon = 0.5 * (D_meV2 / ref_omega - ref_omega * I_phon)

            H_BdG[off_ph_p:off_ph_p+num_phon, off_ph_p:off_ph_p+num_phon] = A_phon
            H_BdG[off_ph_h:off_ph_h+num_phon, off_ph_h:off_ph_h+num_phon] = A_phon
            H_BdG[off_ph_p:off_ph_p+num_phon, off_ph_h:off_ph_h+num_phon] = B_phon
            H_BdG[off_ph_h:off_ph_h+num_phon, off_ph_p:off_ph_p+num_phon] = B_phon
            """

            
            # --- Phonon Block from Eigenvalues ---
            #"""
            A_phon = np.diag(w_phon_source[q_idx])
            
            H_BdG[off_ph_p:off_ph_p+num_phon, off_ph_p:off_ph_p+num_phon] = A_phon
            H_BdG[off_ph_h:off_ph_h+num_phon, off_ph_h:off_ph_h+num_phon] = A_phon
            #"""

            # --- Magnon Blocks ---
            J_k = J_k_all[q_idx] 
            J_m_k = J_m_k_all[q_idx] 

            Omega_k = np.zeros((num_mag, num_mag), dtype=np.complex128)
            Omega_m_k = np.zeros((num_mag, num_mag), dtype=np.complex128)

            for i in range(num_mag):
                for j in range(num_mag):
                    if i == j:
                        Omega_k[i, i] = S_val * (J_k[i, i] - sum_J0_row[i]) 
                        Omega_m_k[i, i] = S_val * (J_m_k[i, i] - sum_J0_row[i])
                    else:
                        Omega_k[i, j] = S_val * J_k[i, j]
                        Omega_m_k[i, j] = S_val * J_m_k[i, j]

            for m in range(num_mag):
                for n in range(num_mag):
                    val_n = -Omega_k[m, n]
                    if m == n:
                        val_n += anisotropy_term
                    H_BdG[off_mag_p + m, off_mag_p + n] = val_n

                    val_h = -Omega_m_k[n, m]
                    if m == n:
                        val_h += anisotropy_term
                    H_BdG[off_mag_h + m, off_mag_h + n] = val_h

            # --- SLC Interaction Blocks ---
            if hasattr(self, 'slc_axis') and is_FM:
                Vp = V_plus_all[q_idx]
                Vm = V_minus_all[q_idx]

                # Enable or disable the following block to include SLC interactions
                """
                # 1. Normal Particle-Particle
                H_BdG[off_mag_p:off_mag_p+num_mag, off_ph_p:off_ph_p+num_phon] = Vp
                H_BdG[off_ph_p:off_ph_p+num_phon, off_mag_p:off_mag_p+num_mag] = Vp.conj().T

                # 2. Normal Hole-Hole
                H_BdG[off_mag_h:off_mag_h+num_mag, off_ph_h:off_ph_h+num_phon] = Vm
                H_BdG[off_ph_h:off_ph_h+num_phon, off_mag_h:off_mag_h+num_mag] = Vm.conj().T

                # 3. Anomalous Particle-Hole (Magnon_p, Phonon_h)
                H_BdG[off_mag_p:off_mag_p+num_mag, off_ph_h:off_ph_h+num_phon] = Vp
                H_BdG[off_ph_h:off_ph_h+num_phon, off_mag_p:off_mag_p+num_mag] = Vp.conj().T

                # 4. Anomalous Hole-Particle (Magnon_h, Phonon_p)
                H_BdG[off_mag_h:off_mag_h+num_mag, off_ph_p:off_ph_p+num_phon] = Vm
                H_BdG[off_ph_p:off_ph_p+num_phon, off_mag_h:off_mag_h+num_mag] = Vm.conj().T
                """

            # --- 3. Diagonalization ---
            if return_full_matrices:
                H_pre[q_idx] = H_BdG  # Save state prior to Cholesky projection

            # Positive Definiteness Enforcement (important near Gamma point)
            min_eig = np.min(np.linalg.eigvalsh(H_BdG))
            if min_eig <= 1e-8:
                np.fill_diagonal(H_BdG, H_BdG.diagonal() + np.abs(min_eig) + 1e-5)

            try:
                energies, para_unitary = diagonalize_bosonic_hamiltonian(H_BdG)
                w_hyb[q_idx] = energies
                eig_hyb[q_idx] = para_unitary

                if return_full_matrices:
                    # Construct the diagonalized energy matrix representation
                    # (Physical modes on the particle and hole blocks)
                    for idx in range(num_phon + num_mag):
                        H_diag[q_idx, idx, idx] = energies[idx]
                        H_diag[q_idx, (num_phon + num_mag) + idx, (num_phon + num_mag) + idx] = energies[idx]

            except RuntimeError as e:
                print(f"Warning at q_idx {q_idx}: {e}")
                w_hyb[q_idx] = np.zeros(num_phon + num_mag)
                
        if return_full_matrices:
            return w_hyb, eig_hyb, H_pre, H_diag
        
        return w_hyb, eig_hyb
    
    """
    def _compute_group_velocities(self):
        
        self.grad_f_phon = np.zeros((self.N, self.phon_branches, 3), dtype=np.float64)
        self.grad_f_mag = np.zeros((self.N, self.n_mag_branches, 3), dtype=np.float64)
        
        N_x, N_y, N_z = self.mesh
        
        # Vectorized Index lookups for Central Differences (Periodic wrapping via modulo)
        idx_x_plus = self.grid_map[(self.q_grid[:, 0] + 1) % N_x, self.q_grid[:, 1], self.q_grid[:, 2]]
        idx_x_minus = self.grid_map[(self.q_grid[:, 0] - 1) % N_x, self.q_grid[:, 1], self.q_grid[:, 2]]
        
        idx_y_plus = self.grid_map[self.q_grid[:, 0], (self.q_grid[:, 1] + 1) % N_y, self.q_grid[:, 2]]
        idx_y_minus = self.grid_map[self.q_grid[:, 0], (self.q_grid[:, 1] - 1) % N_y, self.q_grid[:, 2]]
        
        idx_z_plus = self.grid_map[self.q_grid[:, 0], self.q_grid[:, 1], (self.q_grid[:, 2] + 1) % N_z]
        idx_z_minus = self.grid_map[self.q_grid[:, 0], self.q_grid[:, 1], (self.q_grid[:, 2] - 1) % N_z]
        
        # Phonon Fractional Gradients: dω/df = (ω_plus - ω_minus) / (2 * df)
        # Because df = 1.0 / N_i, we multiply by (N_i / 2.0)
        self.grad_f_phon[:, :, 0] = (self.w_phon[idx_x_plus] - self.w_phon[idx_x_minus]) * (N_x / 2.0)
        self.grad_f_phon[:, :, 1] = (self.w_phon[idx_y_plus] - self.w_phon[idx_y_minus]) * (N_y / 2.0)
        self.grad_f_phon[:, :, 2] = (self.w_phon[idx_z_plus] - self.w_phon[idx_z_minus]) * (N_z / 2.0)
        
        # Magnon Fractional Gradients
        self.grad_f_mag[:, :, 0] = (self.w_mag[idx_x_plus] - self.w_mag[idx_x_minus]) * (N_x / 2.0)
        self.grad_f_mag[:, :, 1] = (self.w_mag[idx_y_plus] - self.w_mag[idx_y_minus]) * (N_y / 2.0)
        self.grad_f_mag[:, :, 2] = (self.w_mag[idx_z_plus] - self.w_mag[idx_z_minus]) * (N_z / 2.0)
        """
    

    def _compute_hybrid_group_velocities(self):
        """
        Computes hybrid energy gradients
        """
        print(" -> Computing exact hybrid group velocities via matrix projection...", flush=True)
        
        num_bands = self.phon_branches + self.n_mag_branches
        self.grad_f_hyb = np.zeros((self.N, num_bands, 3), dtype=np.float64)

        N_x, N_y, N_z = self.mesh

        # Periodic index lookups
        idx_xp = self.grid_map[(self.q_grid[:, 0] + 1) % N_x, self.q_grid[:, 1], self.q_grid[:, 2]]
        idx_xm = self.grid_map[(self.q_grid[:, 0] - 1) % N_x, self.q_grid[:, 1], self.q_grid[:, 2]]
        
        idx_yp = self.grid_map[self.q_grid[:, 0], (self.q_grid[:, 1] + 1) % N_y, self.q_grid[:, 2]]
        idx_ym = self.grid_map[self.q_grid[:, 0], (self.q_grid[:, 1] - 1) % N_y, self.q_grid[:, 2]]
        
        idx_zp = self.grid_map[self.q_grid[:, 0], self.q_grid[:, 1], (self.q_grid[:, 2] + 1) % N_z]
        idx_zm = self.grid_map[self.q_grid[:, 0], self.q_grid[:, 1], (self.q_grid[:, 2] - 1) % N_z]

        for q_idx in range(self.N):
            # 1. Central difference of the pre-diagonalized BdG Hamiltonian matrix
            dH_dx = (self.H_BdG_pre_diagonalized[idx_xp[q_idx]] - self.H_BdG_pre_diagonalized[idx_xm[q_idx]]) * (N_x / 2.0)
            dH_dy = (self.H_BdG_pre_diagonalized[idx_yp[q_idx]] - self.H_BdG_pre_diagonalized[idx_ym[q_idx]]) * (N_y / 2.0)
            dH_dz = (self.H_BdG_pre_diagonalized[idx_zp[q_idx]] - self.H_BdG_pre_diagonalized[idx_zm[q_idx]]) * (N_z / 2.0)

            # 2. Retrieve para-unitary eigenvector matrix
            T = self.Qmatrix[q_idx]
            T_dag = T.conj().T

            # 3. Project operator derivative onto physical basis (first 'num_bands' elements)
            grad_x = np.diag(T_dag @ dH_dx @ T).real[:num_bands]
            grad_y = np.diag(T_dag @ dH_dy @ T).real[:num_bands]
            grad_z = np.diag(T_dag @ dH_dz @ T).real[:num_bands]

            self.grad_f_hyb[q_idx, :, 0] = grad_x
            self.grad_f_hyb[q_idx, :, 1] = grad_y
            self.grad_f_hyb[q_idx, :, 2] = grad_z
        


    def _compute_group_velocities(self):
        """
        Computes the fractional gradients of the phonon energies ∇_f ω.
        Instead of differentiating sorted eigenvalues, we differentiate the 
        dynamical matrix operator D(q) and project it onto the unperturbed 
        eigenvectors using the Hellmann-Feynman theorem.
        """
        print(" -> Computing exact phonon energy gradients via Hellmann-Feynman projection...", flush=True)
        self.grad_f_phon = np.zeros((self.N, self.phon_branches, 3), dtype=np.float64)
        
        N_x, N_y, N_z = self.mesh
        CONV_FACTOR = 15.633302 * 4.135667696 # Converts raw VASP D(q) unit directly to meV
        
        for q_idx in range(self.N):
            qx, qy, qz = self.q_grid[q_idx]
            
            # Identify periodic neighbors
            idx_x_plus = self.grid_map[(qx + 1) % N_x, qy, qz]
            idx_x_minus = self.grid_map[(qx - 1) % N_x, qy, qz]
            idx_y_plus = self.grid_map[qx, (qy + 1) % N_y, qz]
            idx_y_minus = self.grid_map[qx, (qy - 1) % N_y, qz]
            idx_z_plus = self.grid_map[qx, qy, (qz + 1) % N_z]
            idx_z_minus = self.grid_map[qx, qy, (qz - 1) % N_z]
            
            # Get dynamical matrices at neighbors
            # If wrapped across BZ boundary (e.g. qx - 1 < 0), D(-q) = D*(q)
            D_x_p = self.dyn_mat_phon[idx_x_plus]
            D_x_m = self.dyn_mat_phon[idx_x_minus].conj() if qx == 0 else self.dyn_mat_phon[idx_x_minus]
            
            D_y_p = self.dyn_mat_phon[idx_y_plus]
            D_y_m = self.dyn_mat_phon[idx_y_minus].conj() if qy == 0 else self.dyn_mat_phon[idx_y_minus]
            
            D_z_p = self.dyn_mat_phon[idx_z_plus]
            D_z_m = self.dyn_mat_phon[idx_z_minus].conj() if qz == 0 else self.dyn_mat_phon[idx_z_minus]
            
            # Central finite difference of the DYNAMICAL MATRIX
            dD_dfx = (D_x_p - D_x_m) * (N_x / 2.0) * (CONV_FACTOR**2)
            dD_dfy = (D_y_p - D_y_m) * (N_y / 2.0) * (CONV_FACTOR**2)
            dD_dfz = (D_z_p - D_z_m) * (N_z / 2.0) * (CONV_FACTOR**2)
            
            # Enforce hermiticity on the gradient operators to prevent imaginary noise
            dD_dfx = 0.5 * (dD_dfx + dD_dfx.conj().T)
            dD_dfy = 0.5 * (dD_dfy + dD_dfy.conj().T)
            dD_dfz = 0.5 * (dD_dfz + dD_dfz.conj().T)
            
            # Project using the eigenvector to extract the exact branch derivative
            for b in range(self.phon_branches):
                omega = self.w_phon[q_idx, b]
                if omega < 1.0e-3:
                    self.grad_f_phon[q_idx, b, :] = 0.0
                    continue
                    
                # Reconstruct flat eigenvector from the (natom, 3) layout
                e_b = self.eig_phon[q_idx, b].flatten()
                
                # Hellmann-Feynman: d(omega^2)/dx = e^\dagger dD/dx e
                # Chain rule: d(omega)/dx = 1/(2*omega) * d(omega^2)/dx
                grad_x2 = np.dot(e_b.conj().T, np.dot(dD_dfx, e_b)).real
                grad_y2 = np.dot(e_b.conj().T, np.dot(dD_dfy, e_b)).real
                grad_z2 = np.dot(e_b.conj().T, np.dot(dD_dfz, e_b)).real
                
                self.grad_f_phon[q_idx, b, 0] = grad_x2 / (2.0 * omega)
                self.grad_f_phon[q_idx, b, 1] = grad_y2 / (2.0 * omega)
                self.grad_f_phon[q_idx, b, 2] = grad_z2 / (2.0 * omega)
        




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
            
            # DECOUPLING FIX: Read the reciprocal lattice strictly from the path file
            if 'reciprocal_lattice' in f:
                self.path_reciprocal_lattice = f['reciprocal_lattice'][:]
            else:
                print("Warning: reciprocal_lattice not found in band.h5. Falling back to grid lattice.")
                self.path_reciprocal_lattice = self.reciprocal_lattice
                
            # Convert to Cartesian wavevectors using the PATH's native lattice
            self.path_q_cart = np.dot(self.path_q_frac, self.path_reciprocal_lattice * 2.0 * np.pi)

            # Native cumulative path distance as computed by phonopy itself
            # for this run's own q-points (authoritative - avoids recomputing
            # it ourselves, which used minimum-image wrapping unsuited to a
            # non-periodic path and could silently disagree between runs).
            if 'distances' in f:
                self.path_dist_native = f['distances'][:]
            else:
                self.path_dist_native = None

            raw_frequencies = f['frequencies'][:]
            self.path_w_phon = raw_frequencies * 4.135667696
            
            if 'eigenvectors' in f:
                self.path_eig_phon = f['eigenvectors'][:]
            else:
                raise ValueError("Eigenvectors missing from HDF5.")
            
            if 'dynamical_matrices' in f:
                self.path_dyn_mat = f['dynamical_matrices'][:]
            else:
                print("Warning: 'dynamical_matrices' missing from path HDF5. Coupled Hamiltonian requires them!")

            if 'labels_json' in f:
                self.path_labels = json.loads(f['labels_json'][()])
            if 'segment_nqpoint' in f:
                self.path_segments = f['segment_nqpoint'][:]

        # 2. Allocate Path Arrays for Magnons
        self.path_w_mag = np.zeros((self.N_path, self.n_mag_branches), dtype=np.float64)
        self.path_eig_mag = np.zeros((self.N_path, 2*self.n_mag_branches, 2*self.n_mag_branches), dtype=np.complex128)

        # 3. Compute Exact Magnons for the Path using robust Vectorized Math
        atom_to_mag = np.full(self.l_atoms, -1, dtype=np.int32)
        for i, m_idx in enumerate(self.mag_indices):
            atom_to_mag[m_idx] = i

        moments = self.mag_moments[self.mag_indices]
        S_eff = np.abs(moments) / 2.0
        S_val = S_eff[0] if len(S_eff) > 0 else 1.0 
        anisotropy_term = S_val * 2.0 * K_anisotropy
        
        valid_bonds = []
        J_0 = np.zeros((self.n_mag_branches, self.n_mag_branches), dtype=np.float64)
        for row in self.jij_interactions:
            i, j = int(row[4]) - 1, int(row[5]) - 1
            mag_i, mag_j = atom_to_mag[i], atom_to_mag[j]
            if mag_i != -1 and mag_j != -1:
                J_val_scaled = row[3]
                J_0[mag_i, mag_j] += J_val_scaled
                valid_bonds.append((mag_i, mag_j, row[0], row[1], row[2], J_val_scaled))

        sum_J0_row = np.sum(J_0, axis=1)
        
        mag_i_arr = np.array([b[0] for b in valid_bonds])
        mag_j_arr = np.array([b[1] for b in valid_bonds])
        r_cart_arr = np.array([b[2:5] for b in valid_bonds]) * lattice_constant
        J_val_arr = np.array([b[5] for b in valid_bonds])

        # Vectorized Phase Calculation mapping to the independent path cartesian vectors
        phases = np.dot(self.path_q_cart, r_cart_arr.T)
        
        exp_phases_k = np.exp(1j * phases) * J_val_arr
        exp_phases_m_k = np.exp(-1j * phases) * J_val_arr

        J_k_all = np.zeros((self.N_path, self.n_mag_branches, self.n_mag_branches), dtype=np.complex128)
        J_m_k_all = np.zeros((self.N_path, self.n_mag_branches, self.n_mag_branches), dtype=np.complex128)
        
        for b_idx in range(len(valid_bonds)):
            mi, mj = mag_i_arr[b_idx], mag_j_arr[b_idx]
            J_k_all[:, mi, mj] += exp_phases_k[:, b_idx]
            J_m_k_all[:, mi, mj] += exp_phases_m_k[:, b_idx]
            
        for q_idx in range(self.N_path):
            J_k = J_k_all[q_idx]
            J_m_k = J_m_k_all[q_idx]
            
            Omega_k = np.zeros((self.n_mag_branches, self.n_mag_branches), dtype=np.complex128)
            Omega_m_k = np.zeros((self.n_mag_branches, self.n_mag_branches), dtype=np.complex128)
                
            for i in range(self.n_mag_branches):
                for j in range(self.n_mag_branches):
                    if i == j:
                        Omega_k[i, i] = S_val * (J_k[i, i] - sum_J0_row[i]) 
                        Omega_m_k[i, i] = S_val * (J_m_k[i, i] - sum_J0_row[i])
                    else:
                        Omega_k[i, j] = S_val * J_k[i, j]
                        Omega_m_k[i, j] = S_val * J_m_k[i, j]
                        
            H_BdG = np.zeros((2*self.n_mag_branches, 2*self.n_mag_branches), dtype=np.complex128)
            
            for m in range(self.n_mag_branches):
                for n in range(self.n_mag_branches):
                    val_n = -Omega_k[m, n]
                    if m == n:
                        val_n += anisotropy_term
                    H_BdG[m, n] = val_n

                    val_h = -Omega_m_k[n, m]
                    if m == n:
                        val_h += anisotropy_term
                    H_BdG[self.n_mag_branches + m, self.n_mag_branches + n] = val_h

            min_eig = np.min(np.linalg.eigvalsh(H_BdG))
            if min_eig <= 1e-8:
                np.fill_diagonal(H_BdG, H_BdG.diagonal() + np.abs(min_eig) + 1e-5)

            try:
                energies, para_unitary = diagonalize_bosonic_hamiltonian(H_BdG)
                self.path_w_mag[q_idx] = energies
                self.path_eig_mag[q_idx] = para_unitary
            except RuntimeError as e:
                print(f"Warning at path q_idx {q_idx}: {e}")
                self.path_w_mag[q_idx] = np.zeros(self.n_mag_branches)
                
        print(f"-> Evaluated {self.N_path} exact path points for magnons.")


    def get_path_distance_info(self):
        """
        Returns the cumulative path distance (1/Angstrom) for every explicit
        high-symmetry path point, plus a comment-line string recording the
        path distance of each high-symmetry label. Shared by every CSV writer
        that outputs per-path-point data, so post-processing can plot against
        a physically meaningful x-axis.

        Prefers path_dist_native (phonopy's own per-point 'distance' field,
        read straight from the path HDF5 in load_and_evaluate_path_hdf5) when
        available - that is the authoritative value for how far along the
        path phonopy itself placed each point. Falls back to recomputing it
        via consecutive-point Cartesian differences (minimum-image wrapped)
        only for older path files that don't carry 'distances'.
        """
        if getattr(self, 'path_dist_native', None) is not None:
            path_dist = self.path_dist_native
        else:
            path_dist = np.zeros(self.N_path)
            current_dist = 0.0
            for i in range(1, self.N_path):
                dq_frac = self.path_q_frac[i] - self.path_q_frac[i - 1]
                dq_frac = dq_frac - np.round(dq_frac)  # minimum image convention
                dq_cart = np.dot(dq_frac, self.path_reciprocal_lattice * 2.0 * np.pi)
                current_dist += np.linalg.norm(dq_cart)
                path_dist[i] = current_dist

        if hasattr(self, 'path_labels') and hasattr(self, 'path_segments'):
            entries = [f"{self.path_labels[0][0]}={path_dist[0]:.6f}"]
            idx = 0
            for i, seg_len in enumerate(self.path_segments):
                idx += seg_len
                entries.append(f"{self.path_labels[i][1]}={path_dist[idx - 1]:.6f}")
            label_comment = "# path_labels: " + ",".join(entries)
        else:
            label_comment = "# path_labels: unavailable"

        return path_dist, label_comment

    def save_hybrid_path_properties(self, filename="Outputs/hybrid_path_properties.csv"):
        """
        Extracts Spin AM, Phonon AM, and subsystem characters for all hybridized bands 
        along the high-symmetry path using T^dagger * O * T.
        """
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        print(f"\nWriting hybrid path properties to {filename}...")
        
        # 1. Acquire full Angular Momentum matrix (Z-axis)
        L_z_total = self.get_nambu_angular_momentum(axis=2)
        
        num_phon = self.phon_branches
        num_mag = self.n_mag_branches
        dim_block = num_phon + num_mag
        dim_total = 2 * dim_block
        
        # 2. Split L into Phonon and Spin components
        L_z_phon = np.zeros_like(L_z_total)
        L_z_spin = np.zeros_like(L_z_total)
        
        # Phonon block indices
        L_z_phon[:num_phon, :num_phon] = L_z_total[:num_phon, :num_phon]
        L_z_phon[dim_block:dim_block+num_phon, dim_block:dim_block+num_phon] = L_z_total[dim_block:dim_block+num_phon, dim_block:dim_block+num_phon]
        
        # Spin block indices
        L_z_spin[num_phon:dim_block, num_phon:dim_block] = L_z_total[num_phon:dim_block, num_phon:dim_block]
        L_z_spin[dim_block+num_phon:dim_total, dim_block+num_phon:dim_total] = L_z_total[dim_block+num_phon:dim_total, dim_block+num_phon:dim_total]
        
        # 3. Create Commutation Metrics (J) to extract character weights
        J_phon = np.zeros((dim_total, dim_total), dtype=np.float64)
        J_spin = np.zeros((dim_total, dim_total), dtype=np.float64)
        
        np.fill_diagonal(J_phon[:num_phon, :num_phon], 1.0)
        np.fill_diagonal(J_phon[dim_block:dim_block+num_phon, dim_block:dim_block+num_phon], -1.0)
        
        np.fill_diagonal(J_spin[num_phon:dim_block, num_phon:dim_block], 1.0)
        np.fill_diagonal(J_spin[dim_block+num_phon:dim_total, dim_block+num_phon:dim_total], -1.0)
        
        # 4. Iterate path and compute physical expectation values
        path_dist, label_comment = self.get_path_distance_info()
        with open(filename, 'w') as f:
            f.write(label_comment + "\n")
            header = ["q_idx", "qx", "qy", "qz", "band", "energy_meV", "phon_character", "mag_character", "phon_AM_z_hbar", "spin_AM_z_hbar", "path_dist"]
            f.write(",".join(header) + "\n")
            
            for q_idx in range(self.N_path):
                qx, qy, qz = self.path_q_frac[q_idx]
                T = self.path_eig_hyb[q_idx]
                T_dag = T.conj().T
                
                # Transform operators into diagonal hybrid basis
                AM_phon_q = T_dag @ L_z_phon @ T
                AM_spin_q = T_dag @ L_z_spin @ T
                char_phon_q = T_dag @ J_phon @ T
                char_spin_q = T_dag @ J_spin @ T
                
                # Loop only over physical positive-energy bands (first half of the BdG dimension)
                for b in range(dim_block): 
                    energy = self.path_w_hyb[q_idx, b]
                    
                    # Expectation values are on the diagonal
                    phon_char = char_phon_q[b, b].real
                    mag_char = char_spin_q[b, b].real
                    phon_AM = AM_phon_q[b, b].real
                    spin_AM = AM_spin_q[b, b].real
                    
                    row = [f"{q_idx}", f"{qx:.6f}", f"{qy:.6f}", f"{qz:.6f}", f"{b}",
                           f"{energy:.6f}", f"{phon_char:.6f}", f"{mag_char:.6f}",
                           f"{phon_AM:.6e}", f"{spin_AM:.6e}", f"{path_dist[q_idx]:.6f}"]
                    f.write(",".join(row) + "\n")
                    
        print(f"-> Done! (Sanity check: character sum phon+mag should equal exactly 1.0 for all bands)")


    def plot_hybridized_path_dispersions(self, filename="hybridized_path.png", color_mode='character'):
        """
        Plots the hybridized band structure.
        color_mode: None, 'character', 'spin_am', or 'phon_am'
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        k_distances = np.zeros(self.N_path)
        current_dist = 0.0
        
        # 1. Continuous Distance with BZ Wrapping
        for i in range(1, self.N_path):
            dq_frac = self.path_q_frac[i] - self.path_q_frac[i-1]
            dq_frac = dq_frac - np.round(dq_frac) # Apply Minimal Image Convention
            
            dq_cart = np.dot(dq_frac, self.path_reciprocal_lattice * 2.0 * np.pi)
            current_dist += np.linalg.norm(dq_cart)
            k_distances[i] = current_dist

        fig, ax = plt.subplots(figsize=(12/2.52, 14/2.52))
        
        num_bands = self.path_w_hyb.shape[1] 
        
        weights = None
        cbar_label = ""
        cmap = 'coolwarm'
        
        # Default fixed scale limits (Perfect for Character and absolute ℏ scales)
        vmin, vmax = -1.0, 1.0

        # 2. Extract Physical Properties on-the-fly for Color Mapping
        if color_mode and hasattr(self, 'path_eig_hyb'):
            weights = np.zeros((self.N_path, num_bands))
            num_phon = self.phon_branches
            num_mag = self.n_mag_branches
            dim_block = num_phon + num_mag
            dim_total = 2 * dim_block
            
            if color_mode == 'character':
                cbar_label = "Character (Blue=Phonon, Red=Magnon)"
                J_phon = np.zeros((dim_total, dim_total), dtype=np.float64)
                J_spin = np.zeros((dim_total, dim_total), dtype=np.float64)
                
                # Particle blocks
                np.fill_diagonal(J_phon[:num_phon, :num_phon], 1.0)
                np.fill_diagonal(J_spin[num_phon:dim_block, num_phon:dim_block], 1.0)
                # Hole blocks
                np.fill_diagonal(J_phon[dim_block:dim_block+num_phon, dim_block:dim_block+num_phon], -1.0)
                np.fill_diagonal(J_spin[dim_block+num_phon:dim_total, dim_block+num_phon:dim_total], -1.0)
                
                for q in range(self.N_path):
                    T = self.path_eig_hyb[q]
                    T_dag = T.conj().T
                    w_phon = np.diag(T_dag @ J_phon @ T).real[:num_bands]
                    w_mag = np.diag(T_dag @ J_spin @ T).real[:num_bands]
                    # Map to [-1, 1] where -1 is pure phonon, +1 is pure magnon
                    weights[q, :] = w_mag - w_phon 
                    
            elif color_mode in ['spin_am', 'phon_am']:
                L_z_total = self.get_nambu_angular_momentum(axis=2)
                Operator = np.zeros_like(L_z_total)
                
                if color_mode == 'spin_am':
                    cbar_label = r"Spin AM $S_z$ ($\hbar$)"
                    cmap = 'coolwarm'
                    Operator[num_phon:dim_block, num_phon:dim_block] = L_z_total[num_phon:dim_block, num_phon:dim_block]
                    Operator[dim_block+num_phon:, dim_block+num_phon:] = L_z_total[dim_block+num_phon:, dim_block+num_phon:]
                else:
                    cbar_label = r"Phonon AM $L_z$ ($\hbar$)"
                    cmap = 'coolwarm'
                    Operator[:num_phon, :num_phon] = L_z_total[:num_phon, :num_phon]
                    Operator[dim_block:dim_block+num_phon, dim_block:dim_block+num_phon] = L_z_total[dim_block:dim_block+num_phon, dim_block:dim_block+num_phon]
                
                for q in range(self.N_path):
                    T = self.path_eig_hyb[q]
                    w_op = np.diag(T.conj().T @ Operator @ T).real[:num_bands]
                    weights[q, :] = w_op
                    
                # NOTE: The dynamic auto-scaling block was removed here. 
                # vmin and vmax are locked at -1.0 and 1.0. 
                # This ensures the colorbar visually respects the absolute hbar limits.

        # 3. Plotting with Optional Scatter Overlay
        start_idx = 0
        sc = None
        for seg_len in self.path_segments:
            end_idx = start_idx + seg_len
            plot_end = end_idx + 1 if end_idx < self.N_path else end_idx
            
            for b in range(num_bands):
                if weights is not None:
                    # Draw a faint structural line to connect points visually
                    ax.plot(k_distances[start_idx:plot_end], self.path_w_hyb[start_idx:plot_end, b], color='lightgray', lw=0.5, zorder=1)
                    # Scatter plot the color-mapped values on top
                    sc = ax.scatter(k_distances[start_idx:plot_end], self.path_w_hyb[start_idx:plot_end, b], 
                                    c=weights[start_idx:plot_end, b], cmap=cmap, vmin=vmin, vmax=vmax, s=8, zorder=2, edgecolors='none')
                else:
                    # Fallback to plain uniform lines if no color mapping is requested
                    ax.plot(k_distances[start_idx:plot_end], self.path_w_hyb[start_idx:plot_end, b], color='#8c564b', lw=1.5, zorder=2)
                    
            start_idx = end_idx

        # 4. Axes Formatting
        ax.set_ylabel('Energy (meV)', fontsize=14, fontweight='bold')
        ax.set_xlim(0, k_distances[-1])
        ax.set_ylim(bottom=0)
        
        if sc is not None:
            cbar = fig.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label(cbar_label, fontsize=12, fontweight='bold')
        
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
        #plt.show()
        print(f"-> Saved hybridized band plot to '{filename}'")


    def plot_path_dispersions(self, filename="dispersion_verification.png"):
        """
        Plots the exact high-resolution path dispersions using the loaded HDF5 data.
        """
        import matplotlib.pyplot as plt
        
        if not hasattr(self, 'path_w_mag'):
            print("Error: Must call load_and_evaluate_path_hdf5() before plotting.")
            return

        k_distances = np.zeros(self.N_path)
        for i in range(1, self.N_path):
            dq_frac = self.path_q_frac[i] - self.path_q_frac[i-1]
            # DECOUPLING FIX: Use the native path reciprocal lattice for the X-axis
            dq_cart = np.dot(dq_frac, self.path_reciprocal_lattice * 2.0 * np.pi)
            k_distances[i] = k_distances[i-1] + np.linalg.norm(dq_cart)

        fig, ax = plt.subplots(figsize=(10/2.52, 12/2.52))
        
        for b in range(self.phon_branches):
            label = 'Phonons' if b == 0 else ""
            ax.plot(k_distances, self.path_w_phon[:, b], color='#1f77b4', lw=2, label=label)
            
        for b in range(self.n_mag_branches):
            label = 'Magnons' if b == 0 else ""
            ax.plot(k_distances, self.path_w_mag[:, b], color='#d62728', lw=2, linestyle='--', label=label)

        ax.set_ylabel('Energy (meV)', fontsize=14, fontweight='bold')
        ax.set_xlim(0, k_distances[-1])
        ax.set_ylim(bottom=0)
        ax.grid(True, axis='y', linestyle=':', color='gray', alpha=0.5)
        ax.legend(loc='upper right', fontsize=12, framealpha=1.0)
        
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
        #plt.show()
        print(f"-> Saved true path dispersion plot to '{filename}'")

    def use_full_bz_as_path(self):
        """
        Uses the entire Monkhorst-Pack grid as the "path", so the hybrid path
        lifetime pipeline evaluates 1/tau over the whole Brillouin zone. Drop-in
        replacement for load_and_evaluate_path_hdf5() - no band.h5 is read and
        no second diagonalization is done; every array the path kernels consume
        already exists for the grid with identical shapes, so this just aliases.

        Note the deliberate frac/cart asymmetry, mirroring how the kernels
        already treat their grid legs: path_q_frac is the UNCENTERED q/mesh (it
        is only ever used modulo 1 for grid rounding), while path_q_cart is the
        CENTERED cartesian vector (it sets the phase factors in calc_vertex_V,
        where a reciprocal-lattice offset would NOT cancel, since r_ij / r_ik
        are inter-atomic rather than lattice vectors).
        """
        print("\nUsing the full Brillouin zone as the evaluation set (no path file).")

        self.N_path = self.N
        self.path_q_frac = self.q_grid.astype(np.float64) / self.mesh
        self.path_q_cart = self.q_grid_cart
        self.path_reciprocal_lattice = self.reciprocal_lattice

        self.path_w_hyb   = self.w_hyb
        self.path_eig_hyb = self.Qmatrix
        self.path_w_phon  = self.w_phon
        self.path_eig_phon = self.eig_phon
        self.path_w_mag   = self.w_mag
        self.path_eig_mag = self.eig_mag
        self.path_dyn_mat = self.dyn_mat_phon

        # A cumulative arc length is meaningless for an unordered BZ sweep;
        # report |q| from Gamma so path_dist stays physically plottable.
        self.path_dist_native = np.linalg.norm(self.path_q_cart, axis=1)

        # Stale high-symmetry labels would mislabel the output.
        for attr in ("path_labels", "path_segments"):
            if hasattr(self, attr):
                delattr(self, attr)

        print(f"-> Full-BZ evaluation set: {self.N_path} q-points, "
              f"{self.phon_branches + self.n_mag_branches} hybrid branches.")


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

    def extract_full_grid_hybrid_properties(self):
        """
        Extracts Spin AM, Phonon AM, and subsystem characters for all hybridized bands 
        across the entire BZ grid using T^dagger * O * T.
        """
        print(" -> Extracting Angular Momentum and Character for the full BZ...")
        L_z_total = self.get_nambu_angular_momentum(axis=2)
        
        num_phon = self.phon_branches
        num_mag = self.n_mag_branches
        dim_block = num_phon + num_mag
        dim_total = 2 * dim_block
        
        # Isolate blocks
        L_z_phon = np.zeros_like(L_z_total)
        L_z_spin = np.zeros_like(L_z_total)
        
        L_z_phon[:num_phon, :num_phon] = L_z_total[:num_phon, :num_phon]
        L_z_phon[dim_block:dim_block+num_phon, dim_block:dim_block+num_phon] = L_z_total[dim_block:dim_block+num_phon, dim_block:dim_block+num_phon]
        
        L_z_spin[num_phon:dim_block, num_phon:dim_block] = L_z_total[num_phon:dim_block, num_phon:dim_block]
        L_z_spin[dim_block+num_phon:dim_total, dim_block+num_phon:dim_total] = L_z_total[dim_block+num_phon:dim_total, dim_block+num_phon:dim_total]
        
        # Commutation Metrics
        J_phon = np.zeros((dim_total, dim_total), dtype=np.float64)
        J_spin = np.zeros((dim_total, dim_total), dtype=np.float64)
        
        np.fill_diagonal(J_phon[:num_phon, :num_phon], 1.0)
        np.fill_diagonal(J_phon[dim_block:dim_block+num_phon, dim_block:dim_block+num_phon], -1.0)
        
        np.fill_diagonal(J_spin[num_phon:dim_block, num_phon:dim_block], 1.0)
        np.fill_diagonal(J_spin[dim_block+num_phon:dim_total, dim_block+num_phon:dim_total], -1.0)
        
        # Allocate output arrays mapped to (N_points, physical_branches)
        phon_chars = np.zeros((self.N, dim_block), dtype=np.float64)
        mag_chars = np.zeros((self.N, dim_block), dtype=np.float64)
        phon_ams = np.zeros((self.N, dim_block), dtype=np.float64)
        spin_ams = np.zeros((self.N, dim_block), dtype=np.float64)
        
        for q_idx in range(self.N):
            T = self.Qmatrix[q_idx]
            T_dag = T.conj().T
            
            # Diagonal components for the physical (particle) block
            phon_chars[q_idx, :] = np.diag(T_dag @ J_phon @ T).real[:dim_block]
            mag_chars[q_idx, :]  = np.diag(T_dag @ J_spin @ T).real[:dim_block]
            phon_ams[q_idx, :]   = np.diag(T_dag @ L_z_phon @ T).real[:dim_block]
            spin_ams[q_idx, :]   = np.diag(T_dag @ L_z_spin @ T).real[:dim_block]
            
        return phon_chars, mag_chars, phon_ams, spin_ams


    def get_nambu_angular_momentum(self, axis):
        """
        Constructs the Nambu-space Angular Momentum matrix (L) for a given axis.
        axis: 0 for X, 1 for Y, 2 for Z
        Output is strictly in units of ħ.
        """
        num_atoms = self.l_atoms
        num_mag_atoms = self.n_mag_branches
        
        dim_block = 3 * num_atoms + num_mag_atoms
        dim_total = 2 * dim_block
        
        L_matrix = np.zeros((dim_total, dim_total), dtype=np.complex128)
        
        # Offsets mapping to the exact basis: [Phonon, Magnon, Phonon*, Magnon*]
        off_ph_p = 0
        off_mag_p = 3 * num_atoms
        off_ph_h = dim_block
        off_mag_h = dim_block + 3 * num_atoms
        
        # 1. Levi-Civita Symbol Helper
        def epsilon(alpha, beta, ax):
            if ax == 0: # X-axis
                if alpha == 1 and beta == 2: return 1
                if alpha == 2 and beta == 1: return -1
            elif ax == 1: # Y-axis
                if alpha == 2 and beta == 0: return 1
                if alpha == 0 and beta == 2: return -1
            elif ax == 2: # Z-axis
                if alpha == 0 and beta == 1: return 1
                if alpha == 1 and beta == 0: return -1
            return 0

        # 2. Phonon Block (Orbital Angular Momentum)
        for l in range(num_atoms):
            base_idx_p = off_ph_p + 3 * l
            base_idx_h = off_ph_h + 3 * l
            
            for alpha in range(3):
                for beta in range(3):
                    eps = epsilon(alpha, beta, axis)
                    if eps != 0:
                        val = -1j * eps
                        
                        # Particle Block
                        L_matrix[base_idx_p + alpha, base_idx_p + beta] = val
                        # Hole Block: O_hole = -O_particle^*. Since val = -i*hbar*eps is
                        # purely imaginary, -(val)* = val, so the hole block equals the
                        # particle block unchanged.
                        L_matrix[base_idx_h + alpha, base_idx_h + beta] = val

        # 3. Magnon Block (Spin Angular Momentum - Sz)
        # Magnons only carry intrinsic spin angular momentum along the quantization axis
        if axis == 2:
            for m_idx in range(num_mag_atoms):
                # self.mag_indices maps the magnon branch back to the original atom index
                atom_idx = self.mag_indices[m_idx]
                mag_moment = self.mag_moments[atom_idx]
                
                # In antiferromagnets:
                # Sublattice UP (M > 0): Excitations carry -1 spin (reduces moment)
                # Sublattice DOWN (M < 0): Excitations carry +1 spin (increases z-component)
                spin_val = -1.0 if mag_moment > 0 else 1.0
                
                # Particle Block
                L_matrix[off_mag_p + m_idx, off_mag_p + m_idx] = spin_val
                
                # Hole Block
                # For strictly real diagonal operators: O_hole = -O_particle
                L_matrix[off_mag_h + m_idx, off_mag_h + m_idx] = -spin_val
                
        return L_matrix
    

@cuda.jit
def phase_1_scan_path(mesh, grid_q_frac, grid_q_cart, grid_map, 
                    path_q_frac, path_q_cart, path_w_phon, path_w_mag, path_eig_phon,
                    w_phon_grid, w_mag_grid, eig_phon_grid,
                    grad_f_phon, grad_f_mag,
                    slc_axis, slc_rij, slc_rik, slc_J, slc_types, 
                    smearing, chan_indices, chan_weights, channel_count, 
                    atom_masses, mag_moments, gamma_idx):
    """Scans phase space integrating exact path points against the BZ grid using Adaptive Broadening."""
    path_idx, k_idx = cuda.grid(2)
    
    if path_idx >= path_q_frac.shape[0] or k_idx >= grid_q_frac.shape[0]: 
        return
        
    n_mag = path_w_mag.shape[1]
    n_phon = path_w_phon.shape[1]
    
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

                # ---------------------------------------------------------
                # Unified magnon channel: M(path,n) <-> M(k,m) + Ph(k-path,lam)
                # Single delta / full bidirectional bracket (mirrors the grid's
                # unified Process 0), vertex evaluated at (kp=path, q=k-path) to
                # match the confirmed phonon-equation vertex convention.
                # ---------------------------------------------------------
                dE_mag = path_w_mag[path_idx, n] - w_mag_grid[k_idx, m] - w_phon_grid[idx_qmink, lam]
                var_mag = 0.0
                for i in range(3):
                    d_g = -grad_f_mag[k_idx, m, i] + grad_f_phon[idx_qmink, lam, i]
                    step_w = d_g / mesh[i]
                    var_mag += step_w * step_w

                sigma_raw_mag = smearing * math.sqrt(var_mag / 12.0)
                sigma_mag = sigma_raw_mag if sigma_raw_mag > 1e-5 else 1e-5
                
                if abs(dE_mag) < 2.0 * sigma_mag:
                    weight = (0.4179 / sigma_mag) * math.exp(-0.5 * (dE_mag*dE_mag) / (sigma_mag*sigma_mag))
                    kpx, kpy, kpz = path_q_cart[path_idx, 0], path_q_cart[path_idx, 1], path_q_cart[path_idx, 2]
                    qx = grid_q_cart[k_idx, 0] - path_q_cart[path_idx, 0]
                    qy = grid_q_cart[k_idx, 1] - path_q_cart[path_idx, 1]
                    qz = grid_q_cart[k_idx, 2] - path_q_cart[path_idx, 2]
                    V_sq = calc_vertex_V_path(kpx, kpy, kpz, qx, qy, qz, gammax, gammay, gammaz, lam, n, m, slc_axis, slc_rij, slc_rik, slc_J, slc_types, eig_phon_grid[idx_kminq], w_phon_grid[idx_kminq, lam], atom_masses, mag_moments)
                    c_idx = cuda.atomic.add(channel_count, 0, 1)
                    if c_idx < chan_indices.shape[1]:
                        chan_indices[0, c_idx] = 0; chan_indices[1, c_idx] = path_idx; chan_indices[2, c_idx] = k_idx
                        chan_indices[3, c_idx] = idx_qmink; chan_indices[4, c_idx] = n; chan_indices[5, c_idx] = m; chan_indices[6, c_idx] = lam
                        chan_weights[c_idx] = V_sq * weight

                # ---------------------------------------------------------
                # Unified phonon channel: Ph(path,lam) <-> M(k,n) - M(k-path,m)
                # Vertex evaluated at (kp=k, q=-path), matching the confirmed
                # V^{+-}_{k,-q,lam,nm} directly. Already correct as originally
                # written (this is the old "Process 2"); kept unchanged.
                # ---------------------------------------------------------
                dE_phon = path_w_phon[path_idx, lam] + w_mag_grid[idx_kminq, m] - w_mag_grid[k_idx, n]
                var_phon = 0.0
                for i in range(3):
                    d_g = grad_f_mag[idx_kminq, m, i] - grad_f_mag[k_idx, n, i]
                    step_w = d_g / mesh[i]
                    var_phon += step_w * step_w

                sigma_raw_phon = smearing * math.sqrt(var_phon / 12.0)
                sigma_phon = sigma_raw_phon if sigma_raw_phon > 1e-5 else 1e-5

                if abs(dE_phon) < 2.0 * sigma_phon:
                    weight = (0.4179 / sigma_phon) * math.exp(-0.5 * (dE_phon*dE_phon) / (sigma_phon*sigma_phon))
                    kpx, kpy, kpz = grid_q_cart[k_idx, 0], grid_q_cart[k_idx, 1], grid_q_cart[k_idx, 2]
                    qx, qy, qz = -path_q_cart[path_idx, 0], -path_q_cart[path_idx, 1], -path_q_cart[path_idx, 2]
                    V_sq = calc_vertex_V_path(kpx, kpy, kpz, qx, qy, qz, gammax, gammay, gammaz, lam, n, m, slc_axis, slc_rij, slc_rik, slc_J, slc_types, path_eig_phon[path_idx], path_w_phon[path_idx, lam], atom_masses, mag_moments)
                    c_idx = cuda.atomic.add(channel_count, 0, 1)
                    if c_idx < chan_indices.shape[1]:
                        chan_indices[0, c_idx] = 2; chan_indices[1, c_idx] = path_idx; chan_indices[2, c_idx] = k_idx
                        chan_indices[3, c_idx] = idx_kminq; chan_indices[4, c_idx] = n; chan_indices[5, c_idx] = m; chan_indices[6, c_idx] = lam
                        chan_weights[c_idx] = V_sq * weight



                """
                # ---------------------------------------------------------
                # 0: Magnon Emission (Magnon on path -> Magnon + Phonon on grid)
                # ---------------------------------------------------------
                dE_0 = path_w_mag[path_idx, n] - w_mag_grid[k_idx, m] - w_phon_grid[idx_qmink, lam]
                var_0 = 0.0
                for i in range(3):
                    d_g = -grad_f_mag[k_idx, m, i] + grad_f_phon[idx_qmink, lam, i]
                    step_w = d_g / mesh[i]
                    var_0 += step_w * step_w
                
                sigma_raw_0 = smearing * math.sqrt(var_0 / 12.0)
                sigma_0 = sigma_raw_0 if sigma_raw_0 > 1e-5 else 1e-5
                cutoff_0 = 2.0 * sigma_0
                
                if abs(dE_0) < cutoff_0:
                    weight = (0.4179 / sigma_0) * math.exp(-0.5 * (dE_0*dE_0) / (sigma_0*sigma_0))
                    kpx, kpy, kpz = path_q_cart[path_idx, 0], path_q_cart[path_idx, 1], path_q_cart[path_idx, 2]
                    qx, qy, qz = grid_q_cart[idx_qmink, 0], grid_q_cart[idx_qmink, 1], grid_q_cart[idx_qmink, 2]
                    V_sq = calc_vertex_V_path(kpx, kpy, kpz, qx, qy, qz, gammax, gammay, gammaz, lam, n, m, slc_axis, slc_rij, slc_rik, slc_J, slc_types, eig_phon_grid[idx_qmink], w_phon_grid[idx_qmink, lam], atom_masses, mag_moments)
                    c_idx = cuda.atomic.add(channel_count, 0, 1)
                    if c_idx < chan_indices.shape[1]:
                        chan_indices[0, c_idx] = 0; chan_indices[1, c_idx] = path_idx; chan_indices[2, c_idx] = k_idx
                        chan_indices[3, c_idx] = idx_qmink; chan_indices[4, c_idx] = n; chan_indices[5, c_idx] = m; chan_indices[6, c_idx] = lam
                        chan_weights[c_idx] = V_sq * weight

                # ---------------------------------------------------------
                # 1: Magnon Absorption (Magnon on path + Phonon on grid -> Magnon on grid)
                # ---------------------------------------------------------
                dE_1 = path_w_mag[path_idx, n] - w_mag_grid[k_idx, m] + w_phon_grid[idx_kminq, lam]
                var_1 = 0.0
                for i in range(3):
                    d_g = -grad_f_mag[k_idx, m, i] + grad_f_phon[idx_kminq, lam, i]
                    step_w = d_g / mesh[i]
                    var_1 += step_w * step_w
                
                sigma_raw_1 = smearing * math.sqrt(var_1 / 12.0)
                sigma_1 = sigma_raw_1 if sigma_raw_1 > 1e-5 else 1e-5
                cutoff_1 = 3.0 * sigma_1
                
                if abs(dE_1) < cutoff_1:
                    weight = (0.4179 / sigma_1) * math.exp(-0.5 * (dE_1*dE_1) / (sigma_1*sigma_1))
                    kpx, kpy, kpz = path_q_cart[path_idx, 0], path_q_cart[path_idx, 1], path_q_cart[path_idx, 2]
                    qx, qy, qz = grid_q_cart[idx_kminq, 0], grid_q_cart[idx_kminq, 1], grid_q_cart[idx_kminq, 2]
                    V_sq = calc_vertex_V_path(kpx, kpy, kpz, qx, qy, qz, gammax, gammay, gammaz, lam, m, n, slc_axis, slc_rij, slc_rik, slc_J, slc_types, eig_phon_grid[idx_kminq], w_phon_grid[idx_kminq, lam], atom_masses, mag_moments)
                    c_idx = cuda.atomic.add(channel_count, 0, 1)
                    if c_idx < chan_indices.shape[1]:
                        chan_indices[0, c_idx] = 1; chan_indices[1, c_idx] = path_idx; chan_indices[2, c_idx] = k_idx
                        chan_indices[3, c_idx] = idx_kminq; chan_indices[4, c_idx] = n; chan_indices[5, c_idx] = m; chan_indices[6, c_idx] = lam
                        chan_weights[c_idx] = V_sq * weight

                # ---------------------------------------------------------
                # 2: Phonon Scattering (Phonon on path + Magnon on grid -> Magnon on grid)
                # ---------------------------------------------------------
                dE_2 = path_w_phon[path_idx, lam] + w_mag_grid[idx_kminq, m] - w_mag_grid[k_idx, n]
                var_2 = 0.0
                for i in range(3):
                    d_g = grad_f_mag[idx_kminq, m, i] - grad_f_mag[k_idx, n, i]
                    step_w = d_g / mesh[i]
                    var_2 += step_w * step_w
                
                sigma_raw_2 = smearing * math.sqrt(var_2 / 12.0)
                sigma_2 = sigma_raw_2 if sigma_raw_2 > 1e-5 else 1e-5
                cutoff_2 = 3.0 * sigma_2
                
                if abs(dE_2) < cutoff_2:
                    weight = (0.4179 / sigma_2) * math.exp(-0.5 * (dE_2*dE_2) / (sigma_2*sigma_2))
                    kpx, kpy, kpz = grid_q_cart[k_idx, 0], grid_q_cart[k_idx, 1], grid_q_cart[k_idx, 2]
                    qx, qy, qz = -path_q_cart[path_idx, 0], -path_q_cart[path_idx, 1], -path_q_cart[path_idx, 2]
                    V_sq = calc_vertex_V_path(kpx, kpy, kpz, qx, qy, qz, gammax, gammay, gammaz, lam, n, m, slc_axis, slc_rij, slc_rik, slc_J, slc_types, path_eig_phon[path_idx], path_w_phon[path_idx, lam], atom_masses, mag_moments)
                    c_idx = cuda.atomic.add(channel_count, 0, 1)
                    if c_idx < chan_indices.shape[1]:
                        chan_indices[0, c_idx] = 2; chan_indices[1, c_idx] = path_idx; chan_indices[2, c_idx] = k_idx
                        chan_indices[3, c_idx] = idx_kminq; chan_indices[4, c_idx] = n; chan_indices[5, c_idx] = m; chan_indices[6, c_idx] = lam
                        chan_weights[c_idx] = V_sq * weight
                """

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
def gpu_copysign(x, y):
    """NVVM-safe replacement for math.copysign"""
    if y >= 0.0:
        return abs(x)
    else:
        return -abs(x)

@cuda.jit(device=True)
def calc_fourier_transform_vec(kpx, kpy, kpz, qx, qy, qz, slc_axis, slc_rij, slc_rik, slc_J, slc_types, n_type, m_type, l_type, mu_type, J_tilde_out):
    
    for a in range(3):
        for b in range(3):
            J_tilde_out[a, b] = 0.0 + 0.0j

    for i in range(slc_axis.shape[0]):
        
        # Multiply the boolean logic by 1.0
        match_mask = 1.0 * (
            (slc_axis[i] == mu_type) & 
            (slc_types[i, 0] == n_type) & 
            (slc_types[i, 1] == m_type) & 
            (slc_types[i, 2] == l_type)
        )

        if match_mask == 0.0:
            continue
        
        phase_val = (kpx * slc_rij[i, 0] + kpy * slc_rij[i, 1] + kpz * slc_rij[i, 2]) + \
                    (qx * slc_rik[i, 0] + qy * slc_rik[i, 1] + qz * slc_rik[i, 2])
        
        phase_factor = math.cos(phase_val) + 1j * math.sin(phase_val)
        
        for a in range(3):
            for b in range(3):
                J_tilde_out[a, b] += (slc_J[i, a, b] * phase_factor) * match_mask

@cuda.jit(device=True)
def calc_vertex_V_path(kpx, kpy, kpz, qx, qy, qz, gammax, gammay, gammaz, lambda_phon, n, m, 
                    slc_axis, slc_rij, slc_rik, slc_J, slc_types, 
                    eig_phon_q, omega, atom_masses, mag_moments):
    """Calculates the scattering vertex specifically for explicitly projected wavevectors."""
    if omega < 1E-1:
        return 0.0
    
    hbar = 0.6582119569 # meV * ps
    DALTON_TO_meV_PS2_PER_A2 = 0.10364269
    
    S_n = math.fabs(mag_moments[n] / 2.0 ) 
    S_m = math.fabs(mag_moments[m] / 2.0 )

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
            calc_fourier_transform_vec(kpx - qx, kpy - qy, kpz - qz, qx, qy, qz, slc_axis, slc_rij, slc_rik, slc_J, slc_types, n + 1, m + 1, l + 1, mu, J_tilde_dyn)
            W_dynamic = (J_tilde_dyn[0, 0] +  J_tilde_dyn[1, 1] - 1j  * J_tilde_dyn[0, 1] + 1j  * J_tilde_dyn[1, 0]) / math.sqrt(S_n * S_m)
            
            W_static = 0.0 + 0.0j
            if n == m: 
                for mp in range(num_mag_branches):
                    if math.fabs(mag_moments[mp]) > 1e-2:
                        #calc_fourier_transform_vec(gammax, gammay, gammaz, qx, qy, qz, slc_axis, slc_rij, slc_rik, slc_J, slc_types, n + 1, mp + 1, l + 1, mu, J_tilde_stat)
                        calc_fourier_transform_vec(0.0, 0.0, 0.0, qx, qy, qz, slc_axis, slc_rij, slc_rik, slc_J, slc_types, n + 1, mp + 1, l + 1, mu, J_tilde_stat)
                        
                        W_static += (2.0 / S_n)  * J_tilde_stat[2, 2]
                        
            V_complex += disp_amp * e_mu * (W_dynamic - W_static)
            
    return (V_complex.real**2 + V_complex.imag**2)


@cuda.jit(device=True)
def calc_vertex_V(kpx, kpy, kpz, qx, qy, qz, q_idx, lambda_phon, n, m, grid_map, slc_axis, slc_rij, slc_rik, slc_J, slc_types, eig_phon, w_phon, atom_masses, mag_moments):
    
    omega = w_phon[q_idx, lambda_phon]

    if omega < 1.0:
        return 0
    
    omega_mask = 1.0 * (omega >= 1e-3)
    
    # Branchless replacement: 
    # If mask==1, omega_safe = omega * 1 + 0 = omega
    # If mask==0, omega_safe = omega * 0 + 1.0 = 1.0 (Strictly positive, avoids NaN)
    omega_safe = (omega * omega_mask) + (1.0 - omega_mask)

    hbar = 0.6582119569 # meV * ps
    DALTON_TO_meV_PS2_PER_A2 = 0.10364269 
    
    S_n = math.fabs(mag_moments[n] / 2.0)
    S_m = math.fabs(mag_moments[m] / 2.0)

    J_tilde_dyn = cuda.local.array((3, 3), dtype=np.complex128)
    J_tilde_stat = cuda.local.array((3, 3), dtype=np.complex128)

    V_complex = 0.0 + 0.0j
    num_atoms = atom_masses.shape[0]
    num_mag_branches = mag_moments.shape[0]

    is_n_eq_m_mask = 1.0 * (n == m)

    # W^{+-}(k, q) is built from J~(k - q, q): the first Fourier argument is
    # conjugate to r_ij and is the DIFFERENCE of the vertex's two momentum
    # labels, not the first label itself. (The static term below correctly
    # passes 0 into the same slot, per J~(0, q).)
    kmqx = kpx - qx
    kmqy = kpy - qy
    kmqz = kpz - qz

    for l in range(num_atoms):
        mass_l = atom_masses[l] * DALTON_TO_meV_PS2_PER_A2
        disp_amp = math.sqrt((hbar * hbar) / (2.0 * mass_l * omega))

        for mu in range(3):
            e_mu = eig_phon[q_idx, lambda_phon, l, mu]

            calc_fourier_transform_vec(kmqx, kmqy, kmqz, qx, qy, qz, slc_axis, slc_rij, slc_rik, slc_J, slc_types, n + 1, m + 1, l + 1, mu, J_tilde_dyn)

            J_xx = J_tilde_dyn[0, 0]
            J_yy = J_tilde_dyn[1, 1]
            J_xy = J_tilde_dyn[0, 1]
            J_yx = J_tilde_dyn[1, 0]

            W_dynamic = (J_xx + J_yy -
                         1j * J_xy +
                         1j * J_yx) / math.sqrt(S_n * S_m)

            W_static = 0.0 + 0.0j

            for mp in range(num_mag_branches):
                if math.fabs(mag_moments[mp]) <= 1e-2:
                    continue

                calc_fourier_transform_vec(0.0, 0.0, 0.0, qx, qy, qz, slc_axis, slc_rij, slc_rik, slc_J, slc_types, n + 1, mp + 1, l + 1, mu, J_tilde_stat)

                W_static += (2.0 / S_n) * J_tilde_stat[2, 2]

            W_tot = W_dynamic - (W_static * is_n_eq_m_mask)
            V_complex += disp_amp * e_mu * W_tot

    return V_complex #* omega_mask


# ==========================================
# Hybrid vertex caching
# ==========================================
# V1/V2/V3 (inside calc_hybrid_vertex_Gamma) depend only on momenta and
# (n, m, lam) - never on the hybrid band indices (alpha, alpha', alpha'').
# The original implementation recomputed all of them (including the
# expensive SLC-tensor scan inside calc_vertex_V) once per hybrid band
# triple - up to num_bands**3 times per (grid/path) momentum pair, even
# though the result is identical every time. fill_hybrid_vertex_cache
# computes them once per momentum pair instead; calc_hybrid_vertex_Gamma
# then just looks them up.
#
# MAX_HYB_MAG_BRANCHES/MAX_HYB_PHON_BRANCHES are CUDA local-array
# compile-time bounds (numba requires a fixed shape). Raise them if a
# material exceeds these branch counts - current materials are well within
# them (bccFe: 1 mag / 3 phon; CrI3: 2 mag / 24 phon).
MAX_HYB_MAG_BRANCHES = 2
MAX_HYB_PHON_BRANCHES = 24


@cuda.jit(device=True)
def fill_hybrid_vertex_cache(
    kx, ky, kz, qx, qy, qz, q_idx,
    minus_k_plus_qx, minus_k_plus_qy, minus_k_plus_qz, minus_k_plus_q_idx,
    grid_map, slc_axis, slc_rij, slc_rik, slc_J, slc_types,
    eig_phon, w_phon, atom_masses, mag_moments, num_phon, num_mag,
    V1_cache, V2_cache, V3_cache
):
    """
    Fills the V1/V2/V3 caches (indexed [n, m, lam]) for one symmetrization
    configuration. kx,ky,kz / qx,qy,qz play whichever of the "k"/"q" roles
    the caller needs (swapped between the Gamma_kq and Gamma_qk configurations
    in calc_symmetrized_hybrid_vertex_squared); minus_k_plus_q is the same
    in both configurations.
    """
    for n in range(num_mag):
        for m in range(num_mag):
            for lam in range(num_phon):
                V1_cache[n, m, lam] = calc_vertex_V(
                -minus_k_plus_qx, -minus_k_plus_qy, -minus_k_plus_qz, qx, qy, qz, q_idx, lam, n, m,
                grid_map, slc_axis, slc_rij, slc_rik, slc_J, slc_types,
                eig_phon, w_phon, atom_masses, mag_moments
                )
                V2_cache[n, m, lam] = calc_vertex_V(
                    -qx, -qy, -qz, minus_k_plus_qx, minus_k_plus_qy, minus_k_plus_qz, minus_k_plus_q_idx, lam, n, m,
                    grid_map, slc_axis, slc_rij, slc_rik, slc_J, slc_types,
                    eig_phon, w_phon, atom_masses, mag_moments
                )
                V3_cache[n, m, lam] = calc_vertex_V(
                    -kx, -ky, -kz, qx, qy, qz, q_idx, lam, n, m,
                    grid_map, slc_axis, slc_rij, slc_rik, slc_J, slc_types,
                    eig_phon, w_phon, atom_masses, mag_moments
                )


@cuda.jit(device=True)
def fill_hybrid_vertex_cache_pathq(
    kx, ky, kz, qx, qy, qz, path_q_idx,
    minus_k_plus_qx, minus_k_plus_qy, minus_k_plus_qz, minus_k_plus_q_idx,
    grid_map, slc_axis, slc_rij, slc_rik, slc_J, slc_types,
    eig_phon, w_phon, path_eig_phon, path_w_phon,
    atom_masses, mag_moments, num_phon, num_mag,
    V1_cache, V2_cache, V3_cache
):
    """
    Variant of fill_hybrid_vertex_cache for the configuration where the vertex's
    "q" leg is the exact high-symmetry path point.

    V1 and V3 both carry the phonon factor e^mu_{l,lam}(q) / sqrt(omega_{q,lam})
    evaluated at q, so with q = path they read the path's OWN phonon
    eigenvectors/frequencies rather than snapping to the nearest grid point.
    V2's phonon leg is -(k+q), a genuine grid point, so it keeps the grid arrays.
    """
    for n in range(num_mag):
        for m in range(num_mag):
            for lam in range(num_phon):
                V1_cache[n, m, lam] = calc_vertex_V(
                    -minus_k_plus_qx, -minus_k_plus_qy, -minus_k_plus_qz, qx, qy, qz, path_q_idx, lam, n, m,
                    grid_map, slc_axis, slc_rij, slc_rik, slc_J, slc_types,
                    path_eig_phon, path_w_phon, atom_masses, mag_moments
                )
                V2_cache[n, m, lam] = calc_vertex_V(
                    -qx, -qy, -qz, minus_k_plus_qx, minus_k_plus_qy, minus_k_plus_qz, minus_k_plus_q_idx, lam, n, m,
                    grid_map, slc_axis, slc_rij, slc_rik, slc_J, slc_types,
                    eig_phon, w_phon, atom_masses, mag_moments
                )
                V3_cache[n, m, lam] = calc_vertex_V(
                    -kx, -ky, -kz, qx, qy, qz, path_q_idx, lam, n, m,
                    grid_map, slc_axis, slc_rij, slc_rik, slc_J, slc_types,
                    path_eig_phon, path_w_phon, atom_masses, mag_moments
                )


@cuda.jit(device=True)
def calc_hybrid_vertex_Gamma(
    k_idx, q_idx, k_plus_q_idx, minus_k_idx, minus_q_idx, minus_k_plus_q_idx,
    alpha, alpha_prime, alpha_double_prime,
    Qmatrix, V1_cache, V2_cache, V3_cache, num_phon, num_mag
):
    """
    Computes the highly-coupled 3-particle hybridized vertex Gamma, reusing
    precomputed V1/V2/V3 (see fill_hybrid_vertex_cache) instead of
    recomputing them for every hybrid band triple.
    """
    Gamma = 0.0 + 0.0j

    # N is the dimension block size: total number of physical bosonic modes
    N_half = num_phon + num_mag

    for n in range(num_mag):
        I_n = num_phon + n
        for m in range(num_mag):
            I_m = num_phon + m
            for lam in range(num_phon):

                # Term 1: V^{+-}_{k+q, q}
                V1 = V1_cache[n, m, lam]
                P1 = Qmatrix[q_idx, lam, alpha_double_prime] + Qmatrix[q_idx, lam + N_half, alpha_double_prime]
                Q1_n = Qmatrix[k_plus_q_idx, I_n, alpha].conjugate()
                Q1_m = Qmatrix[k_idx, I_m, alpha_prime]

                term1 = V1 * P1 * Q1_n * Q1_m

                # Term 2: V^{+-}_{-q, -(k+q)}
                V2 = V2_cache[n, m, lam]
                P2 = Qmatrix[minus_k_plus_q_idx, lam, alpha].conjugate() + Qmatrix[minus_k_plus_q_idx, lam + N_half, alpha].conjugate()
                Q2_n = Qmatrix[minus_q_idx, I_n, alpha_double_prime + N_half].conjugate()
                Q2_m = Qmatrix[k_idx, I_m, alpha_prime]

                term2 = V2 * P2 * Q2_n * Q2_m

                # Term 3: V^{+-}_{-k, q}
                V3 = V3_cache[n, m, lam]
                P3 = Qmatrix[q_idx, lam, alpha_double_prime] + Qmatrix[q_idx, lam + N_half, alpha_double_prime]
                Q3_n = Qmatrix[minus_k_idx, I_n, alpha_prime + N_half].conjugate()
                Q3_m = Qmatrix[minus_k_plus_q_idx, I_m, alpha + N_half]

                term3 = V3 * P3 * Q3_n * Q3_m

                # Accumulate the total hybridized vertex
                Gamma += (term1 + term2 + term3)

    return Gamma


@cuda.jit(device=True)
def calc_symmetrized_hybrid_vertex_squared(
    k_idx, q_idx, k_plus_q_idx, minus_k_idx, minus_q_idx, minus_k_plus_q_idx,
    alpha, alpha_prime, alpha_double_prime,
    Qmatrix,
    V1_cache_kq, V2_cache_kq, V3_cache_kq,
    V1_cache_qk, V2_cache_qk, V3_cache_qk,
    num_phon, num_mag):
    """
    Computes the squared absolute value of the symmetrized hybridized vertex,
    reusing the two precomputed per-configuration vertex caches (kq and qk)
    instead of recomputing V1/V2/V3 for every hybrid band triple.
    """

    # Gamma(k, q, alpha, alpha', alpha'')
    Gamma_kq = calc_hybrid_vertex_Gamma(
        k_idx, q_idx, k_plus_q_idx, minus_k_idx, minus_q_idx, minus_k_plus_q_idx,
        alpha, alpha_prime, alpha_double_prime,
        Qmatrix, V1_cache_kq, V2_cache_kq, V3_cache_kq, num_phon, num_mag
    )

    # 2. Swapped configuration: Gamma(q, k, alpha, alpha'', alpha')
    Gamma_qk = calc_hybrid_vertex_Gamma(
        q_idx, k_idx, k_plus_q_idx, minus_q_idx, minus_k_idx, minus_k_plus_q_idx,
        alpha, alpha_double_prime, alpha_prime,
        Qmatrix, V1_cache_qk, V2_cache_qk, V3_cache_qk, num_phon, num_mag
    )

    # 3. Symmetrize and return the squared absolute value |Gamma_sym|^2
    Gamma_sym = 0.5 * (Gamma_kq + Gamma_qk)
    return Gamma_sym.real**2 + Gamma_sym.imag**2


@cuda.jit(device=True)
def calc_hybrid_vertex_Gamma_path(
    path_idx, minus_path_grid_idx,
    k_idx, q_idx, minus_k_idx, minus_q_idx,
    alpha, alpha_prime, alpha_double_prime,
    Qmatrix, path_Qmatrix, V1_cache, V2_cache, V3_cache, num_phon, num_mag
):
    """
    The parent leg (k+q) is pinned to an
    exact high-symmetry path point (path_idx), while k_idx/q_idx (the two children)
    remain grid indices, exactly as in the grid version. Reuses precomputed
    V1/V2/V3 (see fill_hybrid_vertex_cache) instead of recomputing them for
    every hybrid band triple.

    The parent's own Qmatrix row is read from path_Qmatrix (exact, no approximation).
    The "-parent" leg is unavoidably approximated onto the nearest grid point
    (minus_path_grid_idx) for its Qmatrix row and its calc_vertex_V eig_phon/w_phon
    lookup, since no continuous "-path" hybrid diagonalization exists - there is no
    grid point at -path in general. The exact (unrounded) cartesian coordinate of
    -path is still used for the phase factors inside calc_vertex_V (baked into the
    cache); only the array lookups that require a discrete index are grid-snapped.
    """
    Gamma = 0.0 + 0.0j


    # Question: Why is minus_path_grid_idx used in the Qmatrix? E.g. for term2, the maths says: V_-q,-(k+q) P^*(-(k+q))? What is the meaning of path here? Dont we cimply call the function with k and q?

    N_half = num_phon + num_mag

    for n in range(num_mag):
        I_n = num_phon + n
        for m in range(num_mag):
            I_m = num_phon + m
            for lam in range(num_phon):

                # Term 1: both children on the grid, parent (k+q) exact via path_Qmatrix
                V1 = V1_cache[n, m, lam]
                P1 = Qmatrix[q_idx, lam, alpha_double_prime] + Qmatrix[q_idx, lam + N_half, alpha_double_prime]
                Q1_n = path_Qmatrix[path_idx, I_n, alpha].conjugate()
                Q1_m = Qmatrix[k_idx, I_m, alpha_prime]

                term1 = V1 * P1 * Q1_n * Q1_m

                # Term 2: -(k+q) leg grid-snapped (minus_path_grid_idx) for both the
                # vertex's phonon lookup and its Qmatrix row.
                V2 = V2_cache[n, m, lam]
                P2 = Qmatrix[minus_path_grid_idx, lam, alpha].conjugate() + Qmatrix[minus_path_grid_idx, lam + N_half, alpha].conjugate()
                Q2_n = Qmatrix[minus_q_idx, I_n, alpha_double_prime + N_half].conjugate()
                Q2_m = Qmatrix[k_idx, I_m, alpha_prime]

                term2 = V2 * P2 * Q2_n * Q2_m

                # Term 3: -(k+q) leg grid-snapped again
                V3 = V3_cache[n, m, lam]
                P3 = Qmatrix[q_idx, lam, alpha_double_prime] + Qmatrix[q_idx, lam + N_half, alpha_double_prime]
                Q3_n = Qmatrix[minus_k_idx, I_n, alpha_prime + N_half].conjugate()
                Q3_m = Qmatrix[minus_path_grid_idx, I_m, alpha + N_half]

                term3 = V3 * P3 * Q3_n * Q3_m

                Gamma += (term1 + term2 + term3)

    return Gamma


@cuda.jit(device=True)
def calc_symmetrized_hybrid_vertex_squared_path(
    path_idx, minus_path_grid_idx,
    k_idx, q_idx, minus_k_idx, minus_q_idx,
    alpha, alpha_prime, alpha_double_prime,
    Qmatrix, path_Qmatrix,
    V1_cache_kq, V2_cache_kq, V3_cache_kq,
    V1_cache_qk, V2_cache_qk, V3_cache_qk,
    num_phon, num_mag):
    """
    The parent (path) leg is held fixed between the two symmetrization calls; only the two grid children swap
    roles. Reuses the two precomputed vertex caches instead of recomputing V1/V2/V3
    for every hybrid band triple.
    """

    Gamma_kq = calc_hybrid_vertex_Gamma_path(
        path_idx, minus_path_grid_idx,
        k_idx, q_idx, minus_k_idx, minus_q_idx,
        alpha, alpha_prime, alpha_double_prime,
        Qmatrix, path_Qmatrix, V1_cache_kq, V2_cache_kq, V3_cache_kq, num_phon, num_mag
    )

    Gamma_qk = calc_hybrid_vertex_Gamma_path(
        path_idx, minus_path_grid_idx,
        q_idx, k_idx, minus_q_idx, minus_k_idx,
        alpha, alpha_double_prime, alpha_prime,
        Qmatrix, path_Qmatrix, V1_cache_qk, V2_cache_qk, V3_cache_qk, num_phon, num_mag
    )

    Gamma_sym = 0.5 * (Gamma_kq + Gamma_qk)
    return Gamma_sym.real**2 + Gamma_sym.imag**2


@cuda.jit(device=True)
def calc_hybrid_vertex_Gamma_path_coal(
    parent_idx, minus_parent_idx,
    c1_idx, c1_is_path, minus_c1_idx,
    c2_idx, c2_is_path, minus_c2_idx,
    alpha, alpha_prime, alpha_double_prime,
    Qmatrix, path_Qmatrix, V1_cache, V2_cache, V3_cache, num_phon, num_mag
):
    """
    Coalescence-topology hybrid vertex: the PARENT (k+q) is a grid point and one
    of the two CHILDREN is the exact high-symmetry path point. This is the mirror
    of calc_hybrid_vertex_Gamma_path (parent = path point, both children on the
    grid), and it is what supplies the second RTA term - the one where the path
    mode absorbs a partner instead of splitting. Without it the path mode is
    structurally always a parent and its coalescence rate is never accumulated.

    c1_is_path / c2_is_path select which child slot reads its Nambu row from
    path_Qmatrix (exact) rather than Qmatrix (grid); exactly one is true. Every
    "minus" leg is a grid index - the negated path point has no exact grid
    counterpart, so it is snapped, the same unavoidable approximation already
    documented in calc_hybrid_vertex_Gamma_path.

    Term structure is identical to calc_hybrid_vertex_Gamma / _path (same three
    terms of Gamma^{alpha alpha' alpha''}_{k,q}); only which array each Nambu row
    is read from differs.
    """
    Gamma = 0.0 + 0.0j
    N_half = num_phon + num_mag

    for n in range(num_mag):
        I_n = num_phon + n
        for m in range(num_mag):
            I_m = num_phon + m
            for lam in range(num_phon):

                # P_{lam, alpha''}(q): the second child
                if c2_is_path:
                    P_c2 = (path_Qmatrix[c2_idx, lam, alpha_double_prime]
                            + path_Qmatrix[c2_idx, lam + N_half, alpha_double_prime])
                else:
                    P_c2 = (Qmatrix[c2_idx, lam, alpha_double_prime]
                            + Qmatrix[c2_idx, lam + N_half, alpha_double_prime])

                # Q_{I_m, alpha'}(k): the first child
                if c1_is_path:
                    Q_c1 = path_Qmatrix[c1_idx, I_m, alpha_prime]
                else:
                    Q_c1 = Qmatrix[c1_idx, I_m, alpha_prime]

                # Term 1: V^{+-}_{k+q, q}; parent is on the grid here
                V1 = V1_cache[n, m, lam]
                Q1_n = Qmatrix[parent_idx, I_n, alpha].conjugate()
                term1 = V1 * P_c2 * Q1_n * Q_c1

                # Term 2: V^{+-}_{-q, -(k+q)}
                V2 = V2_cache[n, m, lam]
                P2 = (Qmatrix[minus_parent_idx, lam, alpha].conjugate()
                      + Qmatrix[minus_parent_idx, lam + N_half, alpha].conjugate())
                Q2_n = Qmatrix[minus_c2_idx, I_n, alpha_double_prime + N_half].conjugate()
                term2 = V2 * P2 * Q2_n * Q_c1

                # Term 3: V^{+-}_{-k, q}
                V3 = V3_cache[n, m, lam]
                Q3_n = Qmatrix[minus_c1_idx, I_n, alpha_prime + N_half].conjugate()
                Q3_m = Qmatrix[minus_parent_idx, I_m, alpha + N_half]
                term3 = V3 * P_c2 * Q3_n * Q3_m

                Gamma += (term1 + term2 + term3)

    return Gamma


@cuda.jit(device=True)
def calc_symmetrized_hybrid_vertex_squared_path_coal(
    path_idx, minus_path_grid_idx,
    s_idx, minus_s_idx,
    k_idx, minus_k_idx,
    alpha, alpha_prime, alpha_double_prime,
    Qmatrix, path_Qmatrix,
    V1_cache_kq, V2_cache_kq, V3_cache_kq,
    V1_cache_qk, V2_cache_qk, V3_cache_qk,
    num_phon, num_mag):
    """
    Symmetrized coalescence vertex |Gamma~|^2 for
    path(alpha') + grid_k(alpha'') -> grid_s(alpha).

    Mirrors calc_symmetrized_hybrid_vertex_squared_path's Gamma_kq / Gamma_qk
    swap: the parent s is held fixed while the two children (the exact path
    point and the grid point k) exchange roles together with their band indices,
    realising Gamma~ = 1/2 (Gamma^{a a' a''}_{k,q} + Gamma^{a a'' a'}_{q,k}).
    """
    # Gamma(k = path, q = k_grid; alpha, alpha', alpha'')
    Gamma_kq = calc_hybrid_vertex_Gamma_path_coal(
        s_idx, minus_s_idx,
        path_idx, True, minus_path_grid_idx,
        k_idx, False, minus_k_idx,
        alpha, alpha_prime, alpha_double_prime,
        Qmatrix, path_Qmatrix, V1_cache_kq, V2_cache_kq, V3_cache_kq, num_phon, num_mag
    )

    # Gamma(k = k_grid, q = path; alpha, alpha'', alpha')
    Gamma_qk = calc_hybrid_vertex_Gamma_path_coal(
        s_idx, minus_s_idx,
        k_idx, False, minus_k_idx,
        path_idx, True, minus_path_grid_idx,
        alpha, alpha_double_prime, alpha_prime,
        Qmatrix, path_Qmatrix, V1_cache_qk, V2_cache_qk, V3_cache_qk, num_phon, num_mag
    )

    Gamma_sym = 0.5 * (Gamma_kq + Gamma_qk)
    return Gamma_sym.real**2 + Gamma_sym.imag**2

# ==========================================
# 2. GPU Kernels: The Main Phases
# ==========================================
@cuda.jit
def phase_1_scan(mesh, q_grid, q_grid_cart, grid_map, w_phon, w_mag, eig_phon, 
                 grad_f_phon, grad_f_mag,
                 slc_axis, slc_rij, slc_rik, slc_J, slc_types, 
                 base_smearing, chan_indices, chan_weights, channel_count, atom_masses, mag_moments, gamma_idx, min_sigma):
    """
    Scans phase space enforcing energy conservation via Adaptive Broadening 
    and strict Cartesian continuous vectors for exact Umklapp phase tracking.
    """
    q_idx, k_idx = cuda.grid(2)
    N = q_grid.shape[0]

    if q_idx >= N or k_idx >= N: 
        return
        
    n_mag = w_mag.shape[1]
    n_phon = w_phon.shape[1]
    
    # --- Mappings for Lookups ---
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
                # Process 0 (Unified): M(q) <--> M(k) + Ph(q-k)
                # ---------------------------------------------------------
                dE = w_mag[q_idx, n] - w_mag[k_idx, m] - w_phon[idx_qmink, lam]
                
                variance = 0.0
                for i in range(3):
                    d_g = -grad_f_mag[k_idx, m, i] + grad_f_phon[idx_qmink, lam, i]
                    step_width = d_g / mesh[i]
                    variance += step_width * step_width
                
                sigma_raw = base_smearing * math.sqrt(variance / 12.0)
                MIN_SIGMA = min_sigma  # meV

                

                sigma = sigma_raw if sigma_raw > MIN_SIGMA else MIN_SIGMA

                sigma = min_sigma

                if abs(dE) < 2.0 * sigma:
                    # 0.4179 normalizes the 2-sigma Gaussian
                    gaussian_norm = 0.4179 / sigma
                    delta_weight = gaussian_norm * math.exp(-0.5 * (dE * dE) / (sigma * sigma))
                    
                    kpx_cart, kpy_cart, kpz_cart = q_grid_cart[q_idx, 0], q_grid_cart[q_idx, 1], q_grid_cart[q_idx, 2]

                    #qx = q_grid_cart[idx_qmink, 0]
                    #qy = q_grid_cart[idx_qmink, 1]
                    #qz = q_grid_cart[idx_qmink, 2]

                    x_qmink_cart = q_grid_cart[q_idx, 0] - q_grid_cart[k_idx, 0]
                    y_qmink_cart = q_grid_cart[q_idx, 1] - q_grid_cart[k_idx, 1]
                    z_qmink_cart = q_grid_cart[q_idx, 2] - q_grid_cart[k_idx, 2]

                    x_kminq_cart = q_grid_cart[k_idx, 0] - q_grid_cart[q_idx, 0]
                    y_kminq_cart = q_grid_cart[k_idx, 1] - q_grid_cart[q_idx, 1]
                    z_kminq_cart = q_grid_cart[k_idx, 2] - q_grid_cart[q_idx, 2]
                    
                    V = calc_vertex_V(kpx_cart, kpy_cart, kpz_cart, x_qmink_cart, y_qmink_cart, z_qmink_cart, idx_qmink, lam, n, m, grid_map, slc_axis, slc_rij, slc_rik, slc_J, slc_types, eig_phon, w_phon, atom_masses, mag_moments)

                    V_Gmp = calc_vertex_V(kpx_cart, kpy_cart, kpz_cart, x_kminq_cart, y_kminq_cart, z_kminq_cart, idx_kminq, lam, n, m, grid_map, slc_axis, slc_rij, slc_rik, slc_J, slc_types, eig_phon, w_phon, atom_masses, mag_moments)
                    V_sq_Gmp = V_Gmp.real**2 + V_Gmp.imag**2

                    V_sq = V.real**2 + V.imag**2

                    c_idx = cuda.atomic.add(channel_count, 0, 1)
                    #if c_idx < chan_indices.shape[1]:

                    chan_indices[0, c_idx] = 0; chan_indices[1, c_idx] = q_idx; chan_indices[2, c_idx] = k_idx
                    chan_indices[3, c_idx] = idx_qmink; chan_indices[4, c_idx] = n; chan_indices[5, c_idx] = m; chan_indices[6, c_idx] = lam
                    chan_weights[c_idx] = V_sq_Gmp * delta_weight
                
                """
                # Process 1
                dE = w_mag[q_idx, n] - w_mag[k_idx, m] - w_phon[idx_qmink, lam]

                variance = 0.0
                for i in range(3):
                    d_g = -grad_f_mag[k_idx, m, i] + grad_f_phon[idx_qmink, lam, i]
                    step_width = d_g / mesh[i]
                    variance += step_width * step_width
                
                sigma_raw = base_smearing * math.sqrt(variance / 12.0)
                MIN_SIGMA = min_sigma # meV
                sigma = sigma_raw if sigma_raw > MIN_SIGMA else MIN_SIGMA


                if abs(dE) < 2.0 * sigma:
                    # 0.4179 normalizes the 2-sigma Gaussian
                    gaussian_norm = 0.4179 / sigma
                    delta_weight = gaussian_norm * math.exp(-0.5 * (dE * dE) / (sigma * sigma))
                    
                    kpx_cart, kpy_cart, kpz_cart = q_grid_cart[q_idx, 0], q_grid_cart[q_idx, 1], q_grid_cart[q_idx, 2]

                    #qx = q_grid_cart[idx_qmink, 0]
                    #qy = q_grid_cart[idx_qmink, 1]
                    #qz = q_grid_cart[idx_qmink, 2]

                    x_qmink_cart = q_grid_cart[q_idx, 0] - q_grid_cart[k_idx, 0]
                    y_qmink_cart = q_grid_cart[q_idx, 1] - q_grid_cart[k_idx, 1]
                    z_qmink_cart = q_grid_cart[q_idx, 2] - q_grid_cart[k_idx, 2]

                    V = calc_vertex_V(kpx_cart, kpy_cart, kpz_cart, x_qmink_cart, y_qmink_cart, z_qmink_cart, idx_qmink, lam, n, m, grid_map, slc_axis, slc_rij, slc_rik, slc_J, slc_types, eig_phon, w_phon, atom_masses, mag_moments)
                    V_sq = V.real**2 + V.imag**2

                    c_idx = cuda.atomic.add(channel_count, 0, 1)
                    if c_idx < chan_indices.shape[1]:
                        chan_indices[0, c_idx] = 0; chan_indices[1, c_idx] = q_idx; chan_indices[2, c_idx] = k_idx
                        chan_indices[3, c_idx] = idx_qmink; chan_indices[4, c_idx] = n; chan_indices[5, c_idx] = m; chan_indices[6, c_idx] = lam
                        chan_weights[c_idx] = V_sq * delta_weight


                    # Process 2
                    dE = w_mag[q_idx, n] - w_mag[k_idx, m] + w_phon[idx_kminq, lam]

                    variance = 0.0
                    for i in range(3):
                        d_g = -grad_f_mag[k_idx, m, i] + grad_f_phon[idx_kminq, lam, i]
                        step_width = d_g / mesh[i]
                        variance += step_width * step_width
                    
                    sigma_raw = base_smearing * math.sqrt(variance / 12.0)
                    sigma = sigma_raw if sigma_raw > MIN_SIGMA else MIN_SIGMA


                    if abs(dE) < 2.0 * sigma:
                        # 0.4179 normalizes the 2-sigma Gaussian
                        gaussian_norm = 0.4179 / sigma
                        delta_weight = gaussian_norm * math.exp(-0.5 * (dE * dE) / (sigma * sigma))
                        
                        kpx_cart, kpy_cart, kpz_cart = q_grid_cart[q_idx, 0], q_grid_cart[q_idx, 1], q_grid_cart[q_idx, 2]

                        x_kminq_cart = q_grid_cart[k_idx, 0] - q_grid_cart[q_idx, 0]
                        y_kminq_cart = q_grid_cart[k_idx, 1] - q_grid_cart[q_idx, 1]
                        z_kminq_cart = q_grid_cart[k_idx, 2] - q_grid_cart[q_idx, 2]

                        V = calc_vertex_V(kpx_cart, kpy_cart, kpz_cart, x_kminq_cart, y_kminq_cart, z_kminq_cart, idx_kminq, lam, n, m, grid_map, slc_axis, slc_rij, slc_rik, slc_J, slc_types, eig_phon, w_phon, atom_masses, mag_moments)
                        V_sq = V.real**2 + V.imag**2

                        c_idx = cuda.atomic.add(channel_count, 0, 1)
                        if c_idx < chan_indices.shape[1]:
                            chan_indices[0, c_idx] = 1; chan_indices[1, c_idx] = q_idx; chan_indices[2, c_idx] = k_idx
                            chan_indices[3, c_idx] = idx_kminq; chan_indices[4, c_idx] = n; chan_indices[5, c_idx] = m; chan_indices[6, c_idx] = lam
                            chan_weights[c_idx] = V_sq * delta_weight

                    # Process 3 
                    dE = w_mag[idx_kminq, n] - w_mag[k_idx, m] + w_phon[q_idx, lam]

                    variance = 0.0
                    for i in range(3):
                        d_g = grad_f_mag[idx_kminq, n, i] - grad_f_mag[k_idx, m, i]
                        step_width = d_g / mesh[i]
                        variance += step_width * step_width
                    
                    sigma_raw = base_smearing * math.sqrt(variance / 12.0)
                    sigma = sigma_raw if sigma_raw > MIN_SIGMA else MIN_SIGMA

                    if abs(dE) < 2.0 * sigma:
                        gaussian_norm = 0.4179 / sigma
                        delta_weight = gaussian_norm * math.exp(-0.5 * (dE * dE) / (sigma * sigma))
                        
                        kpx_cart, kpy_cart, kpz_cart = q_grid_cart[k_idx, 0], q_grid_cart[k_idx, 1], q_grid_cart[k_idx, 2]
                        qx_cart_cont = q_grid_cart[k_idx, 0] - q_grid_cart[idx_kminq, 0]
                        qy_cart_cont = q_grid_cart[k_idx, 1] - q_grid_cart[idx_kminq, 1]
                        qz_cart_cont = q_grid_cart[k_idx, 2] - q_grid_cart[idx_kminq, 2]
                        V = calc_vertex_V(kpx_cart, kpy_cart, kpz_cart, qx_cart_cont, qy_cart_cont, qz_cart_cont, q_idx, lam, m, n, grid_map, slc_axis, slc_rij, slc_rik, slc_J, slc_types, eig_phon, w_phon, atom_masses, mag_moments)
                        V_sq = V.real**2 + V.imag**2

                        c_idx = cuda.atomic.add(channel_count, 0, 1)
                        if c_idx < chan_indices.shape[1]:
                            chan_indices[0, c_idx] = 2; chan_indices[1, c_idx] = q_idx; chan_indices[2, c_idx] = k_idx
                            chan_indices[3, c_idx] = idx_kminq; chan_indices[4, c_idx] = n; chan_indices[5, c_idx] = m; chan_indices[6, c_idx] = lam
                            chan_weights[c_idx] = V_sq * delta_weight


                """
                    

@cuda.jit
def phase_1_scan_hybrid(mesh, q_grid, q_grid_cart, grid_map, w_hyb, grad_f_hyb, Qmatrix,
                        slc_axis, slc_rij, slc_rik, slc_J, slc_types,
                        eig_phon, w_phon, atom_masses, mag_moments,
                        base_smearing, min_sigma, chan_indices, chan_weights, channel_count, num_phon, num_mag):
    """
    Unified hybridized phase space scan. Because all particles are identical polarons, 
    every collision is mathematically represented by a single splitting topology.
    """
    q_idx, k_idx = cuda.grid(2)
    N = q_grid.shape[0]

    if q_idx >= N or k_idx >= N:
        return

    # Derive p_idx = q_idx - k_idx
    qx_p = (q_grid[q_idx, 0] - q_grid[k_idx, 0] + mesh[0]) % mesh[0]
    qy_p = (q_grid[q_idx, 1] - q_grid[k_idx, 1] + mesh[1]) % mesh[1]
    qz_p = (q_grid[q_idx, 2] - q_grid[k_idx, 2] + mesh[2]) % mesh[2]
    p_idx = grid_map[qx_p, qy_p, qz_p]

    mx_k = (-q_grid[k_idx, 0] + mesh[0]) % mesh[0]
    my_k = (-q_grid[k_idx, 1] + mesh[1]) % mesh[1]
    mz_k = (-q_grid[k_idx, 2] + mesh[2]) % mesh[2]
    minus_k_idx = grid_map[mx_k, my_k, mz_k]

    mx_p = (-q_grid[p_idx, 0] + mesh[0]) % mesh[0]
    my_p = (-q_grid[p_idx, 1] + mesh[1]) % mesh[1]
    mz_p = (-q_grid[p_idx, 2] + mesh[2]) % mesh[2]
    minus_p_idx = grid_map[mx_p, my_p, mz_p]

    mx_q = (-q_grid[q_idx, 0] + mesh[0]) % mesh[0]
    my_q = (-q_grid[q_idx, 1] + mesh[1]) % mesh[1]
    mz_q = (-q_grid[q_idx, 2] + mesh[2]) % mesh[2]
    minus_q_idx = grid_map[mx_q, my_q, mz_q]

    kx, ky, kz = q_grid_cart[k_idx, 0], q_grid_cart[k_idx, 1], q_grid_cart[k_idx, 2]

    px = q_grid_cart[q_idx, 0] - q_grid_cart[k_idx, 0]
    py = q_grid_cart[q_idx, 1] - q_grid_cart[k_idx, 1]
    pz = q_grid_cart[q_idx, 2] - q_grid_cart[k_idx, 2]

    mqx = -q_grid_cart[q_idx, 0]
    mqy = -q_grid_cart[q_idx, 1]
    mqz = -q_grid_cart[q_idx, 2]

    num_bands = num_phon + num_mag

    # Precompute the vertex caches once per (q_idx, k_idx) grid pair (see
    # fill_hybrid_vertex_cache) instead of once per hybrid band triple below.
    V1_cache_kq = cuda.local.array((MAX_HYB_MAG_BRANCHES, MAX_HYB_MAG_BRANCHES, MAX_HYB_PHON_BRANCHES), dtype=np.complex128)
    V2_cache_kq = cuda.local.array((MAX_HYB_MAG_BRANCHES, MAX_HYB_MAG_BRANCHES, MAX_HYB_PHON_BRANCHES), dtype=np.complex128)
    V3_cache_kq = cuda.local.array((MAX_HYB_MAG_BRANCHES, MAX_HYB_MAG_BRANCHES, MAX_HYB_PHON_BRANCHES), dtype=np.complex128)
    V1_cache_qk = cuda.local.array((MAX_HYB_MAG_BRANCHES, MAX_HYB_MAG_BRANCHES, MAX_HYB_PHON_BRANCHES), dtype=np.complex128)
    V2_cache_qk = cuda.local.array((MAX_HYB_MAG_BRANCHES, MAX_HYB_MAG_BRANCHES, MAX_HYB_PHON_BRANCHES), dtype=np.complex128)
    V3_cache_qk = cuda.local.array((MAX_HYB_MAG_BRANCHES, MAX_HYB_MAG_BRANCHES, MAX_HYB_PHON_BRANCHES), dtype=np.complex128)

    fill_hybrid_vertex_cache(
        kx, ky, kz, px, py, pz, p_idx, mqx, mqy, mqz, minus_q_idx,
        grid_map, slc_axis, slc_rij, slc_rik, slc_J, slc_types,
        eig_phon, w_phon, atom_masses, mag_moments, num_phon, num_mag,
        V1_cache_kq, V2_cache_kq, V3_cache_kq
    )
    fill_hybrid_vertex_cache(
        px, py, pz, kx, ky, kz, q_idx, mqx, mqy, mqz, minus_q_idx,
        grid_map, slc_axis, slc_rij, slc_rik, slc_J, slc_types,
        eig_phon, w_phon, atom_masses, mag_moments, num_phon, num_mag,
        V1_cache_qk, V2_cache_qk, V3_cache_qk
    )

    for b_q in range(num_bands):
        w_q = w_hyb[q_idx, b_q]
        for b_k in range(num_bands):
            w_k = w_hyb[k_idx, b_k]
            for b_p in range(num_bands):
                dE = w_q - w_k - w_hyb[p_idx, b_p]


                # Adaptive Broadening Calculation
                """
                variance = 0.0
                for i in range(3):
                    d_g = grad_f_hyb[q_idx, b_q, i] - grad_f_hyb[k_idx, b_k, i] - grad_f_hyb[p_idx, b_p, i]
                    step_width = d_g / mesh[i]
                    variance += step_width * step_width

                sigma_raw = base_smearing * math.sqrt(variance / 12.0)
                sigma = sigma_raw if sigma_raw > min_sigma else min_sigma
                """
                sigma = min_sigma

                cutoff = 2.0 * sigma

                if abs(dE) < cutoff:
                    gaussian_norm = 0.4179 / sigma
                    delta_weight = gaussian_norm * math.exp(-0.5 * (dE * dE) / (sigma * sigma))

                   #V_sq = 0.01

                    V_sq = calc_symmetrized_hybrid_vertex_squared(
                        k_idx, p_idx, q_idx, minus_k_idx, minus_p_idx, minus_q_idx,
                        b_q, b_k, b_p,
                        Qmatrix,
                        V1_cache_kq, V2_cache_kq, V3_cache_kq,
                        V1_cache_qk, V2_cache_qk, V3_cache_qk,
                        num_phon, num_mag
                    )

                    c_idx = cuda.atomic.add(channel_count, 0, 1)
                    if c_idx < chan_indices.shape[1]:
                        chan_indices[1, c_idx] = q_idx
                        chan_indices[2, c_idx] = k_idx
                        chan_indices[3, c_idx] = p_idx
                        chan_indices[4, c_idx] = b_q
                        chan_indices[5, c_idx] = b_k
                        chan_indices[6, c_idx] = b_p
                        chan_weights[c_idx] = V_sq * delta_weight


@cuda.jit
def phase_lifetime_hybrid(chan_indices, chan_weights, num_channels, n_hyb, gamma_hyb, N_points):
    """
    Computes SMRTA lifetimes dynamically applying splitting and coalescence kinematics.
    """
    idx = cuda.grid(1)
    if idx >= num_channels[0] or idx >= chan_weights.shape[0]: 
        return

    q_idx = chan_indices[1, idx]
    k_idx = chan_indices[2, idx]
    p_idx = chan_indices[3, idx]
    b_q   = chan_indices[4, idx]
    b_k   = chan_indices[5, idx]
    b_p   = chan_indices[6, idx]

    V_sq = chan_weights[idx]

    hbar = 0.6582119569 # meV * ps
    prefactor_split = (math.pi / hbar) / N_points
    prefactor_coal  = (2.0 * math.pi / hbar) / N_points
    
    n_q = n_hyb[q_idx, b_q]
    n_k = n_hyb[k_idx, b_k]
    n_p = n_hyb[p_idx, b_p]

    num_bands = n_hyb.shape[1]

    # 1. Splitting contribution to parent state (q)
    rate_q = prefactor_split * V_sq * (n_k + n_p + 1.0)
    cuda.atomic.add(gamma_hyb, q_idx * num_bands + b_q, rate_q)

    # 2. Coalescence contribution to child 1 state (k)
    rate_k = prefactor_coal * V_sq * (n_p - n_q)
    cuda.atomic.add(gamma_hyb, k_idx * num_bands + b_k, rate_k)

    # 3. Coalescence contribution to child 2 state (p)
    rate_p = prefactor_coal * V_sq * (n_k - n_q)
    cuda.atomic.add(gamma_hyb, p_idx * num_bands + b_p, rate_p)


@cuda.jit
def phase_1_scan_hybrid_path(mesh, q_grid, grid_q_frac, grid_q_cart, grid_map,
                              path_q_frac, path_q_cart, path_w_hyb, path_eig_hyb,
                              w_hyb, Qmatrix,
                              slc_axis, slc_rij, slc_rik, slc_J, slc_types,
                              eig_phon, w_phon, path_eig_phon, path_w_phon,
                              grad_f_hyb,
                              atom_masses, mag_moments,
                              base_smearing, min_sigma, chan_indices, chan_weights, channel_count,
                              num_phon, num_mag):
    """
      Scattering channels:
      c_type 0  SPLITTING    hybrid(path) -> hybrid(k) + hybrid(p),  p = round(path - k)
      c_type 1  COALESCENCE  hybrid(path) + hybrid(k) -> hybrid(s),  s = round(path + k)

    The path mode is the parent in the first and a child in
    the second. k, p and s are grid points.
    """
    path_idx, k_idx = cuda.grid(2)
    N_path = path_q_frac.shape[0]
    N_grid = grid_q_cart.shape[0]

    MIN_SCATTERING_ENERGY_MEV = 0.01

    if path_idx >= N_path or k_idx >= N_grid:
        return

    # p_idx = round_to_grid(path - k) | splitting
    px_int = int(math.floor(((path_q_frac[path_idx, 0] - grid_q_frac[k_idx, 0]) % 1.0) * mesh[0] + 0.5)) % mesh[0]
    py_int = int(math.floor(((path_q_frac[path_idx, 1] - grid_q_frac[k_idx, 1]) % 1.0) * mesh[1] + 0.5)) % mesh[1]
    pz_int = int(math.floor(((path_q_frac[path_idx, 2] - grid_q_frac[k_idx, 2]) % 1.0) * mesh[2] + 0.5)) % mesh[2]
    p_idx = grid_map[px_int, py_int, pz_int]

    # -k (exact grid negation - k is already on-grid)
    mx_k = (-q_grid[k_idx, 0] + mesh[0]) % mesh[0]
    my_k = (-q_grid[k_idx, 1] + mesh[1]) % mesh[1]
    mz_k = (-q_grid[k_idx, 2] + mesh[2]) % mesh[2]
    minus_k_idx = grid_map[mx_k, my_k, mz_k]

    # -p (exact grid negation - p is already on-grid)
    mx_p = (-q_grid[p_idx, 0] + mesh[0]) % mesh[0]
    my_p = (-q_grid[p_idx, 1] + mesh[1]) % mesh[1]
    mz_p = (-q_grid[p_idx, 2] + mesh[2]) % mesh[2]
    minus_p_idx = grid_map[mx_p, my_p, mz_p]

    # -path, rounded to the nearest grid point. This is the one unavoidable
    # approximation - see calc_hybrid_vertex_Gamma_path's docstring.
    mx_q = int(math.floor(((-path_q_frac[path_idx, 0]) % 1.0) * mesh[0] + 0.5)) % mesh[0]
    my_q = int(math.floor(((-path_q_frac[path_idx, 1]) % 1.0) * mesh[1] + 0.5)) % mesh[1]
    mz_q = int(math.floor(((-path_q_frac[path_idx, 2]) % 1.0) * mesh[2] + 0.5)) % mesh[2]
    minus_path_grid_idx = grid_map[mx_q, my_q, mz_q]

    # s = round_to_grid(path + k) | coalescence
    sx_int = int(math.floor(((path_q_frac[path_idx, 0] + grid_q_frac[k_idx, 0]) % 1.0) * mesh[0] + 0.5)) % mesh[0]
    sy_int = int(math.floor(((path_q_frac[path_idx, 1] + grid_q_frac[k_idx, 1]) % 1.0) * mesh[1] + 0.5)) % mesh[1]
    sz_int = int(math.floor(((path_q_frac[path_idx, 2] + grid_q_frac[k_idx, 2]) % 1.0) * mesh[2] + 0.5)) % mesh[2]
    s_idx = grid_map[sx_int, sy_int, sz_int]

    # -s (exact grid negation - s is already on-grid)
    mx_s = (-q_grid[s_idx, 0] + mesh[0]) % mesh[0]
    my_s = (-q_grid[s_idx, 1] + mesh[1]) % mesh[1]
    mz_s = (-q_grid[s_idx, 2] + mesh[2]) % mesh[2]
    minus_s_idx = grid_map[mx_s, my_s, mz_s]

    kx, ky, kz = grid_q_cart[k_idx, 0], grid_q_cart[k_idx, 1], grid_q_cart[k_idx, 2]

    px = path_q_cart[path_idx, 0] - grid_q_cart[k_idx, 0]
    py = path_q_cart[path_idx, 1] - grid_q_cart[k_idx, 1]
    pz = path_q_cart[path_idx, 2] - grid_q_cart[k_idx, 2]

    mqx = -path_q_cart[path_idx, 0]
    mqy = -path_q_cart[path_idx, 1]
    mqz = -path_q_cart[path_idx, 2]

    # Exact (unrounded) cartesian path+k and -(path+k), used for the vertex
    # phase factors; only the index lookups above are grid-snapped.
    sx_cart = path_q_cart[path_idx, 0] + grid_q_cart[k_idx, 0]
    sy_cart = path_q_cart[path_idx, 1] + grid_q_cart[k_idx, 1]
    sz_cart = path_q_cart[path_idx, 2] + grid_q_cart[k_idx, 2]

    num_bands = num_phon + num_mag

    # Precompute the vertex caches once per (path_idx, k_idx) pair
    # We have kq and qk caches for the symmetrized vertex calculation
    V1_cache_kq = cuda.local.array((MAX_HYB_MAG_BRANCHES, MAX_HYB_MAG_BRANCHES, MAX_HYB_PHON_BRANCHES), dtype=np.complex128)
    V2_cache_kq = cuda.local.array((MAX_HYB_MAG_BRANCHES, MAX_HYB_MAG_BRANCHES, MAX_HYB_PHON_BRANCHES), dtype=np.complex128)
    V3_cache_kq = cuda.local.array((MAX_HYB_MAG_BRANCHES, MAX_HYB_MAG_BRANCHES, MAX_HYB_PHON_BRANCHES), dtype=np.complex128)
    V1_cache_qk = cuda.local.array((MAX_HYB_MAG_BRANCHES, MAX_HYB_MAG_BRANCHES, MAX_HYB_PHON_BRANCHES), dtype=np.complex128)
    V2_cache_qk = cuda.local.array((MAX_HYB_MAG_BRANCHES, MAX_HYB_MAG_BRANCHES, MAX_HYB_PHON_BRANCHES), dtype=np.complex128)
    V3_cache_qk = cuda.local.array((MAX_HYB_MAG_BRANCHES, MAX_HYB_MAG_BRANCHES, MAX_HYB_PHON_BRANCHES), dtype=np.complex128)

    # kp
    fill_hybrid_vertex_cache(
        kx, ky, kz, px, py, pz, p_idx, mqx, mqy, mqz, minus_path_grid_idx,
        grid_map, slc_axis, slc_rij, slc_rik, slc_J, slc_types,
        eig_phon, w_phon, atom_masses, mag_moments, num_phon, num_mag,
        V1_cache_kq, V2_cache_kq, V3_cache_kq
    )

    # pk
    fill_hybrid_vertex_cache(
        px, py, pz, kx, ky, kz, k_idx, mqx, mqy, mqz, minus_path_grid_idx,
        grid_map, slc_axis, slc_rij, slc_rik, slc_J, slc_types,
        eig_phon, w_phon, atom_masses, mag_moments, num_phon, num_mag,
        V1_cache_qk, V2_cache_qk, V3_cache_qk
    )

    # ---------------------------------------------------------------------
    # SPLITTING channel: hybrid(path, b_q) -> hybrid(k, b_k) + hybrid(p, b_p).
    # Determines the  RTA term with weight (1 + n_k + n_p).
    # ---------------------------------------------------------------------
    for b_q in range(num_bands):
        w_q = path_w_hyb[path_idx, b_q]
        for b_k in range(num_bands):
            w_k = w_hyb[k_idx, b_k]
            for b_p in range(num_bands):
                w_p = w_hyb[p_idx, b_p]
                dE = w_q - w_k - w_p

                # Adaptive broadening: sigma tracks how much dE actually varies
                # across one grid cell, so the delta resolution follows the local
                # band steepness instead of being one constant across the band.
                # min_sigma acts as a floor for flat regions.
                variance = 0.0
                for i in range(3):
                    d_g = grad_f_hyb[p_idx, b_p, i] - grad_f_hyb[k_idx, b_k, i]
                    step_width = d_g / mesh[i]
                    variance += step_width * step_width

                sigma_raw = base_smearing * math.sqrt(variance / 12.0)
                sigma = sigma_raw if sigma_raw > min_sigma else min_sigma
                cutoff = 2.0 * sigma

                if (abs(dE) < cutoff
                        and w_q >= MIN_SCATTERING_ENERGY_MEV
                        and w_k >= MIN_SCATTERING_ENERGY_MEV
                        and w_p >= MIN_SCATTERING_ENERGY_MEV):
                    gaussian_norm = 0.4179 / sigma
                    delta_weight = gaussian_norm * math.exp(-0.5 * (dE * dE) / (sigma * sigma))

                    V_sq = calc_symmetrized_hybrid_vertex_squared_path(
                        path_idx, minus_path_grid_idx,
                        k_idx, p_idx, minus_k_idx, minus_p_idx,
                        b_q, b_k, b_p,
                        Qmatrix, path_eig_hyb,
                        V1_cache_kq, V2_cache_kq, V3_cache_kq,
                        V1_cache_qk, V2_cache_qk, V3_cache_qk,
                        num_phon, num_mag
                    )

                    c_idx = cuda.atomic.add(channel_count, 0, 1)
                    if c_idx < chan_indices.shape[1]:
                        chan_indices[0, c_idx] = 0
                        chan_indices[1, c_idx] = path_idx
                        chan_indices[2, c_idx] = k_idx
                        chan_indices[3, c_idx] = p_idx
                        chan_indices[4, c_idx] = b_q
                        chan_indices[5, c_idx] = b_k
                        chan_indices[6, c_idx] = b_p
                        chan_weights[c_idx] = V_sq * delta_weight

    # ---------------------------------------------------------------------
    # COALESCENCE channel: hybrid(path, b_q) + hybrid(k, b_k) -> hybrid(s, b_s).
    # Determines the RTA term with weight (n_k - n_s). 
    # The path mode is a
    # CHILD here; the splitting scan above can never reach this topology because
    # there the path point is structurally always the parent.
    #
    # The caches are refilled in place for the new (parent, child, child)
    # assignment - reusing the same six local arrays rather than declaring six
    # more keeps per-thread local memory flat.
    # ---------------------------------------------------------------------
    fill_hybrid_vertex_cache(
        path_q_cart[path_idx, 0], path_q_cart[path_idx, 1], path_q_cart[path_idx, 2],
        kx, ky, kz, k_idx,
        -sx_cart, -sy_cart, -sz_cart, minus_s_idx,
        grid_map, slc_axis, slc_rij, slc_rik, slc_J, slc_types,
        eig_phon, w_phon, atom_masses, mag_moments, num_phon, num_mag,
        V1_cache_kq, V2_cache_kq, V3_cache_kq
    )
    fill_hybrid_vertex_cache_pathq(
        kx, ky, kz,
        path_q_cart[path_idx, 0], path_q_cart[path_idx, 1], path_q_cart[path_idx, 2], path_idx,
        -sx_cart, -sy_cart, -sz_cart, minus_s_idx,
        grid_map, slc_axis, slc_rij, slc_rik, slc_J, slc_types,
        eig_phon, w_phon, path_eig_phon, path_w_phon,
        atom_masses, mag_moments, num_phon, num_mag,
        V1_cache_qk, V2_cache_qk, V3_cache_qk
    )

    for b_q in range(num_bands):
        w_q = path_w_hyb[path_idx, b_q]
        for b_k in range(num_bands):
            w_k = w_hyb[k_idx, b_k]
            for b_s in range(num_bands):
                w_s = w_hyb[s_idx, b_s]
                dE = w_s - w_q - w_k

                # Same adaptive broadening; here the parent is s and the two
                # children are the path mode and k, matching dE's sign pattern.
                variance = 0.0
                for i in range(3):
                    d_g = grad_f_hyb[s_idx, b_s, i] - grad_f_hyb[k_idx, b_k, i]
                    step_width = d_g / mesh[i]
                    variance += step_width * step_width

                sigma_raw = base_smearing * math.sqrt(variance / 12.0)
                sigma = sigma_raw if sigma_raw > min_sigma else min_sigma
                cutoff = 2.0 * sigma

                if (abs(dE) < cutoff
                        and w_q >= MIN_SCATTERING_ENERGY_MEV
                        and w_k >= MIN_SCATTERING_ENERGY_MEV
                        and w_s >= MIN_SCATTERING_ENERGY_MEV):
                    gaussian_norm = 0.4179 / sigma
                    delta_weight = gaussian_norm * math.exp(-0.5 * (dE * dE) / (sigma * sigma))

                    V_sq = calc_symmetrized_hybrid_vertex_squared_path_coal(
                        path_idx, minus_path_grid_idx,
                        s_idx, minus_s_idx,
                        k_idx, minus_k_idx,
                        b_s, b_q, b_k,
                        Qmatrix, path_eig_hyb,
                        V1_cache_kq, V2_cache_kq, V3_cache_kq,
                        V1_cache_qk, V2_cache_qk, V3_cache_qk,
                        num_phon, num_mag
                    )

                    c_idx = cuda.atomic.add(channel_count, 0, 1)
                    if c_idx < chan_indices.shape[1]:
                        chan_indices[0, c_idx] = 1
                        chan_indices[1, c_idx] = path_idx
                        chan_indices[2, c_idx] = k_idx
                        chan_indices[3, c_idx] = s_idx
                        chan_indices[4, c_idx] = b_q
                        chan_indices[5, c_idx] = b_k
                        chan_indices[6, c_idx] = b_s
                        chan_weights[c_idx] = V_sq * delta_weight


@cuda.jit
def phase_lifetime_hybrid_path(chan_indices, chan_weights, num_channels, n_hyb_grid, gamma_hyb_path, N_grid_points):
    """
    Computes SMRTA relaxation rates for hybrid modes along the explicit path, using
    grid-evaluated occupations for the grid-bound partner modes. Accumulates BOTH
    RTA terms onto the path mode:

      c_type 0 (splitting,   path -> k + p): pi/hbar   * |Gamma~|^2 * (1 + n_k + n_p)
      c_type 1 (coalescence, path + k -> s): 2*pi/hbar * |Gamma~|^2 * (n_k - n_s)

    Only the path mode's own rate is written - the grid partners' lifetimes are
    already accounted for by the grid hybrid pass.
    """
    idx = cuda.grid(1)
    if idx >= num_channels[0] or idx >= chan_weights.shape[0]:
        return

    c_type   = chan_indices[0, idx]
    path_idx = chan_indices[1, idx]
    k_idx = chan_indices[2, idx]
    o_idx = chan_indices[3, idx]   # p (splitting) or s (coalescence)
    b_q   = chan_indices[4, idx]
    b_k   = chan_indices[5, idx]
    b_o   = chan_indices[6, idx]
    V_sq = chan_weights[idx]

    hbar = 0.6582119569  # meV * ps
    prefactor_split = (math.pi / hbar) / N_grid_points
    prefactor_coal  = (math.pi / hbar) / N_grid_points 


    # as far as I can see: in the case of bcc Fe only coalesence gives phonon lifetimes

    n_k = n_hyb_grid[k_idx, b_k]
    n_o = n_hyb_grid[o_idx, b_o]

    num_bands = n_hyb_grid.shape[1]

    if c_type == 0:
        rate_q = prefactor_split * V_sq * (n_k + n_o + 1.0)
    if c_type == 1:
        rate_q = prefactor_coal * V_sq * (n_k - n_o)

    cuda.atomic.add(gamma_hyb_path, path_idx * num_bands + b_q, rate_q)


@cuda.jit
def compute_G_mp_kernel(chan_indices, chan_weights, d_channel_count, w_mag, w_phon, temperature, G_mp_out, N_points):
    """
    Calculates the 3TM spin-lattice coupling constant G_mp by summing over
    the magnon emission phase space (c_type == 0).
    """
    idx = cuda.grid(1)
    if idx >= d_channel_count[0] or idx >= chan_weights.shape[0]:
        return

    # The G_mp formula restricts the sum to the magnon emission channel
    c_type = chan_indices[0, idx]
    if c_type != 0:
        return

    q_idx = chan_indices[1, idx]  # Initial Magnon (parent)
    k_idx = chan_indices[2, idx]  # Final Magnon (child)
    p_idx = chan_indices[3, idx]  # Emitted Phonon (child)
    n     = chan_indices[4, idx]
    m     = chan_indices[5, idx]
    lam   = chan_indices[6, idx]

    V_sq_delta = chan_weights[idx]  # Contains |V|^2 * delta(E)

    E_init = w_mag[q_idx, n]
    E_fin  = w_mag[k_idx, m]
    E_phon = w_phon[p_idx, lam]

    kB = 0.08617333262 # meV/K
    if temperature <= 0.0:
        return

    beta = 1.0 / (kB * temperature)

    # Filter zero-energy modes (acoustic modes at Gamma) to prevent infinities
    if E_init < 1e-5 or E_fin < 1e-5 or E_phon < 1e-5:
        return

    # Equilibrium Bose-Einstein distributions
    n_mag_init = 1.0 / (math.exp(E_init * beta) - 1.0)
    n_mag_fin  = 1.0 / (math.exp(E_fin * beta) - 1.0)
    n_phon     = 1.0 / (math.exp(E_phon * beta) - 1.0)

    hbar = 0.6582119569 # meV * ps
    prefactor = (2.0 * math.pi / hbar) / (N_points * N_points)

    # Summand: (n_ph + 1)(n_mag_fin + 1)(n_mag_init) * (E_ph^2) / (k_B * T^2)
    term = prefactor * V_sq_delta * (n_phon + 1.0) * (n_mag_fin + 1.0) * n_mag_init  * (E_phon * E_phon) * (beta / temperature)

    cuda.atomic.add(G_mp_out, 0, term)

def calc_heat_capacity_per_cell(w_grid, temperature):
    """
    Volumetric (per primitive-cell) heat capacity from a boson mode spectrum
    w_grid[q_idx, branch] (meV), evaluated at `temperature` (K).
    """
    kB = 0.08617333262  # meV/K
    if temperature <= 0.0:
        return 0.0

    N_q = w_grid.shape[0]

    # 1. Mask out non-physical modes and the Gamma point
    # A tiny threshold (1e-4 meV ~ 0.02 THz) filters out the acoustic Goldstone modes
    # at q=0 and completely ignores any imaginary (negative) artifacts.
    valid_mask = w_grid > 1e-4
    
    # Initialize the output array with zeros
    c_mode = np.zeros_like(w_grid)
    
    # Calculate x only for valid, positive-frequency modes
    x_valid = w_grid[valid_mask] / (kB * temperature)
    
    # 2. Prevent floating-point overflow for deeply frozen optical modes at low T
    # e^(-150) is ~10^-66, which is zero in float64. 
    # Attempting to sinh(150/2) is fine, but beyond x ~ 1400 it overflows to inf.
    compute_mask = x_valid < 150.0
    x_compute = x_valid[compute_mask]
    
    # Apply the stable hyperbolic sine identity
    ratio = x_compute / (2.0 * np.sinh(x_compute / 2.0))
    
    # Map the computed values back to the original array shape
    c_mode_valid = np.zeros_like(x_valid)
    c_mode_valid[compute_mask] = kB * (ratio * ratio)
    c_mode[valid_mask] = c_mode_valid
    
    return np.sum(c_mode) / N_q


def calculate_and_save_Gmp_vs_T(chan_indices_active, chan_weights_active, d_channel_count, num_channels,
                                 d_w_mag, d_w_phon, w_mag_host, w_phon_host, N_points,
                                 filename="Outputs/G_mp_temperature_scan.csv"):
    """
    Evaluates the G_mp(T) kernel, the magnon/lattice heat capacities, and the
    resulting spin-lattice relaxation time tau(T) over a temperature grid, and
    writes them to a CSV.

    Units (all "per primitive unit cell", consistent throughout):
        G_mp : meV / (K * ps)   (already the native output of compute_G_mp_kernel)
        C_s, C_l : meV / K      (magnon / phonon heat capacity, see calc_heat_capacity_per_cell)
        tau = [G_mp * (1/C_s + 1/C_l)]^-1
            units: [meV/(K*ps)] / [meV/K] = 1/ps  =>  tau in ps
    """
    print(f"\nCalculating temperature-dependent G_mp, heat capacities, and tau, saving to {filename}...")
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    d_G_mp_out = cuda.device_array(1, dtype=np.float64)

    threads_per_block = 256
    blocks_eval = math.ceil(num_channels / threads_per_block)

    temperatures = [1.0 , 4.0, 10.0, 20.0, 30.0, 50.0, 300.0, 500.0, 600.0, 800.0]

    with open(filename, 'w') as f:
        f.write("Temperature_K,G_mp_meV_per_K_ps_per_cell,C_s_meV_per_K_per_cell,tau_ps\n")

        for T in temperatures:
            d_G_mp_out.copy_to_device(np.zeros(1, dtype=np.float64))

            compute_G_mp_kernel[blocks_eval, threads_per_block](
                chan_indices_active,
                chan_weights_active,
                d_channel_count,
                d_w_mag,
                d_w_phon,
                T,
                d_G_mp_out,
                N_points
            )
            cuda.synchronize()

            G_mp_val = d_G_mp_out.copy_to_host()[0]

            C_s_val = calc_heat_capacity_per_cell(w_mag_host, T)

            if G_mp_val > 0.0 and C_s_val > 0.0:
                tau_val = 1.0 / (G_mp_val * (1.0 / C_s_val ))
            else:
                tau_val = float('inf')

            f.write(f"{T:.2f},{G_mp_val:.6e},{C_s_val:.6e},{tau_val:.6e}\n")

    print(f"-> G_mp(T), C_s(T), and tau(T) calculation complete!")


def init_bose_einstein(w_distribution, temperature_K):
    if temperature_K <= 0.0:
        return np.zeros_like(w_distribution, dtype=np.float64)
        
    kB = 0.08617333262  # meV/K
    
    with np.errstate(divide='ignore', invalid='ignore'):
        occ = 1.0 / (np.exp(w_distribution / (kB * temperature_K)) - 1.0)
        
    occ[~np.isfinite(occ)] = 0.0
    return occ


"""
@cuda.jit
def phase_1_scan(mesh, q_grid, q_grid_cart, grid_map, w_phon, w_mag, eig_phon, slc_axis, slc_rij, slc_rik, slc_J, slc_types, smearing, chan_indices, chan_weights, channel_count, atom_masses, mag_moments, gamma_idx):
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
"""

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
    
    # ---------------------------------------------------------
    # UNIFIED UPDATE: M(q) <--> M(k) + Ph(p)
    # ---------------------------------------------------------
    nk_mag = n_mag[k_idx, m]
    nq_mag = n_mag[q_idx, n]
    np_phon = n_phon[p_idx, lam]
    
    # rate_q represents dn_q / dt. 
    # If positive, M(q) is created. If negative, M(q) is destroyed.
    rate_q = fgr_prefactor * V_sq * ((nq_mag + 1.0) * nk_mag * np_phon - nq_mag * (nk_mag + 1.0) * (np_phon + 1.0))

    # Apply identically symmetric changes to all three particles
    cuda.atomic.add(dn_mag, q_idx * num_mag_branches + n, rate_q)
    cuda.atomic.add(dn_mag, k_idx * num_mag_branches + m, -rate_q)
    cuda.atomic.add(dn_phon, p_idx * num_phon_branches + lam, -rate_q)
    
    """
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
    """

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
    fgr_prefactor = (2.0 * math.pi / hbar) / N_grid_points
    
    num_mag_branches = n_mag_grid.shape[1]
    num_phon_branches = n_phon_grid.shape[1]
    
    if c_type == 0:
        # Unified magnon channel: full bidirectional bracket (mirrors phase_lifetime's gamma_q)
        nk_mag = n_mag_grid[k_idx, m]
        n_ph = n_phon_grid[p_idx, lam]
        rate = fgr_prefactor * V_sq * (1.0 + n_ph + nk_mag)
        cuda.atomic.add(gamma_mag_path, path_idx * num_mag_branches + n, rate)
        
    elif c_type == 2:
        # Unified phonon channel (mirrors the confirmed phonon RTA rate)
        nk_mag = n_mag_grid[k_idx, n]
        n_kminq_mag = n_mag_grid[p_idx, m]
        rate = fgr_prefactor * V_sq * (n_kminq_mag - nk_mag)
        cuda.atomic.add(gamma_phon_path, path_idx * num_phon_branches + lam, rate)


    """

    # Channel 1
    # dE = w_mag[q_idx, n] - w_mag[k_idx, m] - w_phon[idx_qmink, lam]
    
    if c_type == 0:
        nk_mag = n_mag[k_idx, m]
        np_phon = n_phon[p_idx, lam]
        
        gamma_q = fgr_prefactor * V_sq * (1.0 + np_phon + nk_mag)
        cuda.atomic.add(gamma_mag, q_idx * num_mag_branches + n, gamma_q)

    # Phase 2
    #dE = w_mag[q_idx, n] - w_mag[k_idx, m] + w_phon[idx_kminq, lam]

    if c_type == 1:
        nk_mag = n_mag[k_idx, m]
        np_phon = n_phon[p_idx, lam]
        
        gamma_q = fgr_prefactor * V_sq * (np_phon - nk_mag)
        cuda.atomic.add(gamma_mag, q_idx * num_mag_branches + n, gamma_q)
    
    # Phase 3
    #dE = w_mag[idx_kminq, n] - w_mag[k_idx, m] + w_phon[q_idx, lam]
    if c_type == 2:
        nk_mag = n_mag[k_idx, m]
        np_mag = n_mag[p_idx, n]
        
        gamma_q = fgr_prefactor * V_sq * (np_mag - nk_mag)
        cuda.atomic.add(gamma_phon, q_idx * num_phon_branches + lam, gamma_q)


    """




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
    idx = cuda.grid(1)
    
    if idx < dn_mag.shape[0]:
        q_idx = idx // n_mag.shape[1]
        b = idx % n_mag.shape[1]
        
        new_n = n_mag[q_idx, b] + dn_mag[idx] * dt
        n_mag[q_idx, b] = new_n if new_n > 1e-15 else 1e-15
        dn_mag[idx] = 0.0  
        
    if idx < dn_phon.shape[0]:
        q_idx = idx // n_phon.shape[1]
        b = idx % n_phon.shape[1]
        
        new_n = n_phon[q_idx, b] + dn_phon[idx] * dt
        n_phon[q_idx, b] = new_n if new_n > 1e-15 else 1e-15
        dn_phon[idx] = 0.0
        



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

    # --- CLI Argument Parsing ---
    parser = argparse.ArgumentParser(description="SpinPhony Boltzmann Transport Simulation")
    parser.add_argument("--config", type=str, default="config.toml", help="Path to the TOML configuration file.")
    parser.add_argument("--material", type=str, default="bccFe", help="Target material defined in the config file.")
    parser.add_argument("--out_dir", type=str, default="Outputs", help="Directory to write output files into (created if missing).")

    # Optional overrides 
    parser.add_argument("--smearing", type=float, help="Override default smearing (meV)")
    parser.add_argument("--min_sigma", type=float, help="Override default minimum smearing (meV)") # NEW
    parser.add_argument("--tmag", type=float, help="Override initial Magnon temperature (K)")
    parser.add_argument("--tphon", type=float, help="Override initial Phonon temperature (K)")
    parser.add_argument("--steps", type=int, help="Override total integration steps")
    parser.add_argument("--full_bz", action="store_true",help="Evaluate hybrid lifetimes over the whole Brillouin zone.")
    
    args = parser.parse_args()

    # --- Load Configuration ---
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file '{args.config}' not found.")
        
    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    if args.material not in config["materials"]:
        raise ValueError(f"Material '{args.material}' not found in {args.config}. Available: {list(config['materials'].keys())}")

    sim_config = config["simulation"]
    mat_config = config["materials"][args.material]

    # --- Apply Settings and Overrides ---
    smearing = args.smearing if args.smearing is not None else sim_config["smearing"]
    min_sigma = args.min_sigma if args.min_sigma is not None else sim_config.get("min_sigma", 0.5)
    T_mag_init = args.tmag if args.tmag is not None else sim_config["T_mag_init"]
    T_phon_init = args.tphon if args.tphon is not None else sim_config["T_phon_init"]
    steps = args.steps if args.steps is not None else int(sim_config["steps"])
    
    anticipated_fraction = sim_config["anticipated_fraction"]
    dt = sim_config["dt"]

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    lattice_constant = mat_config["lattice_constant"]
    anisotropy = mat_config["anisotropy"]
    mesh = mat_config["mesh"]
    Jijs = mat_config["jij"]
    band = mat_config["band"]
    slc_files = mat_config["slc_files"]

    print(f"==================================================")
    print(f" Initializing Run: {args.material}")
    print(f"==================================================")
    print(f" Smearing  : {smearing} meV")
    print(f" Temp Mag  : {T_mag_init} K")
    print(f" Temp Phon : {T_phon_init} K")
    print(f" Steps     : {steps:,}")
    print(f"==================================================")

    crystal_data = CrystalDataSoA(
        mesh, 
        Jijs,
        slc_files=slc_files,
        lattice_constant=lattice_constant,
        anisotropy=anisotropy,
    )
    
    crystal_data.print_summary()

    gpu_data = crystal_data.push_to_gpu()

    gamma_idx = int(crystal_data.grid_map[0, 0, 0])

    if args.full_bz:
        crystal_data.use_full_bz_as_path()
    else:
        # Load the high-symmetry path from the HDF5 band file
        crystal_data.load_and_evaluate_path_hdf5(band, K_anisotropy=anisotropy, lattice_constant=lattice_constant)
        crystal_data.save_path_dispersions(f"{out_dir}/path_dispersions.csv")
        crystal_data.plot_path_dispersions(f"{out_dir}/exact_path_dispersions.png")

        print("\nEvaluating magnon-polaron hybridization along the high-symmetry path...")
        crystal_data.path_w_hyb, crystal_data.path_eig_hyb = crystal_data._calculate_coupled_hamiltonian(
            q_cart_array=crystal_data.path_q_cart,
            dyn_mat=crystal_data.path_dyn_mat,
            K_anisotropy=anisotropy,
            lattice_constant=lattice_constant
        )

        crystal_data.plot_hybridized_path_dispersions(f"{out_dir}/hybridized_character.png", color_mode='character')
        crystal_data.plot_hybridized_path_dispersions(f"{out_dir}/hybridized_spin_AM.png", color_mode='spin_am')
        crystal_data.plot_hybridized_path_dispersions(f"{out_dir}/hybridized_phon_AM.png", color_mode='phon_am')

    crystal_data.save_hybrid_path_properties(f"{out_dir}/hybrid_path_properties.csv")


    # Push path data to GPU for scanning kernels
    d_path_q_frac = cuda.to_device(crystal_data.path_q_frac)
    d_path_q_cart = cuda.to_device(crystal_data.path_q_cart)
    d_path_w_phon = cuda.to_device(crystal_data.path_w_phon)
    d_path_w_mag = cuda.to_device(crystal_data.path_w_mag)
    d_path_eig_phon = cuda.to_device(crystal_data.path_eig_phon)
    d_path_eig_mag = cuda.to_device(crystal_data.path_eig_mag)
    d_path_eig_hyb = cuda.to_device(crystal_data.path_eig_hyb)
    d_path_w_hyb = cuda.to_device(crystal_data.path_w_hyb)

    # 2. Setup Phase 1 memory
    N_points = crystal_data.N

    threads_per_block = 256

    print(f"\nInitializing populations at thermal equilibrium:")
    print(f" -> Magnons: {T_mag_init} K")
    print(f" -> Phonons: {T_phon_init} K")

    # Generate population profiles matching the actual branch dispersions
    n_mag_cpu = init_bose_einstein(crystal_data.w_mag, T_mag_init)
    n_phon_cpu = init_bose_einstein(crystal_data.w_phon, T_phon_init)

    # Push initial states
    d_n_mag = cuda.to_device(n_mag_cpu)
    d_n_phon = cuda.to_device(n_phon_cpu)



    # =============== Hybrid Lifetimes ( Whole BZ) ===============
    """
    num_hyb_branches = crystal_data.phon_branches + crystal_data.n_mag_branches

    total_loops = N_points**2 * num_hyb_branches**3
    max_channels = int(total_loops * anticipated_fraction)
    
    # 7 Rows format exactly maps the generic triad coordinates
    d_chan_indices = cuda.device_array((7, max_channels), dtype=np.int32)
    d_chan_weights = cuda.device_array(max_channels, dtype=np.float64)
    d_channel_count = cuda.to_device(np.zeros(1, dtype=np.int64))

    print(f"\nInitializing global Bose-Einstein populations at {T_mag_init} K...")
    n_hyb_cpu = init_bose_einstein(crystal_data.w_hyb, T_mag_init)
    d_n_hyb = cuda.to_device(n_hyb_cpu)

    # 2. Execute Phase 1
    print("\nStarting Phase 1: Scanning Hybrid Phase Space and Computing Vertices...")

    threads_per_block_2d = (16, 16) 
    blocks_x = math.ceil(N_points / threads_per_block_2d[0])
    blocks_y = math.ceil(N_points / threads_per_block_2d[1])
    blocks_per_grid_2d = (blocks_x, blocks_y)

    phase_1_scan_hybrid[blocks_per_grid_2d, threads_per_block_2d](
        gpu_data["mesh"], 
        gpu_data["q_grid"], 
        gpu_data["q_grid_cart"],
        gpu_data["grid_map"], 
        gpu_data["w_hyb"], 
        gpu_data["grad_f_hyb"], 
        gpu_data["Qmatrix"],
        gpu_data["slc_axis"], 
        gpu_data["slc_rij"], 
        gpu_data["slc_rik"], 
        gpu_data["slc_J"], 
        gpu_data["slc_types"], 
        gpu_data["eig_phon"],
        gpu_data["w_phon"],
        gpu_data["atom_masses"], 
        gpu_data["mag_moments"],
        smearing,                 
        min_sigma,             
        d_chan_indices,   
        d_chan_weights, 
        d_channel_count,
        crystal_data.phon_branches,
        crystal_data.n_mag_branches
    )

    cuda.synchronize()
    
    num_channels = d_channel_count.copy_to_host()[0]
    print(f"Allowed Hybrid Channels found: {num_channels:,}")
    print(f" -> Phase space ratio captured: {num_channels / total_loops:.2%}")

    d_chan_indices_active = d_chan_indices[:, :num_channels]
    d_chan_weights_active = d_chan_weights[:num_channels]
    
    # 3. Lifetimes Evaluation Phase 
    print("\nCalculating equilibrium SMRTA relaxation rates...")
    d_gamma_hyb = cuda.to_device(np.zeros(N_points * num_hyb_branches, dtype=np.float64))

    threads_per_block = 256
    blocks_eval = math.ceil(num_channels / threads_per_block)


    phase_lifetime_hybrid[blocks_eval, threads_per_block](
        d_chan_indices_active,
        d_chan_weights_active,
        d_channel_count, 
        d_n_hyb,      
        d_gamma_hyb, 
        N_points
    )

    
    cuda.synchronize()

    # 4. Save Extracted Data
    gamma_hyb_cpu = d_gamma_hyb.copy_to_host().reshape((N_points, num_hyb_branches))

    phon_chars, mag_chars, phon_ams, spin_ams = crystal_data.extract_full_grid_hybrid_properties()

    os.makedirs("Outputs", exist_ok=True)
    out_file = f"{out_dir}/hybrid_equilibrium_lifetimes.csv"
    
    with open(out_file, "w") as f:
        # Appended headers
        f.write("q_idx,qx,qy,qz,branch,energy_meV,vx,vy,vz,phon_char,mag_char,phon_AM,spin_AM,gamma_ps-1,tau_ps\n")
        
        for q_idx in range(N_points):
            qx, qy, qz = crystal_data.q_grid[q_idx]
            for branch in range(num_hyb_branches):
                energy = crystal_data.w_hyb[q_idx, branch]
                gamma = gamma_hyb_cpu[q_idx, branch]
                tau = 1.0 / gamma if gamma > 1e-12 else float('inf')
                
                # Retrieve local characters and AM
                pc = phon_chars[q_idx, branch]
                mc = mag_chars[q_idx, branch]
                pa = phon_ams[q_idx, branch]
                sa = spin_ams[q_idx, branch]
                
                # Retrieve the hybridized group velocities
                vx = crystal_data.grad_f_hyb[q_idx, branch, 0]
                vy = crystal_data.grad_f_hyb[q_idx, branch, 1]
                vz = crystal_data.grad_f_hyb[q_idx, branch, 2]
                
                f.write(f"{q_idx},{qx},{qy},{qz},{branch},{energy:.6f},{vx:.6f},{vy:.6f},{vz:.6f},{pc:.6f},{mc:.6f},{pa:.6e},{sa:.6e},{gamma:.6e},{tau:.6e}\n")

    print(f"-> Saved equilibrium hybrid lifetimes, velocities, and characters to {out_file}.")
    """



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
        gpu_data["grad_f_phon"], gpu_data["grad_f_mag"],
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

    path_dist, label_comment = crystal_data.get_path_distance_info()

    with open(f"{out_dir}/path_lifetimes.csv", "w") as f:
        f.write(label_comment + "\n")
        f.write("q_idx,qx,qy,qz,particle,branch,energy_meV,gamma_ps-1,tau_ps,path_dist\n")

        # Magnons
        for i in range(N_path):
            qx, qy, qz = crystal_data.path_q_frac[i]
            for b in range(crystal_data.n_mag_branches):
                energy = crystal_data.path_w_mag[i, b]
                gamma = gamma_mag_path_cpu[i, b]
                tau = 1.0 / gamma if gamma > 1e-12 else float('inf')
                f.write(f"{i},{qx:.6f},{qy:.6f},{qz:.6f},magnon,{b},{energy:.6f},{gamma:.6e},{tau:.6e},{path_dist[i]:.6f}\n")

        # Phonons
        for i in range(N_path):
            qx, qy, qz = crystal_data.path_q_frac[i]
            for b in range(crystal_data.phon_branches):
                energy = crystal_data.path_w_phon[i, b]
                gamma = gamma_phon_path_cpu[i, b]
                tau = 1.0 / gamma if gamma > 1e-12 else float('inf')
                f.write(f"{i},{qx:.6f},{qy:.6f},{qz:.6f},phonon,{b},{energy:.6f},{gamma:.6e},{tau:.6e},{path_dist[i]:.6f}\n")
                
    print(f"-> Saved true path lifetimes to {out_dir}/path_lifetimes.csv")


    # ========================== Hybrid Path Lifetime Evaluation ==========================
    print("\nStarting Hybrid Path Lifetime Evaluation...")

    num_hyb_branches = crystal_data.phon_branches + crystal_data.n_mag_branches
    n_hyb_cpu = init_bose_einstein(crystal_data.w_hyb, T_mag_init)
    d_n_hyb = cuda.to_device(n_hyb_cpu)

    # phase_1_scan_hybrid_path now scans two scattering processes (coalesence and splitting) 
    #  so the enumerated loop count, and hence the channel buffer, doubles.
    total_path_hyb_loops = 2 * N_path * N_points * num_hyb_branches**3
    max_path_hyb_channels = int(total_path_hyb_loops * anticipated_fraction)

    d_path_hyb_chan_indices = cuda.device_array((7, max_path_hyb_channels), dtype=np.int32)
    d_path_hyb_chan_weights = cuda.device_array(max_path_hyb_channels, dtype=np.float64)
    d_path_hyb_channel_count = cuda.to_device(np.zeros(1, dtype=np.int64))

    print(" -> Scanning Hybrid Phase Space (Path x Grid)...")
    phase_1_scan_hybrid_path[(blocks_x_path, blocks_y_path), threads_2d_path](
        gpu_data["mesh"], gpu_data["q_grid"], d_grid_q_frac, gpu_data["q_grid_cart"], gpu_data["grid_map"],
        d_path_q_frac, d_path_q_cart, d_path_w_hyb, d_path_eig_hyb,
        gpu_data["w_hyb"], gpu_data["Qmatrix"],
        gpu_data["slc_axis"], gpu_data["slc_rij"], gpu_data["slc_rik"], gpu_data["slc_J"], gpu_data["slc_types"],
        gpu_data["eig_phon"], gpu_data["w_phon"], d_path_eig_phon, d_path_w_phon,
        gpu_data["grad_f_hyb"],
        gpu_data["atom_masses"], gpu_data["mag_moments"],
        smearing, min_sigma, d_path_hyb_chan_indices, d_path_hyb_chan_weights, d_path_hyb_channel_count,
        crystal_data.phon_branches, crystal_data.n_mag_branches
    )

    cuda.synchronize()

    path_hyb_num_channels = d_path_hyb_channel_count.copy_to_host()[0]
    print(f" -> Hybrid Path Channels found: {path_hyb_num_channels:,}")

    d_path_hyb_chan_indices_active = d_path_hyb_chan_indices[:, :path_hyb_num_channels]
    d_path_hyb_chan_weights_active = d_path_hyb_chan_weights[:path_hyb_num_channels]

    d_gamma_hyb_path = cuda.to_device(np.zeros(N_path * num_hyb_branches, dtype=np.float64))

    print(" -> Calculating explicit hybrid path lifetimes...")
    blocks_eval_path_hyb = math.ceil(path_hyb_num_channels / threads_per_block)

    phase_lifetime_hybrid_path[blocks_eval_path_hyb, threads_per_block](
        d_path_hyb_chan_indices_active,
        d_path_hyb_chan_weights_active,
        d_path_hyb_channel_count,
        d_n_hyb,
        d_gamma_hyb_path,
        N_points
    )
    cuda.synchronize()

    gamma_hyb_path_cpu = d_gamma_hyb_path.copy_to_host().reshape((N_path, num_hyb_branches))

    path_dist, label_comment = crystal_data.get_path_distance_info()

    with open(f"{out_dir}/hybrid_path_lifetimes.csv", "w") as f:
        f.write(label_comment + "\n")
        f.write("q_idx,qx,qy,qz,branch,energy_meV,gamma_ps-1,tau_ps,path_dist\n")
        for i in range(N_path):
            qx, qy, qz = crystal_data.path_q_frac[i]
            for b in range(num_hyb_branches):
                energy = crystal_data.path_w_hyb[i, b]
                gamma = gamma_hyb_path_cpu[i, b]
                tau = 1.0 / gamma if gamma > 1e-12 else float('inf')
                f.write(f"{i},{qx:.6f},{qy:.6f},{qz:.6f},{b},{energy:.6f},{gamma:.6e},{tau:.6e},{path_dist[i]:.6f}\n")

    print(f" -> Hybrid Path Channels: found_channels / max_channels : {path_hyb_num_channels / max_path_hyb_channels * 100:.2f}%")

    print(f"-> Saved hybrid path lifetimes to {out_dir}/hybrid_path_lifetimes.csv")

    

    # 3. Execute Phase 1
    print("\nStarting Phase 1: Scanning Phase Space and computing FT Vertices...")

    threads_per_block_2d = (16, 16) # 256 threads total per block
    blocks_x = math.ceil(N_points / threads_per_block_2d[0])
    blocks_y = math.ceil(N_points / threads_per_block_2d[1])
    blocks_per_grid_2d = (blocks_x, blocks_y)

    total_loops = N_points**2 * crystal_data.n_mag_branches**2 * crystal_data.phon_branches * 3
    max_channels = int(total_loops * anticipated_fraction)

    # --- SoA ALLOCATION --- (moved here, right before use: this O(N_points^2)
    # buffer is by far the largest allocation in the whole run and is only
    # needed for this scan - allocating it earlier held it resident on the
    # GPU throughout the unrelated path/hybrid-path work above.)
    # Shape is (7, max_channels) so the last axis is contiguous.
    # Row 0: c_type | Row 1: q_idx | Row 2: k_idx | Row 3: p_idx
    # Row 4: n      | Row 5: m     | Row 6: lam
    d_chan_indices = cuda.device_array((7, max_channels), dtype=np.int32)
    d_chan_weights = cuda.device_array(max_channels, dtype=np.float64)
    d_channel_count = cuda.to_device(np.zeros(1, dtype=np.int64))

    phase_1_scan[blocks_per_grid_2d, threads_per_block_2d](
        gpu_data["mesh"], 
        gpu_data["q_grid"], 
        gpu_data["q_grid_cart"],
        gpu_data["grid_map"], 
        gpu_data["w_phon"], 
        gpu_data["w_mag"], 
        gpu_data["eig_phon"],
        gpu_data["grad_f_phon"],  
        gpu_data["grad_f_mag"],   
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
        gamma_idx,
        min_sigma
    )

    # Wait for the GPU to finish the scan
    cuda.synchronize()
    
    # Extract the final count BEFORE slicing
    num_channels = d_channel_count.copy_to_host()[0]
    print(f"Allowed Channels found: {num_channels:,}", flush=True)
    print(f" -> Percentage of phase space allowed: {num_channels / total_loops:.2%}", flush=True)

    # Slice the device arrays so Phase 2 ONLY iterates over valid channels
    # Slicing along the 2nd axis preserves the C-contiguous layout
    d_chan_indices_active = d_chan_indices[:, :num_channels]
    d_chan_weights_active = d_chan_weights[:num_channels]
    
    blocks_eval = math.ceil(num_channels / threads_per_block)

    # ====== Lifetime and scattering rate phase ======

    d_dn_mag = cuda.to_device(np.zeros(N_points * crystal_data.n_mag_branches, dtype=np.float64))
    d_dn_phon = cuda.to_device(np.zeros(N_points * crystal_data.phon_branches, dtype=np.float64))

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
        d_n_mag,     
        d_n_phon,    
        d_gamma_mag, 
        d_gamma_phon, 
        N_points
    )
    cuda.synchronize()

    # Pull back and reshape
    gamma_mag_cpu = d_gamma_mag.copy_to_host().reshape((N_points, crystal_data.n_mag_branches))
    gamma_phon_cpu = d_gamma_phon.copy_to_host().reshape((N_points, crystal_data.phon_branches))

    # Write to CSV
    os.makedirs("Outputs", exist_ok=True)
    with open(f"{out_dir}/equilibrium_lifetimes.csv", "w") as f:
        f.write("q_idx,qx,qy,qz,particle,branch,energy_meV,vx,vy,vz,gamma_ps-1,tau_ps\n")
        # Write Magnon Lifetimes
        for q_idx in range(N_points):
            qx, qy, qz = crystal_data.q_grid[q_idx]
            for branch in range(crystal_data.n_mag_branches):
                energy = crystal_data.w_mag[q_idx, branch]
                gamma = gamma_mag_cpu[q_idx, branch]
                vx = crystal_data.grad_f_mag[q_idx, branch, 0]
                vy = crystal_data.grad_f_mag[q_idx, branch, 1]
                vz = crystal_data.grad_f_mag[q_idx, branch, 2]
                tau = 1.0 / gamma if gamma > 1e-12 else float('inf')
                f.write(f"{q_idx},{qx},{qy},{qz},magnon,{branch},{energy:.6f},{vx:.6f},{vy:.6f},{vz:.6f},{gamma:.6e},{tau:.6e}\n")
                
        # Write Phonon Lifetimes
        for q_idx in range(N_points):
            qx, qy, qz = crystal_data.q_grid[q_idx]
            for branch in range(crystal_data.phon_branches):
                energy = crystal_data.w_phon[q_idx, branch]
                gamma = gamma_phon_cpu[q_idx, branch]
                vx = crystal_data.grad_f_phon[q_idx, branch, 0]
                vy = crystal_data.grad_f_phon[q_idx, branch, 1]
                vz = crystal_data.grad_f_phon[q_idx, branch, 2]
                tau = 1.0 / gamma if gamma > 1e-12 else float('inf')
                f.write(f"{q_idx},{qx},{qy},{qz},phonon,{branch},{energy:.6f},{vx:.6f},{vy:.6f},{vz:.6f},{gamma:.6e},{tau:.6e}\n")

    print(f"-> Saved equilibrium lifetimes.", flush=True)



    # ========================== G_mp Temperature Scan ==========================
    # We pass the GPU arrays for w_mag/w_phon to prevent cross-memory-space crashes
    calculate_and_save_Gmp_vs_T(
        d_chan_indices_active,
        d_chan_weights_active,
        d_channel_count,
        num_channels,
        gpu_data["w_mag"],
        gpu_data["w_phon"],
        crystal_data.w_mag,
        crystal_data.w_phon,
        N_points,
        filename=f"{out_dir}/G_mp_temperature_scan.csv"
    )

    # Save
    np.savetxt(f"{out_dir}/w_mag_grid.csv", crystal_data.w_mag, delimiter=",")
    np.savetxt(f"{out_dir}/w_phon_grid.csv", crystal_data.w_phon, delimiter=",")

    # ========================== Time-evolution Phase ==========================
    
    # Setup temperatures
    T_mag_init = 500
    T_phon_init = 300
    
    print(f"\nInitializing populations for the dynamics:")
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
    
    # Grid sizes for both kernels
    blocks_eval = math.ceil(num_channels / threads_per_block)
    max_elements = max(N_points * crystal_data.n_mag_branches, N_points * crystal_data.phon_branches)
    blocks_euler = math.ceil(max_elements / threads_per_block)
    
    obs_file = open(f"{out_dir}/observables_dynamics.txt", "w")
    obs_file.write("Step\tTime_ps\tE_tot_meV\tE_mag_meV\tE_phon_meV\tN_mag\tN_phon\tT_eff_mag_K\tT_eff_phon_K\n")
    
    print(f"\nStarting Phase 2: Time Integration ({steps} steps)...")

    current_time = 0.0

    exit()
    pass

    for step in range(steps):
        
        # 1. Calculate derivatives (dn) on the GPU
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
        cuda.synchronize()

            
        # CPU Interaction & Correctly Placed Debug Block
        if step % 1000 == 0:
            n_mag_cpu = d_n_mag.copy_to_host()
            n_phon_cpu = d_n_phon.copy_to_host()
            compute_and_write_observables(
                step=step,
                current_time=current_time,
                n_mag=n_mag_cpu,      # Reuse the pulled array to save PCIe bandwidth
                n_phon=n_phon_cpu,
                w_mag=crystal_data.w_mag,   
                w_phon=crystal_data.w_phon,
                file_handle=obs_file
            )
            
            # DEBUG: Check Detailed Balance mathematically
            #net_rate = np.sum(dn_mag_cpu)
            #print(f"Step {step} | safe_dt: {safe_dt:.2e} ps | Net mag rate sum(dn): {net_rate:.6e}")
            #if abs(net_rate) > 1e-8:
            #    print(" -> WARNING: Detailed balance is broken in the collision physics!")

        # 4. Apply the safe_dt globally using the pure Euler kernel
        apply_euler_and_reset[blocks_euler, threads_per_block](
            d_n_mag, d_n_phon, d_dn_mag, d_dn_phon, dt
        )
        
        current_time += dt

    obs_file.close()
    print("Simulation Complete.")


