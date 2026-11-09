import numpy as np
import os
from numba import cuda
import cmath
import math
import h5py
import argparse
import sys
import mode_id

try:
    import tomllib
except ImportError:
    import tomli as tomllib


class CrystalDataSoA:
    def __init__(self, hdf5_filename, exchange_filename, slc_files=None, lattice_constant=1.0, anisotropy=0.01,
                 include_slc=True):
        # Master switch for the spin-lattice coupling blocks in the BdG
        # Hamiltonian. False leaves the magnon and phonon sectors uncoupled, so
        # the "hybrid" modes reduce to bare magnons and bare phonons. Read by
        # _calculate_coupled_hamiltonian for both the grid and path calls.
        self.include_slc = include_slc
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
        if not BARE_LEGACY:
            # dw/df by central differences of the sorted mesh energies (vectorised, O(N)); with the zero array the adaptive Gaussian width of every
            # magnon-phonon scan ignored the phonon dispersion (d_g = -grad_mag + grad_phon)
            for i_ax in range(3):
                shift = np.zeros(3, dtype=np.int64); shift[i_ax] = 1
                ip = self.grid_map[tuple(((self.q_grid + shift) % self.mesh).T)]
                im = self.grid_map[tuple(((self.q_grid - shift) % self.mesh).T)]
                self.grad_f_phon[:, :, i_ax] = (self.w_phon[ip] - self.w_phon[im]) * (self.mesh[i_ax] / 2.0)


        # Cartesian conversion
        q_frac_array = self.q_grid / self.mesh
        # Fold fractional coordinates strictly to [-0.5, 0.5) before Cartesian mapping
        q_frac_centered = q_frac_array - np.round(q_frac_array)
        self.q_grid_cart = np.dot(q_frac_centered, self.reciprocal_lattice * 2.0 * np.pi)
        
        # 4. Parse SLC Tensors
        if slc_files and len(slc_files) == 3:
            self._parse_slc_tensors(slc_files[0], slc_files[1], slc_files[2], lattice_constant)
            #self.enforce_slc_asr()

        #"""
        # Full matrices are necessary for hybrid group velocity calculations
        self.w_hyb, self.Qmatrix, self.H_BdG_pre_diagonalized, self.H_BdG_diagonalized = self._calculate_coupled_hamiltonian(
            q_cart_array=self.q_grid_cart,
            dyn_mat=self.dyn_mat_phon,
            K_anisotropy=anisotropy,
            lattice_constant=lattice_constant,
            return_full_matrices=True
        )
        #"""

        self.w_hyb, self.Qmatrix = self._calculate_coupled_hamiltonian(
            q_cart_array=self.q_grid_cart,
            dyn_mat=self.dyn_mat_phon,
            K_anisotropy=anisotropy,
            lattice_constant=lattice_constant,
            return_full_matrices=False
        )

        self.grad_f_hyb = np.zeros((self.N, self.phon_branches + self.n_mag_branches, 3), dtype=np.float64)
        self._compute_hybrid_group_velocities()


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

    def enforce_slc_asr(self, mode="proportional", verbose=True):
        """
        Imposes the acoustic sum rule sum_k J^{ab,mu}_{ij,k} = 0 on the parsed
        spin-lattice tensor, the same remedy used for truncated force constants.

        A rigid translation of the crystal cannot change the exchange between
        i and j, so the derivative summed over all displaced atoms must vanish.
        A real-space cutoff breaks this, leaving a residual that makes
        J~(x, q->0) tend to a constant instead of to zero - which turns
        |V^{+-}|^2 ~ |J~|^2 / omega into a 1/omega divergence and lets the
        near-Gamma region dominate the whole scattering sum.

        mode = "self":
            Subtract the whole residual from the group's r_ik = 0 entry. This
            is the usual force-constant remedy, but note that entry's Fourier
            phase is 1 for EVERY q, so the correction shifts J~(x, q) by a
            q-independent constant at all momenta - a blunt change when the
            residual is a large fraction of the tensor.

        mode = "proportional" (default):
            Spread the residual over the group weighted by |J| of each entry.
            The correction then carries the group's own phase structure and
            perturbs each term by the same relative amount, rather than
            concentrating a large error on one physically meaningful term.

        Neither recovers the missing data: a large residual means the
        real-space sum over displaced atoms was truncated too early, and the
        only real fix is extending that cutoff upstream. Enforcement removes
        the unphysical 1/omega divergence but redistributes the error.

        Must be called before push_to_gpu(), and before the coupled Hamiltonian
        is built if the SLC hybridization blocks are enabled.
        """
        rij_r = np.round(self.slc_rij, 6)
        rik_r = np.round(self.slc_rik, 6)

        groups = {}
        for idx in range(self.slc_axis.shape[0]):
            key = (int(self.slc_axis[idx]), int(self.slc_types[idx, 0]),
                   int(self.slc_types[idx, 1]), tuple(rij_r[idx]))
            groups.setdefault(key, []).append(idx)

        scale = np.abs(self.slc_J).max()
        worst_before = 0.0
        n_no_self = 0

        for idxs in groups.values():
            idxs = np.asarray(idxs)
            residual = self.slc_J[idxs].sum(axis=0)
            worst_before = max(worst_before, np.abs(residual).max())

            if mode == "proportional":
                w = np.abs(self.slc_J[idxs]).reshape(len(idxs), -1).max(axis=1)
                total = w.sum()
                if total <= 0:
                    continue
                self.slc_J[idxs] -= residual[None, :, :] * (w / total)[:, None, None]
            else:
                norms = np.linalg.norm(rik_r[idxs], axis=1)
                j = int(np.argmin(norms))
                if norms[j] > 1e-6:
                    n_no_self += 1  # no exact r_ik = 0 term; use the closest one
                self.slc_J[idxs[j]] -= residual

        worst_after = max(np.abs(self.slc_J[np.asarray(v)].sum(axis=0)).max()
                          for v in groups.values())

        if verbose:
            print(f"\n[ASR] enforced over {len(groups)} (mu, n, m, r_ij) groups")
            print(f"[ASR] max |sum_k J|: {worst_before:.4e} -> {worst_after:.4e} "
                  f"({worst_before / scale * 100:.2f}% -> {worst_after / scale * 100:.2e}% of max|J|)")
            if n_no_self:
                print(f"      note: {n_no_self} group(s) had no r_ik = 0 entry; the "
                      "correction went to the nearest displaced atom instead.")

    def check_slc_asr(self, mu=0, n_type=1, m_type=1, l_type=1,
                      x_cart=(0.3, 0.0, 0.0), n_pts=6):
        """
        Tests the soft-phonon limit of the magnon-phonon vertex.

        Translational invariance requires that a rigid displacement of the whole
        crystal leave the exchange energy unchanged, i.e. sum_k J^{ab,mu}_{ij,k} = 0
        for every spin pair (i,j). In Fourier space that means

            J~^{ab,mu}_{nml}(x, q)  ->  0   linearly in q,

        which is what makes |V^{+-}|^2 ~ |J~|^2 / omega vanish as q->0 rather than
        diverge. If it holds, results are insensitive to the `omega < ...` guard
        in calc_vertex_V; if it fails, the near-Gamma region dominates the whole
        scattering sum and the guard silently sets the answer.

        This is a plain NumPy replica of calc_fourier_transform_vec operating on
        the same slc_* arrays the CUDA kernel reads, so it cannot disagree with
        what the vertex actually computes.
        """
        sel = ((self.slc_axis == mu)
               & (self.slc_types[:, 0] == n_type)
               & (self.slc_types[:, 1] == m_type)
               & (self.slc_types[:, 2] == l_type))

        J, rij, rik = self.slc_J[sel], self.slc_rij[sel], self.slc_rik[sel]
        if J.shape[0] == 0:
            print(f"[ASR] no SLC entries for (mu={mu}, n={n_type}, m={m_type}, l={l_type})")
            return None

        scale = np.abs(J).max()
        print(f"\n[ASR] mu={mu} types=({n_type},{m_type},{l_type}): "
              f"{J.shape[0]} entries, max|J| = {scale:.4e}")

        # --- 1. Raw sum rule, grouped by spin-pair vector r_ij --------------
        # J~(x, 0) vanishes for EVERY x only if the sum over the displaced atom
        # vanishes separately within each r_ij group.
        worst, worst_r = 0.0, None
        keys = np.round(rij, 6)
        for r in np.unique(keys, axis=0):
            grp = np.all(keys == r, axis=1)
            v = np.abs(J[grp].sum(axis=0)).max()
            if v > worst:
                worst, worst_r = v, r
        print(f"[ASR] max |sum_k J| over r_ij groups = {worst:.4e}"
              f"  ({worst / scale * 100:.2f}% of max|J|)  at r_ij = {worst_r}")

        # --- 2. Direct q -> 0 scaling of J~ ---------------------------------
        b1 = np.dot(np.array([1.0, 0.0, 0.0]), self.reciprocal_lattice) * 2.0 * np.pi
        x = np.asarray(x_cart, dtype=np.float64)
        phase_x = rij @ x

        qs, mags = [], []
        for i in range(n_pts):
            q = b1 * (0.25 * 0.5 ** i)
            Jt = (J * np.exp(1j * (phase_x + rik @ q))[:, None, None]).sum(axis=0)
            qs.append(np.linalg.norm(q))
            mags.append(np.abs(Jt).max())
            print(f"      |q| = {qs[-1]:.5f} 1/A   max|J~| = {mags[-1]:.4e}")

        # The verdict rests on the REAL-SPACE sum, which is exact and
        # unambiguous. A log-log slope over the q-scan is unreliable: the
        # large-q points dominate any fit and say nothing about the q->0
        # limit, and J~ can pass accidentally close to zero at isolated q.
        mags = np.array(mags)
        monotonic = bool(np.all(np.diff(mags) < 0))
        rel = worst / scale

        print(f"[ASR] |J~| decreasing monotonically toward q=0: {monotonic}")
        if rel < 0.01 and monotonic:
            print("      -> OK: sum_k J cancels and J~ vanishes with q. |V|^2 ~ q,")
            print("         soft phonons are harmless and the omega cutoff is inert.")
        else:
            print(f"      -> BROKEN: sum_k J leaves {rel * 100:.1f}% of the tensor scale,")
            print("         so J~ tends to a constant as q->0 and |V|^2 ~ 1/omega.")
            print("         The near-Gamma region then dominates the scattering sum and")
            print("         the omega cutoff is masking that, not fixing it.")
            print("         Call enforce_slc_asr() before push_to_gpu() and re-check.")

        return rel

    

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

        PHONON_FLOOR_MEV = 0.01

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

        # =====================================================================
        # 1.5 Precalculate SLC Tensors (FM Case) in the BARE PHONON BRANCH basis
        #     Vectorised: one exp per UNIQUE r_ik + one BLAS matmul, instead of
        #     a full sweep over all N_pts q-points per SLC entry.
        #     MUST stay outside the q-loop - it is computed for all q at once.
        # =====================================================================
        V_plus_all = np.zeros((N_pts, num_mag, num_phon), dtype=np.complex128)
        V_minus_all = np.zeros((N_pts, num_mag, num_phon), dtype=np.complex128)

        if hasattr(self, 'slc_axis') and is_FM and self.include_slc:
            # Gather valid entries once; fold the mass weighting in here, since
            # 1/sqrt(M_l) depends only on the Cartesian index p = 3*l + mu and
            # must be applied BEFORE the rotation mixes p.
            rows, cols, cp, cm = [], [], [], []
            for i in range(self.slc_axis.shape[0]):
                n_mag_i = atom_to_mag[self.slc_types[i, 0] - 1]
                if n_mag_i == -1 or atom_to_mag[self.slc_types[i, 1] - 1] == -1:
                    continue
                disp_atom = self.slc_types[i, 2] - 1
                p_idx = 3 * disp_atom + self.slc_axis[i]
                inv_sqrt_m = 1.0 / np.sqrt(self.atom_masses[disp_atom] * DALTON_TO_meV_PS2_PER_A2)

                Jxz, Jyz = self.slc_J[i, 0, 2], self.slc_J[i, 1, 2]
                rows.append(i)
                cols.append(n_mag_i * num_phon + p_idx)
                cp.append((Jxz + 1j * Jyz) * inv_sqrt_m)
                cm.append((Jxz - 1j * Jyz) * inv_sqrt_m)

            if rows:
                rows = np.asarray(rows)
                cols = np.asarray(cols)

                # (a) one exp per UNIQUE displacement vector
                rik_u, inv = np.unique(np.round(self.slc_rik[rows], 8),
                                       axis=0, return_inverse=True)
                pf_u = np.exp(1j * (q_cart_array @ rik_u.T))        # (N_pts, n_uniq)

                n_cols = num_mag * num_phon
                C_p = np.zeros((rik_u.shape[0], n_cols), dtype=np.complex128)
                C_m = np.zeros((rik_u.shape[0], n_cols), dtype=np.complex128)
                np.add.at(C_p, (inv, cols), np.asarray(cp))
                np.add.at(C_m, (inv, cols), np.asarray(cm))

                # Cartesian, mass-weighted - whole accumulation is one matmul
                V_plus_all = (pf_u @ C_p).reshape(N_pts, num_mag, num_phon)
                V_minus_all = (pf_u @ C_m).reshape(N_pts, num_mag, num_phon)

                # (b) Cartesian -> bare phonon branch basis.
                #     eig_phon[q, lambda, atom, mu] -> E[q, lambda, p].
                E = eig_phon_source.reshape(N_pts, num_phon, num_phon)
                V_plus_all = np.matmul(V_plus_all, np.swapaxes(E, 1, 2))
                V_minus_all = np.matmul(V_minus_all, np.swapaxes(E, 1, 2))

                # (c) per-BRANCH frequency factor with the REAL frequency.
                #     ref_omega no longer appears in this construction.
                w_safe = np.maximum(w_phon_source, PHONON_FLOOR_MEV)
                freq = np.sqrt((hbar * hbar) / (S_val * w_safe))    # (N_pts, num_phon)
                V_plus_all *= freq[:, None, :]
                V_minus_all *= freq[:, None, :]

        # ==========================================
        # 2. HAMILTONIAN ASSEMBLY
        # ==========================================
        for q_idx in range(N_pts):
            H_BdG = np.zeros((dim, dim), dtype=np.complex128)

            """
            # --- Phonon Blocks from the Dynamical Matrix (Cartesian/ref_omega) ---
            # NOTE: incompatible with the branch-basis SLC block above. Only use
            # this together with the Cartesian/ref_omega SLC construction.
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

            # --- Phonon Block from Eigenvalues (branch basis, B_phon = 0) ---
            A_phon = np.diag(w_phon_source[q_idx])

            H_BdG[off_ph_p:off_ph_p+num_phon, off_ph_p:off_ph_p+num_phon] = A_phon
            H_BdG[off_ph_h:off_ph_h+num_phon, off_ph_h:off_ph_h+num_phon] = A_phon

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
            # Toggled by self.include_slc (CLI: --no_slc disables it).
            if hasattr(self, 'slc_axis') and is_FM and self.include_slc:
                Vp = V_plus_all[q_idx]
                Vm = V_minus_all[q_idx]

                #"""
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
                #"""

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

    def get_path_group_velocities(self, grad_f=None):
        """
        Cartesian group velocities (Angstrom/ps) of the hybrid modes at every
        explicit path point, shape (N_path, n_bands, 3).

        grad_f_hyb holds dE/df (meV per unit FRACTIONAL coordinate) on the grid.
        Path points lie on the grid, so the gradient is read from the matching
        grid point and converted: q_cart = f @ M with M = 2*pi*reciprocal_lattice
        (rows b_i), hence grad_f = M @ grad_q and v = grad_q / hbar.
        """
        hbar = 0.6582119569  # meV * ps
        grad = self.grad_f_hyb if grad_f is None else grad_f

        mesh = np.asarray(self.mesh)
        scaled = self.path_q_frac * mesh
        ijk = np.rint(scaled).astype(int) % mesh
        n_off = int((np.abs(scaled - np.rint(scaled)).max(axis=1) > 1e-3).sum())
        if n_off:
            print(f" WARNING: {n_off}/{self.N_path} path points are off-grid; their group velocity "
                  f"is taken from the nearest grid point.")
        idx = self.grid_map[ijk[:, 0], ijk[:, 1], ijk[:, 2]]

        # Band ordering check: on-grid path energies must equal the grid energies.
        dev = np.abs(self.path_w_hyb - self.w_hyb[idx]).max()
        if dev > 1e-3:
            print(f" WARNING: path vs grid hybrid energies differ by up to {dev:.3e} meV; "
                  f"check lattice_constant / band ordering before trusting the velocities.")

        M_inv = np.linalg.inv(self.reciprocal_lattice * 2.0 * np.pi)
        grad_q = np.einsum('ai,pni->pna', M_inv, grad[idx])
        return grad_q / hbar

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
        

        L_cart = L_z_total[:num_phon, :num_phon]

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

                # Rotate the Cartesian PAM onto the bare phonon eigenmodes, so
                # its index matches the branch index carried by path_eig_hyb.
                # E[p, lambda] with p = 3*atom + mu.
                E_q = self.path_eig_phon[q_idx].reshape(num_phon, num_phon).T
                L_branch = E_q.conj().T @ L_cart @ E_q

                L_z_phon = np.zeros((dim_total, dim_total), dtype=np.complex128)
                L_z_phon[:num_phon, :num_phon] = L_branch
                # Hole block: O_hole = -O_particle^*. The Cartesian shortcut
                # (hole == particle, valid because -i*hbar*eps is purely
                # imaginary) does NOT survive the rotation - L_branch is
                # Hermitian but generally complex.
                L_z_phon[dim_block:dim_block+num_phon,
                         dim_block:dim_block+num_phon] = -L_branch.conj()

                
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
        
        # --- Batched, diagonal-only evaluation -------------------------------
        # Only diag(T^dag O T) is needed, so contract straight to it via
        # einsum instead of forming the full dim x dim product per q-point.
        # T is (N, dim_total, dim_total); keep the first dim_block columns.
        # Slice to the physical columns FIRST (a view), then conjugate, so the
        # temporary is (N, dim_total, dim_block) rather than the full matrix.
        Tp = self.Qmatrix[:, :, :dim_block]                # view
        Tc = Tp.conj()

        def _diag_expect(O):
            return np.einsum('qib,ij,qjb->qb', Tc, O, Tp, optimize=True).real

        phon_chars = _diag_expect(J_phon)
        mag_chars = _diag_expect(J_spin)
        spin_ams = _diag_expect(L_z_spin)

        # --- Phonon angular momentum needs the per-q basis rotation ----------
        # L^z is intrinsically CARTESIAN (Levi-Civita on x,y,z). With the BdG
        # Hamiltonian assembled in the bare-phonon-branch basis, Qmatrix's
        # phonon rows are branch indices, so L must be rotated onto that basis
        # before contracting - exactly as save_hybrid_path_properties does.
        # Without this the per-branch PAM comes out scrambled across branches.
        L_cart = L_z_total[:num_phon, :num_phon]
        E = self.eig_phon.reshape(self.N, num_phon, num_phon)          # [q, lambda, p]
        Eq = np.swapaxes(E, 1, 2)                                      # [q, p, lambda]
        L_branch = np.matmul(np.swapaxes(Eq, 1, 2).conj(),
                             np.matmul(L_cart[None, :, :], Eq))        # (N, n_ph, n_ph)

        # L_z_phon is block-diagonal (phonon particle block = L_branch, phonon
        # hole block = -L_branch^*, everything else zero), so contract the two
        # blocks directly instead of materialising the (N, dim, dim) operator -
        # that array would be ~1.4 GB for an 8-atom cell on a 32^3 mesh.
        Tc_p, Tp_p = Tc[:, :num_phon, :], Tp[:, :num_phon, :]
        Tc_h = Tc[:, dim_block:dim_block + num_phon, :]
        Tp_h = Tp[:, dim_block:dim_block + num_phon, :]

        phon_ams = (
            np.einsum('qib,qij,qjb->qb', Tc_p, L_branch, Tp_p, optimize=True)
            - np.einsum('qib,qij,qjb->qb', Tc_h, L_branch.conj(), Tp_h, optimize=True)
        ).real

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
    """Scans phase space integrating exact path points against the BZ grid using Adaptive Broadening.

    c_type 0  M(path,n) -> M(k,m) + Ph(path-k,lam)              V^{+-}_{q,q-k}   (tex 2.1, emission term)
    c_type 1  M(path,n) + Ph(k-path,lam) -> M(k,m)              V^{+-}_{q,q-k}   (tex 2.1, absorption term; not with --legacy_bare)
    c_type 2  Ph(path,lam) + M(k-path,m) -> M(k,n)              V^{+-}_{k,+q}    (absorption of the phonon; tex 1.2 with the first label k-q)
    Vertex bookkeeping and audit: theory/bare_vertex_audit.md; --legacy_bare restores the old arguments.
    The relativistic (V^{++}, V^{--}) processes c_type 3/4 are scanned by phase_1_scan_path_rel (--relativistic): a
    separate kernel, so that this one is untouched (the extra branch cost +69 % runtime through register pressure).
    """
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
                # Magnon emission: M(path,n) -> M(k,m) + Ph(path-k,lam), bracket 1 + n_ph + n_k.
                # Vertex V_{q,q-k} (kp=path, q=path-k) as in tex 2.1; the absorption term follows below.
                # ---------------------------------------------------------
                dE_mag = path_w_mag[path_idx, n] - w_mag_grid[k_idx, m] - w_phon_grid[idx_qmink, lam]
                var_mag = 0.0
                for i in range(3):
                    d_g = -grad_f_mag[k_idx, m, i] + grad_f_phon[idx_qmink, lam, i]
                    step_w = d_g / mesh[i]
                    var_mag += step_w * step_w

                sigma_raw_mag = smearing * math.sqrt(var_mag / 12.0)
                sigma_mag = sigma_raw_mag if sigma_raw_mag > 1e-5 else 1e-5
                
                if (abs(dE_mag) < 2.0 * sigma_mag and path_w_mag[path_idx, n] >= BARE_MIN_SCATTERING_ENERGY
                        and w_mag_grid[k_idx, m] >= BARE_MIN_SCATTERING_ENERGY
                        and w_phon_grid[idx_qmink, lam] >= BARE_MIN_SCATTERING_ENERGY):
                    weight = (0.4179 / sigma_mag) * math.exp(-0.5 * (dE_mag*dE_mag) / (sigma_mag*sigma_mag))
                    kpx, kpy, kpz = path_q_cart[path_idx, 0], path_q_cart[path_idx, 1], path_q_cart[path_idx, 2]
                    if BARE_LEGACY:
                        qx = grid_q_cart[k_idx, 0] - path_q_cart[path_idx, 0]
                        qy = grid_q_cart[k_idx, 1] - path_q_cart[path_idx, 1]
                        qz = grid_q_cart[k_idx, 2] - path_q_cart[path_idx, 2]
                        V_sq = calc_vertex_V_path(kpx, kpy, kpz, qx, qy, qz, gammax, gammay, gammaz, lam, n, m, slc_axis, slc_rij, slc_rik, slc_J, slc_types, eig_phon_grid[idx_kminq], w_phon_grid[idx_kminq, lam], atom_masses, mag_moments)
                    else:
                        # tex 2.1:  V^{+-}_{q, q-k, lam, nm}: phonon momentum q-k, J~(k, q-k)
                        qx = path_q_cart[path_idx, 0] - grid_q_cart[k_idx, 0]
                        qy = path_q_cart[path_idx, 1] - grid_q_cart[k_idx, 1]
                        qz = path_q_cart[path_idx, 2] - grid_q_cart[k_idx, 2]
                        V_sq = calc_vertex_V_path(kpx, kpy, kpz, qx, qy, qz, gammax, gammay, gammaz, lam, n, m, slc_axis, slc_rij, slc_rik, slc_J, slc_types, eig_phon_grid[idx_qmink], w_phon_grid[idx_qmink, lam], atom_masses, mag_moments)
                    c_idx = cuda.atomic.add(channel_count, 0, 1)
                    if c_idx < chan_indices.shape[1]:
                        chan_indices[0, c_idx] = 0; chan_indices[1, c_idx] = path_idx; chan_indices[2, c_idx] = k_idx
                        chan_indices[3, c_idx] = idx_qmink; chan_indices[4, c_idx] = n; chan_indices[5, c_idx] = m; chan_indices[6, c_idx] = lam
                        chan_weights[c_idx] = V_sq * weight

                # ---------------------------------------------------------
                # Magnon absorption (tex 2.1, second term):  M(path,n) + Ph(k-path,lam) -> M(k,m)
                #   delta(E_q - E_k + E_ph(k-q)) [ n_ph(k-q) - n_k ],  same vertex V_{q,q-k}.  Not part of the legacy kernel.
                # ---------------------------------------------------------
                if not BARE_LEGACY:
                    dE_abs = path_w_mag[path_idx, n] - w_mag_grid[k_idx, m] + w_phon_grid[idx_kminq, lam]
                    var_abs = 0.0
                    for i in range(3):
                        d_g = -grad_f_mag[k_idx, m, i] + grad_f_phon[idx_kminq, lam, i]
                        step_w = d_g / mesh[i]
                        var_abs += step_w * step_w
                    sigma_abs_raw = smearing * math.sqrt(var_abs / 12.0)
                    sigma_abs = sigma_abs_raw if sigma_abs_raw > 1e-5 else 1e-5
                    if (abs(dE_abs) < 2.0 * sigma_abs and path_w_mag[path_idx, n] >= BARE_MIN_SCATTERING_ENERGY
                            and w_mag_grid[k_idx, m] >= BARE_MIN_SCATTERING_ENERGY
                            and w_phon_grid[idx_kminq, lam] >= BARE_MIN_SCATTERING_ENERGY):
                        weight_abs = (0.4179 / sigma_abs) * math.exp(-0.5 * (dE_abs * dE_abs) / (sigma_abs * sigma_abs))
                        kpx, kpy, kpz = path_q_cart[path_idx, 0], path_q_cart[path_idx, 1], path_q_cart[path_idx, 2]
                        qx = path_q_cart[path_idx, 0] - grid_q_cart[k_idx, 0]
                        qy = path_q_cart[path_idx, 1] - grid_q_cart[k_idx, 1]
                        qz = path_q_cart[path_idx, 2] - grid_q_cart[k_idx, 2]
                        V_sq = calc_vertex_V_path(kpx, kpy, kpz, qx, qy, qz, gammax, gammay, gammaz, lam, n, m, slc_axis, slc_rij, slc_rik, slc_J, slc_types, eig_phon_grid[idx_kminq], w_phon_grid[idx_kminq, lam], atom_masses, mag_moments)
                        c_idx = cuda.atomic.add(channel_count, 0, 1)
                        if c_idx < chan_indices.shape[1]:
                            chan_indices[0, c_idx] = 1; chan_indices[1, c_idx] = path_idx; chan_indices[2, c_idx] = k_idx
                            chan_indices[3, c_idx] = idx_kminq; chan_indices[4, c_idx] = n; chan_indices[5, c_idx] = m; chan_indices[6, c_idx] = lam
                            chan_weights[c_idx] = V_sq * weight_abs

                # ---------------------------------------------------------
                # Phonon channel: Ph(path,lam) + M(k-path,m) -> M(k,n), bracket n_{k-q} - n_k.
                # Vertex V_{k,+q} = V*_{k-q,-q} (kp=k, q=+path); the old kernel (and tex 1.2 as written) used V_{k,-q}.
                # ---------------------------------------------------------
                dE_phon = path_w_phon[path_idx, lam] + w_mag_grid[idx_kminq, m] - w_mag_grid[k_idx, n]
                var_phon = 0.0
                for i in range(3):
                    d_g = grad_f_mag[idx_kminq, m, i] - grad_f_mag[k_idx, n, i]
                    step_w = d_g / mesh[i]
                    var_phon += step_w * step_w

                sigma_raw_phon = smearing * math.sqrt(var_phon / 12.0)
                sigma_phon = sigma_raw_phon if sigma_raw_phon > 1e-5 else 1e-5

                if (abs(dE_phon) < 2.0 * sigma_phon and path_w_phon[path_idx, lam] >= BARE_MIN_SCATTERING_ENERGY
                        and w_mag_grid[idx_kminq, m] >= BARE_MIN_SCATTERING_ENERGY
                        and w_mag_grid[k_idx, n] >= BARE_MIN_SCATTERING_ENERGY):
                    weight = (0.4179 / sigma_phon) * math.exp(-0.5 * (dE_phon*dE_phon) / (sigma_phon*sigma_phon))
                    kpx, kpy, kpz = grid_q_cart[k_idx, 0], grid_q_cart[k_idx, 1], grid_q_cart[k_idx, 2]
                    if BARE_LEGACY:
                        qx, qy, qz = -path_q_cart[path_idx, 0], -path_q_cart[path_idx, 1], -path_q_cart[path_idx, 2]
                    else:
                        # phonon q absorbed by the magnon k-q -> k: amplitude V_{k, +q} = V_{k-q, -q}^*, J~(k-q, q)  (tex 1.2 writes V_{k,-q}: the
                        # first label should be k-q for the low magnon of the pair, see theory/bare_vertex_audit.md)
                        qx, qy, qz = path_q_cart[path_idx, 0], path_q_cart[path_idx, 1], path_q_cart[path_idx, 2]
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

    if omega < 1:
        return 0.0 + 0.0j
    
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
# Relativistic (magnon-number NON-conserving) bare vertices        [--relativistic]
# ==========================================
# Theory: theory/bare_scattering_rates_plus_relativistic.tex, "Relaxation Rates in the Multi-Sublattice Basis".
# The same spin-lattice coupling J^{ab,mu} that gives the number-conserving vertex V^{+-} (calc_vertex_V: isotropic
# exchange striction  J^xx + J^yy,  J^zz  and the antisymmetric  -i J^xy + i J^yx) also contains
#     H ~ sum_q (a_q + a^+_-q) [ V^{++}_{kq,nm} b^+_{k n} b^+_{q-k m} + V^{--}_{kq,nm} b_{k n} b_{-k-q m} ] ,
#     W^{++}_{nml}(k,q) = 1/(2 sqrt(S_n S_m)) [ J~^xx - s_n s_m J~^yy + i s_m J~^xy + i s_n J~^yx ](k-q, q)
#     W^{--}_{nml}(k,q) = 1/(2 sqrt(S_n S_m)) [ J~^xx - s_n s_m J~^yy - i s_m J~^xy - i s_n J~^yx ](-k-q, q)
#     V^{nu}_{kq lam,nm} = 1/sqrt(N) sum_{l,mu} sqrt(hbar / (2 M_l omega_q)) e^mu_{l lam}(q) W^{nu,mu}_{nml}
# (s_n = sign of the moment of magnetic atom n; 1/sqrt(N) is applied by the callers as the 1/N of |V|^2.)
# These vanish for an isotropic coupling J^{ab} = J delta_ab (exchange striction) and need an anisotropic
# (J^xx != J^yy) or a symmetric-off-diagonal (J^xy + J^yx != 0) part, or, for antiparallel sublattices, an
# antisymmetric one: they are the *relativistic* channels  ph(q) <-> mag(k) + mag(q-k)  (the phonon decays into two
# magnons / two magnons merge into a phonon, changing S^z by 2 hbar).


@cuda.jit(device=True)
def calc_vertex_nc(kind, kx, ky, kz, qx, qy, qz, conj_e, eig_phon_q, lambda_phon, omega, n, m,
                   slc_axis, slc_rij, slc_rik, slc_J, slc_types, atom_masses, mag_moments, min_omega):
    """
    Complex amplitude V^{++} (kind = 0) or V^{--} (kind = 1) of tex Sec. "Final Compact Form", without the 1/sqrt(N).

      (kx,ky,kz)   magnon label k of the vertex V_{k q lam, n m}  (the first magnon operator is b_{k n})
      (qx,qy,qz)   phonon label q of the vertex (the phonon of the field a_q + a^+_{-q}); omega = omega_{q lam}
      eig_phon_q   eigenvectors [lam, atom, mu] AT THE MOMENTUM WHOSE LABEL IS q; conj_e = 1 conjugates them. A
                   vertex V_{k,-q} needs e(-q) = conj(e(q)): pass the eigenvectors of q with conj_e = 1 and the
                   label -q.
    """
    if omega < min_omega:
        return 0.0 + 0.0j

    hbar = 0.6582119569  # meV * ps
    DALTON_TO_meV_PS2_PER_A2 = 0.10364269

    S_n = math.fabs(mag_moments[n] / 2.0)
    S_m = math.fabs(mag_moments[m] / 2.0)
    sig_n = 1.0
    if mag_moments[n] < 0.0:
        sig_n = -1.0
    sig_m = 1.0
    if mag_moments[m] < 0.0:
        sig_m = -1.0

    # first Fourier argument of J~ (conjugate to r_ij)
    if kind == 0:
        ax = kx - qx
        ay = ky - qy
        az = kz - qz
        sgn = 1.0
    else:
        ax = -kx - qx
        ay = -ky - qy
        az = -kz - qz
        sgn = -1.0

    J_tilde = cuda.local.array((3, 3), dtype=np.complex128)
    V_complex = 0.0 + 0.0j
    num_atoms = atom_masses.shape[0]
    pref_W = 1.0 / (2.0 * math.sqrt(S_n * S_m))

    for l in range(num_atoms):
        mass_l = atom_masses[l] * DALTON_TO_meV_PS2_PER_A2
        disp_amp = math.sqrt((hbar * hbar) / (2.0 * mass_l * omega))
        for mu in range(3):
            e_mu = eig_phon_q[lambda_phon, l, mu]
            if conj_e != 0:
                e_mu = complex(e_mu.real, -e_mu.imag)

            calc_fourier_transform_vec(ax, ay, az, qx, qy, qz, slc_axis, slc_rij, slc_rik, slc_J, slc_types,
                                       n + 1, m + 1, l + 1, mu, J_tilde)

            W = (J_tilde[0, 0] - sig_n * sig_m * J_tilde[1, 1]
                 + sgn * 1j * sig_m * J_tilde[0, 1] + sgn * 1j * sig_n * J_tilde[1, 0]) * pref_W
            V_complex += disp_amp * e_mu * W

    return V_complex


@cuda.jit(device=True)
def calc_pair_amplitude_nc(kind, k1x, k1y, k1z, k2x, k2y, k2z, qx, qy, qz, conj_e, eig_phon_q, lambda_phon, omega, n, m,
                           slc_axis, slc_rij, slc_rik, slc_J, slc_types, atom_masses, mag_moments, min_omega):
    """
    Exact golden-rule amplitude of the process with the unordered magnon pair {(k1,n),(k2,m)} and the phonon (q, lam):
        A = V_{k1,q,nm} + V_{k2,q,mn}
    (the sum over both orderings of the identical-boson operator product b_{k1 n} b_{k2 m}, which the ordered sum of
    the Hamiltonian contains twice). The tex replaces it by 2 V assuming V_{k1,nm} = V_{k2,mn}; that is exact only if the
    spin-lattice rows are closed under swapping the two magnetic atoms - the stored rows are not (a cut-off on the
    displaced atom), so the sum is used and no symmetry is assumed. The labels must obey k1 + k2 = +q (kind 0, V^{++}) or
    k1 + k2 = -q (kind 1, V^{--}) EXACTLY (literal, not folded) so that both terms use the same gauge.
    """
    V1 = calc_vertex_nc(kind, k1x, k1y, k1z, qx, qy, qz, conj_e, eig_phon_q, lambda_phon, omega, n, m,
                        slc_axis, slc_rij, slc_rik, slc_J, slc_types, atom_masses, mag_moments, min_omega)
    V2 = calc_vertex_nc(kind, k2x, k2y, k2z, qx, qy, qz, conj_e, eig_phon_q, lambda_phon, omega, m, n,
                        slc_axis, slc_rij, slc_rik, slc_J, slc_types, atom_masses, mag_moments, min_omega)
    return V1 + V2


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

# Bare scans: modes below this energy never scatter (same cut-off as MIN_SCATTERING_ENERGY_MEV of the hybrid scans). Without it
# a gapless Gamma magnon (n_BE ~ kT/E ~ 1e8) dominates the rates (bare CrI3 phonon rates of 1e4-1e6 ps^-1).
BARE_MIN_SCATTERING_ENERGY = 0.2      # meV

# Bare number-conserving magnon-phonon rates: vertex bookkeeping of theory/mag_ph.txt (audit: theory/bare_vertex_audit.md).
# False (default): V^{+-}_{K,Q} with the annihilated magnon as first Fourier argument, i.e. for the process magnon a <-> magnon b + phonon p
#                  the vertex is V(K=a, Q=a-b); path magnon rates include the phonon-absorption term; phonon energy gradients enter
#                  the adaptive width. True (--legacy_bare): the old kernels (V(K=a, Q=b-a), emission term only, zero phonon gradients).
BARE_LEGACY = False


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

                #sigma = min_sigma

                if (abs(dE) < 2.0 * sigma and w_mag[q_idx, n] >= BARE_MIN_SCATTERING_ENERGY
                        and w_mag[k_idx, m] >= BARE_MIN_SCATTERING_ENERGY
                        and w_phon[idx_qmink, lam] >= BARE_MIN_SCATTERING_ENERGY):
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

                    V_sq = V.real**2 + V.imag**2
                    if BARE_LEGACY:
                        V_Gmp = calc_vertex_V(kpx_cart, kpy_cart, kpz_cart, x_kminq_cart, y_kminq_cart, z_kminq_cart, idx_kminq, lam, n, m, grid_map, slc_axis, slc_rij, slc_rik, slc_J, slc_types, eig_phon, w_phon, atom_masses, mag_moments)
                        V_sq = V_Gmp.real**2 + V_Gmp.imag**2

                    c_idx = cuda.atomic.add(channel_count, 0, 1)
                    if c_idx < chan_indices.shape[1]:
                        chan_indices[0, c_idx] = 0; chan_indices[1, c_idx] = q_idx; chan_indices[2, c_idx] = k_idx
                        chan_indices[3, c_idx] = idx_qmink; chan_indices[4, c_idx] = n; chan_indices[5, c_idx] = m; chan_indices[6, c_idx] = lam
                        chan_weights[c_idx] = V_sq * delta_weight
                
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
def phase_lifetime(chan_indices, chan_weights, num_channels, n_mag, n_phon, gamma_mag, gamma_phon, N_points):
    """
    SMRTA rates on the full grid from the unified channels of phase_1_scan:
        M(q, n) <-> M(k, m) + Ph(p, lam)
    Every channel contributes to all three participants:
      magnon q  (parent, splitting):            (1 + n_k + n_p)
      magnon k  (child, absorbs phonon p -> q): (n_p - n_q)
      phonon p  (child, absorbed with k -> q):  (n_k - n_q)
    """
    idx = cuda.grid(1)
    if idx >= num_channels[0] or idx >= chan_weights.shape[0]:
        return

    q_idx = chan_indices[1, idx]
    k_idx = chan_indices[2, idx]
    p_idx = chan_indices[3, idx]
    n     = chan_indices[4, idx]
    m     = chan_indices[5, idx]
    lam   = chan_indices[6, idx]
    V_sq  = chan_weights[idx]

    hbar = 0.6582119569  # meV * ps
    pref = (2.0 * math.pi / hbar) / N_points

    n_q = n_mag[q_idx, n]
    n_k = n_mag[k_idx, m]
    n_p = n_phon[p_idx, lam]

    nmb = n_mag.shape[1]
    npb = n_phon.shape[1]

    cuda.atomic.add(gamma_mag,  q_idx * nmb + n,   pref * V_sq * (1.0 + n_k + n_p))
    cuda.atomic.add(gamma_mag,  k_idx * nmb + m,   pref * V_sq * (n_p - n_q))
    cuda.atomic.add(gamma_phon, p_idx * npb + lam, pref * V_sq * (n_k - n_q))


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

    MIN_SCATTERING_ENERGY_MEV = 0.2

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

                #"""
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
                #"""

                #sigma =  min_sigma
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

                #"""

                # Same adaptive broadening; here the parent is s and the two
                # children are the path mode and k, matching dE's sign pattern.
                variance = 0.0
                for i in range(3):
                    d_g = grad_f_hyb[s_idx, b_s, i] - grad_f_hyb[k_idx, b_k, i]
                    step_width = d_g / mesh[i]
                    variance += step_width * step_width

                sigma_raw = base_smearing * math.sqrt(variance / 12.0)
                sigma = sigma_raw if sigma_raw > min_sigma else min_sigma
                #"""
                
                
                #sigma = min_sigma
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

      c_type 0 (splitting,   path -> k + p): 4 * pi/hbar   * |Gamma~|^2 * (1 + n_k + n_p)
      c_type 1 (coalescence, path + k -> s): 8 * pi/hbar * |Gamma~|^2 * (n_k - n_s)

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
    prefactor_split = (4 * math.pi / hbar) / N_grid_points
    prefactor_coal  = (8 * math.pi / hbar) / N_grid_points 


    # as far as I can see: in the case of bcc Fe only coalesence gives phonon lifetimes

    n_k = n_hyb_grid[k_idx, b_k]
    n_o = n_hyb_grid[o_idx, b_o]

    num_bands = n_hyb_grid.shape[1]

    if c_type == 0:
        rate_q = prefactor_split * V_sq * (n_k + n_o + 1.0)
    if c_type == 1:
        rate_q = prefactor_coal * V_sq * (n_k - n_o)

    cuda.atomic.add(gamma_hyb_path, path_idx * num_bands + b_q, rate_q)



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
def phase_lifetime_path(chan_indices, chan_weights, num_channels, n_mag_grid, n_phon_grid, gamma_mag_path, gamma_phon_path,
                        gamma_mag_rel_path, gamma_phon_rel_path, N_grid_points):
    """Calculates lifetimes for the path array by evaluating the thermal distributions on the regular grid.

    c_type 0 / 2   number-conserving V^{+-}  ->  gamma_mag_path / gamma_phon_path
    c_type 3 / 4   relativistic V^{--} / V^{++} (--relativistic) -> gamma_phon_rel_path / gamma_mag_rel_path
        3  phonon(path,lam):  (2 pi/hbar N) * 1/2 |A^{--}|^2 delta (1 + n_k,n + n_{q-k},m)     [tex 1.1: 2|V|^2 if A = 2V]
        4  magnon(path,n)  :  (2 pi/hbar N) *     |A^{++}|^2 delta (n_k,m - n_ph(k+q,lam))      [tex 2.2: 4|V|^2 if A = 2V]
       A = V_{k,nm} + V_{p,mn} is the exact two-ordering amplitude of the magnon pair (calc_pair_amplitude_nc);
       chan_weights = |A|^2 delta; the sums run over all ordered (k, n, m) / (k, m, lam) of the scan.
    """
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
        
    elif c_type == 1:
        # Magnon absorption M(path,n) + Ph(p=k-path,lam) -> M(k,m), tex 2.1 second term:  n_ph - n_k
        nk_mag = n_mag_grid[k_idx, m]
        n_ph = n_phon_grid[p_idx, lam]
        rate = fgr_prefactor * V_sq * (n_ph - nk_mag)
        cuda.atomic.add(gamma_mag_path, path_idx * num_mag_branches + n, rate)

    elif c_type == 2:
        # Unified phonon channel (mirrors the confirmed phonon RTA rate)
        nk_mag = n_mag_grid[k_idx, n]
        n_kminq_mag = n_mag_grid[p_idx, m]
        rate = fgr_prefactor * V_sq * (n_kminq_mag - nk_mag)
        cuda.atomic.add(gamma_phon_path, path_idx * num_phon_branches + lam, rate)

    elif c_type == 3:
        # Relativistic phonon channel: ordered (k,n),(p=q-k,m), each unordered pair is enumerated twice -> 1/2
        nk_mag = n_mag_grid[k_idx, n]
        np_mag = n_mag_grid[p_idx, m]
        rate = 0.5 * fgr_prefactor * V_sq * (1.0 + nk_mag + np_mag)
        cuda.atomic.add(gamma_phon_rel_path, path_idx * num_phon_branches + lam, rate)

    elif c_type == 4:
        # Relativistic magnon channel: M(path,n) + M(k,m) <-> Ph(s=path+k,lam); one partner per unordered pair, |A|^2 = |V_nm + V_mn|^2
        nk_mag = n_mag_grid[k_idx, m]
        nph = n_phon_grid[p_idx, lam]
        rate = fgr_prefactor * V_sq * (nk_mag - nph)
        cuda.atomic.add(gamma_mag_rel_path, path_idx * num_mag_branches + n, rate)


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





@cuda.jit
def phase_1_scan_path_rel(mesh, grid_q_frac, grid_q_cart, grid_map,
                          path_q_frac, path_q_cart, path_w_phon, path_w_mag, path_eig_phon,
                          w_phon_grid, w_mag_grid, eig_phon_grid,
                          grad_f_phon, grad_f_mag,
                          slc_axis, slc_rij, slc_rik, slc_J, slc_types,
                          smearing, min_sigma, chan_indices, chan_weights, channel_count,
                          atom_masses, mag_moments, gamma_idx):
    """
    Relativistic (magnon-number non-conserving) processes of the path scan (--relativistic); appends to the channel
    buffers that phase_1_scan_path filled (same layout; run it AFTER that kernel, same channel_count).
    c_type 3  Ph(path,lam) <-> M(k,n) + M(path-k,m)    chan_weights = |A^{--}|^2 delta,  A = V^{--}_{k,-path,nm} + V^{--}_{path-k,-path,mn}
    c_type 4  M(path,n) + M(k,m) <-> Ph(path+k,lam)    chan_weights = |A^{++}|^2 delta,  A = V^{++}_{path,s,nm}  + V^{++}_{s-path,s,mn}
    Gaussian width max(adaptive, min_sigma); partner magnons/phonons at Gamma and zero-weight events are not stored.
    (Separate kernel: the extra branch in phase_1_scan_path cost +69 % runtime of the default scan.)
    """
    path_idx, k_idx = cuda.grid(2)
    if path_idx >= path_q_frac.shape[0] or k_idx >= grid_q_frac.shape[0]:
        return

    n_mag = path_w_mag.shape[1]
    n_phon = path_w_phon.shape[1]

    # path - k -> p (grid nearest neighbour)
    px_int = int(math.floor(((path_q_frac[path_idx, 0] - grid_q_frac[k_idx, 0]) % 1.0) * mesh[0] + 0.5)) % mesh[0]
    py_int = int(math.floor(((path_q_frac[path_idx, 1] - grid_q_frac[k_idx, 1]) % 1.0) * mesh[1] + 0.5)) % mesh[1]
    pz_int = int(math.floor(((path_q_frac[path_idx, 2] - grid_q_frac[k_idx, 2]) % 1.0) * mesh[2] + 0.5)) % mesh[2]
    idx_qmink = grid_map[px_int, py_int, pz_int]

    # 3. Kinematics mapping (relativistic type 4): path + k -> s(grid nearest neighbour)
    sx_int = int(math.floor(((path_q_frac[path_idx, 0] + grid_q_frac[k_idx, 0]) % 1.0) * mesh[0] + 0.5)) % mesh[0]
    sy_int = int(math.floor(((path_q_frac[path_idx, 1] + grid_q_frac[k_idx, 1]) % 1.0) * mesh[1] + 0.5)) % mesh[1]
    sz_int = int(math.floor(((path_q_frac[path_idx, 2] + grid_q_frac[k_idx, 2]) % 1.0) * mesh[2] + 0.5)) % mesh[2]
    idx_pplusk = grid_map[sx_int, sy_int, sz_int]
    # Phonon label of the type-4 vertex V^{++}_{q,k+q}. UMKLAPP GAUGE: the eigenvector of the grid point s is in the
    # atom-position gauge and is only valid together with the FOLDED momentum of that grid point, so the label is
    # the folded one (not path + k, which can differ from it by a reciprocal-lattice vector).
    # grid index of the path point itself (-1 when it is not a grid point, e.g. a high-symmetry path)
    P_grid_idx = -1
    fx = path_q_frac[path_idx, 0] * mesh[0]
    fy = path_q_frac[path_idx, 1] * mesh[1]
    fz = path_q_frac[path_idx, 2] * mesh[2]
    if (abs(fx - math.floor(fx + 0.5)) < 1e-6 and abs(fy - math.floor(fy + 0.5)) < 1e-6
            and abs(fz - math.floor(fz + 0.5)) < 1e-6):
        P_grid_idx = grid_map[int(math.floor(fx + 0.5)) % mesh[0], int(math.floor(fy + 0.5)) % mesh[1],
                              int(math.floor(fz + 0.5)) % mesh[2]]
    s_lab_x = grid_q_cart[idx_pplusk, 0]
    s_lab_y = grid_q_cart[idx_pplusk, 1]
    s_lab_z = grid_q_cart[idx_pplusk, 2]


    for n in range(n_mag):
        for m in range(n_mag):
            for lam in range(n_phon):
                # ---------------------------------------------------------
                # Relativistic phonon channel (c_type 3): Ph(path,lam) <-> M(k,n) + M(path-k,m).
                # Merge amplitude V^{--}_{k,-q,lam,nm}: phonon label -path, so its eigenvector is
                # e(-path) = conj(e(path)) (the path point's own eigenvector, conjugated).
                # ---------------------------------------------------------
                dE_3 = path_w_phon[path_idx, lam] - w_mag_grid[k_idx, n] - w_mag_grid[idx_qmink, m]
                var_3 = 0.0
                for i in range(3):
                    d_g = -grad_f_mag[k_idx, n, i] + grad_f_mag[idx_qmink, m, i]
                    step_w = d_g / mesh[i]
                    var_3 += step_w * step_w
                sigma_raw_3 = smearing * math.sqrt(var_3 / 12.0)
                sigma_3 = sigma_raw_3 if sigma_raw_3 > min_sigma else min_sigma

                if (abs(dE_3) < 2.0 * sigma_3 and k_idx != gamma_idx and idx_qmink != gamma_idx and w_mag_grid[k_idx, n] >= BARE_MIN_SCATTERING_ENERGY
                        and w_mag_grid[idx_qmink, m] >= BARE_MIN_SCATTERING_ENERGY and path_w_phon[path_idx, lam] >= BARE_MIN_SCATTERING_ENERGY):   # Gamma partner: not in the bath
                    weight = (0.4179 / sigma_3) * math.exp(-0.5 * (dE_3 * dE_3) / (sigma_3 * sigma_3))
                    # canonical label pair (lower grid index first, ties by band), see phase_1_scan_nc
                    if k_idx < idx_qmink or (k_idx == idx_qmink and n <= m):
                        V = calc_pair_amplitude_nc(1, grid_q_cart[k_idx, 0], grid_q_cart[k_idx, 1], grid_q_cart[k_idx, 2],
                                       path_q_cart[path_idx, 0] - grid_q_cart[k_idx, 0],
                                       path_q_cart[path_idx, 1] - grid_q_cart[k_idx, 1],
                                       path_q_cart[path_idx, 2] - grid_q_cart[k_idx, 2],
                                       -path_q_cart[path_idx, 0], -path_q_cart[path_idx, 1], -path_q_cart[path_idx, 2],
                                       1, path_eig_phon[path_idx], lam, path_w_phon[path_idx, lam], n, m,
                                       slc_axis, slc_rij, slc_rik, slc_J, slc_types, atom_masses, mag_moments,
                                       PHONON_COUPLING_CUTOFF)
                    else:
                        V = calc_pair_amplitude_nc(1, grid_q_cart[idx_qmink, 0], grid_q_cart[idx_qmink, 1], grid_q_cart[idx_qmink, 2],
                                       path_q_cart[path_idx, 0] - grid_q_cart[idx_qmink, 0],
                                       path_q_cart[path_idx, 1] - grid_q_cart[idx_qmink, 1],
                                       path_q_cart[path_idx, 2] - grid_q_cart[idx_qmink, 2],
                                       -path_q_cart[path_idx, 0], -path_q_cart[path_idx, 1], -path_q_cart[path_idx, 2],
                                       1, path_eig_phon[path_idx], lam, path_w_phon[path_idx, lam], m, n,
                                       slc_axis, slc_rij, slc_rik, slc_J, slc_types, atom_masses, mag_moments,
                                       PHONON_COUPLING_CUTOFF)
                    V_sq = V.real * V.real + V.imag * V.imag
                    c_idx = -1
                    if V_sq > 0.0:                      # zero-weight events are not stored
                        c_idx = cuda.atomic.add(channel_count, 0, 1)
                    if c_idx >= 0 and c_idx < chan_indices.shape[1]:
                        chan_indices[0, c_idx] = 3; chan_indices[1, c_idx] = path_idx; chan_indices[2, c_idx] = k_idx
                        chan_indices[3, c_idx] = idx_qmink; chan_indices[4, c_idx] = n; chan_indices[5, c_idx] = m; chan_indices[6, c_idx] = lam
                        chan_weights[c_idx] = V_sq * weight

                # ---------------------------------------------------------
                # Relativistic magnon channel (c_type 4): M(path,n) + M(k,m) <-> Ph(path+k,lam).
                # Creation amplitude V^{++}_{q,k+q,lam,nm}: magnon label q = path, phonon label s = path + k
                # (eigenvector of the grid point s, plain).
                # ---------------------------------------------------------
                dE_4 = path_w_mag[path_idx, n] + w_mag_grid[k_idx, m] - w_phon_grid[idx_pplusk, lam]
                var_4 = 0.0
                for i in range(3):
                    d_g = grad_f_mag[k_idx, m, i] - grad_f_phon[idx_pplusk, lam, i]
                    step_w = d_g / mesh[i]
                    var_4 += step_w * step_w
                sigma_raw_4 = smearing * math.sqrt(var_4 / 12.0)
                sigma_4 = sigma_raw_4 if sigma_raw_4 > min_sigma else min_sigma

                if (abs(dE_4) < 2.0 * sigma_4 and k_idx != gamma_idx and idx_pplusk != gamma_idx and w_mag_grid[k_idx, m] >= BARE_MIN_SCATTERING_ENERGY
                        and path_w_mag[path_idx, n] >= BARE_MIN_SCATTERING_ENERGY and w_phon_grid[idx_pplusk, lam] >= BARE_MIN_SCATTERING_ENERGY):   # Gamma partner: not in the bath
                    weight = (0.4179 / sigma_4) * math.exp(-0.5 * (dE_4 * dE_4) / (sigma_4 * sigma_4))
                    # canonical label pair (lower grid index first, ties by band) when the path point is itself a grid point
                    if P_grid_idx >= 0 and (k_idx < P_grid_idx or (k_idx == P_grid_idx and m < n)):
                        V = calc_pair_amplitude_nc(0, grid_q_cart[k_idx, 0], grid_q_cart[k_idx, 1], grid_q_cart[k_idx, 2],
                                       s_lab_x - grid_q_cart[k_idx, 0], s_lab_y - grid_q_cart[k_idx, 1],
                                       s_lab_z - grid_q_cart[k_idx, 2],
                                       s_lab_x, s_lab_y, s_lab_z,
                                       0, eig_phon_grid[idx_pplusk], lam, w_phon_grid[idx_pplusk, lam], m, n,
                                       slc_axis, slc_rij, slc_rik, slc_J, slc_types, atom_masses, mag_moments,
                                       PHONON_COUPLING_CUTOFF)
                    else:
                        V = calc_pair_amplitude_nc(0, path_q_cart[path_idx, 0], path_q_cart[path_idx, 1], path_q_cart[path_idx, 2],
                                       s_lab_x - path_q_cart[path_idx, 0], s_lab_y - path_q_cart[path_idx, 1],
                                       s_lab_z - path_q_cart[path_idx, 2],
                                       s_lab_x, s_lab_y, s_lab_z,
                                       0, eig_phon_grid[idx_pplusk], lam, w_phon_grid[idx_pplusk, lam], n, m,
                                       slc_axis, slc_rij, slc_rik, slc_J, slc_types, atom_masses, mag_moments,
                                       PHONON_COUPLING_CUTOFF)
                    V_sq = V.real * V.real + V.imag * V.imag
                    c_idx = -1
                    if V_sq > 0.0:                      # zero-weight events are not stored
                        c_idx = cuda.atomic.add(channel_count, 0, 1)
                    if c_idx >= 0 and c_idx < chan_indices.shape[1]:
                        chan_indices[0, c_idx] = 4; chan_indices[1, c_idx] = path_idx; chan_indices[2, c_idx] = k_idx
                        chan_indices[3, c_idx] = idx_pplusk; chan_indices[4, c_idx] = n; chan_indices[5, c_idx] = m; chan_indices[6, c_idx] = lam
                        chan_weights[c_idx] = V_sq * weight





# ==========================================
# Relativistic bare dynamics: ph(q) <-> mag(k,n) + mag(p=q-k,m)            [--relativistic]
# ==========================================
# One stored event per ORDERED triple (q,lam ; k,n ; p,m) (both orderings of the magnon pair are present, because the
# scan runs over all k). With w = |A|^2 delta, A = V^{--}_{k,-q,nm} + V^{--}_{p,-q,mn} (exact pair amplitude), every event
# updates ALL THREE populations by
#       R = 1/2 (2 pi / hbar N) w [ (n_ph+1) n_k n_p - n_ph (n_k+1)(n_p+1) ],   dn_ph += R,  dn_k -= R,  dn_p -= R .
# Summed over the ordered events this gives the phonon rate (2 pi/hbar) sum_unordered |A|^2 [..] (= tex 1.1, 2|V|^2 per
# ordered pair, when A = 2V) and, because an unordered pair is seen twice (once per ordering, each updating both
# magnons), the magnon rate (2 pi/hbar)|A|^2 [..] per partner (= tex 2.2, 4|V|^2). Magnon number changes by -2 per merge,
# the phonon number by +1. Linearised: gamma_ph = pref w (1 + n_k + n_p), gamma_k = pref w (n_p - n_ph), pref = (2 pi/hbar N)/2.


@cuda.jit
def phase_1_scan_nc(mesh, q_grid, q_grid_cart, grid_map, w_phon, w_mag, eig_phon,
                    grad_f_phon, grad_f_mag,
                    slc_axis, slc_rij, slc_rik, slc_J, slc_types,
                    base_smearing, min_sigma, nc_indices, nc_weights, nc_count, atom_masses, mag_moments, gamma_idx):
    """Phase-space scan of the relativistic events on the BZ grid (thread = (phonon q, magnon k)).
    Events in which the Gamma point takes part (phonon q or either magnon) are skipped: like in the diagnostics, Gamma is not
    part of the bath (its occupations are initialised to 0, whereas n_BE of the gapped Gamma magnon is huge)."""
    q_idx, k_idx = cuda.grid(2)
    N = q_grid.shape[0]
    if q_idx >= N or k_idx >= N:
        return

    n_mag = w_mag.shape[1]
    n_phon = w_phon.shape[1]

    qx_i = (q_grid[q_idx, 0] - q_grid[k_idx, 0] + mesh[0]) % mesh[0]
    qy_i = (q_grid[q_idx, 1] - q_grid[k_idx, 1] + mesh[1]) % mesh[1]
    qz_i = (q_grid[q_idx, 2] - q_grid[k_idx, 2] + mesh[2]) % mesh[2]
    p_idx = grid_map[qx_i, qy_i, qz_i]

    if q_idx == gamma_idx or k_idx == gamma_idx or p_idx == gamma_idx:
        return

    for lam in range(n_phon):
        omega = w_phon[q_idx, lam]
        if omega < PHONON_COUPLING_CUTOFF or omega < BARE_MIN_SCATTERING_ENERGY:   # vertex cut-off / scattering cut-off
            continue
        for n in range(n_mag):
            if w_mag[k_idx, n] < BARE_MIN_SCATTERING_ENERGY:
                continue
            for m in range(n_mag):
                if w_mag[p_idx, m] < BARE_MIN_SCATTERING_ENERGY:
                    continue
                dE = omega - w_mag[k_idx, n] - w_mag[p_idx, m]
                variance = 0.0
                for i in range(3):
                    d_g = -grad_f_mag[k_idx, n, i] + grad_f_mag[p_idx, m, i]
                    step_width = d_g / mesh[i]
                    variance += step_width * step_width
                sigma_raw = base_smearing * math.sqrt(variance / 12.0)
                sigma = sigma_raw if sigma_raw > min_sigma else min_sigma

                if abs(dE) < 2.0 * sigma:
                    delta_weight = (0.4179 / sigma) * math.exp(-0.5 * (dE * dE) / (sigma * sigma))
                    # CANONICAL LABELS: the pair amplitude is ONE matrix element, so both orderings of the pair must
                    # carry the same weight. The label pair is always (folded momentum of the magnon with the LOWER
                    # grid index - ties: the lower band - and q minus it) with the bands ordered accordingly.
                    if k_idx < p_idx or (k_idx == p_idx and n <= m):
                        V = calc_pair_amplitude_nc(1, q_grid_cart[k_idx, 0], q_grid_cart[k_idx, 1], q_grid_cart[k_idx, 2],
                                       q_grid_cart[q_idx, 0] - q_grid_cart[k_idx, 0],
                                       q_grid_cart[q_idx, 1] - q_grid_cart[k_idx, 1],
                                       q_grid_cart[q_idx, 2] - q_grid_cart[k_idx, 2],
                                       -q_grid_cart[q_idx, 0], -q_grid_cart[q_idx, 1], -q_grid_cart[q_idx, 2],
                                       1, eig_phon[q_idx], lam, omega, n, m,
                                       slc_axis, slc_rij, slc_rik, slc_J, slc_types, atom_masses, mag_moments,
                                       PHONON_COUPLING_CUTOFF)
                    else:
                        V = calc_pair_amplitude_nc(1, q_grid_cart[p_idx, 0], q_grid_cart[p_idx, 1], q_grid_cart[p_idx, 2],
                                       q_grid_cart[q_idx, 0] - q_grid_cart[p_idx, 0],
                                       q_grid_cart[q_idx, 1] - q_grid_cart[p_idx, 1],
                                       q_grid_cart[q_idx, 2] - q_grid_cart[p_idx, 2],
                                       -q_grid_cart[q_idx, 0], -q_grid_cart[q_idx, 1], -q_grid_cart[q_idx, 2],
                                       1, eig_phon[q_idx], lam, omega, m, n,
                                       slc_axis, slc_rij, slc_rik, slc_J, slc_types, atom_masses, mag_moments,
                                       PHONON_COUPLING_CUTOFF)
                    V_sq = V.real * V.real + V.imag * V.imag
                    if V_sq == 0.0:
                        continue
                    c_idx = cuda.atomic.add(nc_count, 0, 1)
                    if c_idx < nc_indices.shape[1]:
                        nc_indices[0, c_idx] = 3; nc_indices[1, c_idx] = q_idx; nc_indices[2, c_idx] = k_idx
                        nc_indices[3, c_idx] = p_idx; nc_indices[4, c_idx] = n; nc_indices[5, c_idx] = m
                        nc_indices[6, c_idx] = lam
                        nc_weights[c_idx] = V_sq * delta_weight


@cuda.jit
def phase_2_time_step_nc(nc_indices, nc_weights, num_nc, n_mag, n_phon, dn_mag, dn_phon, N_points):
    idx = cuda.grid(1)
    if idx >= num_nc[0] or idx >= nc_weights.shape[0]:
        return
    q_idx = nc_indices[1, idx]
    k_idx = nc_indices[2, idx]
    p_idx = nc_indices[3, idx]
    n = nc_indices[4, idx]
    m = nc_indices[5, idx]
    lam = nc_indices[6, idx]
    hbar = 0.6582119569  # meV * ps
    pref = 0.5 * (2.0 * math.pi / hbar) / N_points

    nk = n_mag[k_idx, n]
    npp = n_mag[p_idx, m]
    nph = n_phon[q_idx, lam]
    R = pref * nc_weights[idx] * ((nph + 1.0) * nk * npp - nph * (nk + 1.0) * (npp + 1.0))

    cuda.atomic.add(dn_phon, q_idx * n_phon.shape[1] + lam, R)
    cuda.atomic.add(dn_mag, k_idx * n_mag.shape[1] + n, -R)
    cuda.atomic.add(dn_mag, p_idx * n_mag.shape[1] + m, -R)


@cuda.jit
def phase_2_flux_diagnostic_nc(nc_indices, nc_weights, num_nc, n_mag, n_phon, w_mag, w_phon,
                               N_points, inv_bin_width, n_bins, acc, flux_branch, flux_spec):
    """Same bookkeeping as phase_2_flux_diagnostic for the relativistic events (acc has 4 entries; the conserving kernel
    only fills 0..2):
        acc[0] += dE_mag/dt = -R (w_k + w_p)    acc[1] += dE_phon/dt = R w_q
        acc[2] += dN_phon/dt = R                acc[3] += dN_mag/dt  = -2 R
    flux_branch / flux_spec: magnon -> phonon energy flux R (w_k + w_p) by phonon branch / phonon energy."""
    idx = cuda.grid(1)
    if idx >= num_nc[0] or idx >= nc_weights.shape[0]:
        return
    q_idx = nc_indices[1, idx]
    k_idx = nc_indices[2, idx]
    p_idx = nc_indices[3, idx]
    n = nc_indices[4, idx]
    m = nc_indices[5, idx]
    lam = nc_indices[6, idx]
    hbar = 0.6582119569  # meV * ps
    pref = 0.5 * (2.0 * math.pi / hbar) / N_points

    nk = n_mag[k_idx, n]
    npp = n_mag[p_idx, m]
    nph = n_phon[q_idx, lam]
    R = pref * nc_weights[idx] * ((nph + 1.0) * nk * npp - nph * (nk + 1.0) * (npp + 1.0))
    wk = w_mag[k_idx, n]
    wp = w_mag[p_idx, m]
    wq = w_phon[q_idx, lam]

    cuda.atomic.add(acc, 0, -R * (wk + wp))
    cuda.atomic.add(acc, 1, R * wq)
    cuda.atomic.add(acc, 2, R)
    cuda.atomic.add(acc, 3, -2.0 * R)
    cuda.atomic.add(flux_branch, lam, R * (wk + wp))
    b = int(wq * inv_bin_width)
    if b < 0:
        b = 0
    if b >= n_bins:
        b = n_bins - 1
    cuda.atomic.add(flux_spec, b, R * (wk + wp))


@cuda.jit
def phase_lifetime_nc_dyn(nc_indices, nc_weights, num_nc, n_mag, n_phon, gamma_mag, gamma_phon, N_points):
    """Linearised relaxation rates of every mode from exactly the events the dynamics integrates (n = the given
    reference occupations): gamma_ph += pref w (1 + n_k + n_p), gamma_k += pref w (n_p - n_ph), gamma_p += pref w (n_k - n_ph),
    with pref = (2 pi / hbar N) / 2. Compare with phase_lifetime_path (c_type 3 / 4) for a start-up consistency check."""
    idx = cuda.grid(1)
    if idx >= num_nc[0] or idx >= nc_weights.shape[0]:
        return
    q_idx = nc_indices[1, idx]
    k_idx = nc_indices[2, idx]
    p_idx = nc_indices[3, idx]
    n = nc_indices[4, idx]
    m = nc_indices[5, idx]
    lam = nc_indices[6, idx]
    hbar = 0.6582119569  # meV * ps
    pref = 0.5 * (2.0 * math.pi / hbar) / N_points * nc_weights[idx]

    nk = n_mag[k_idx, n]
    npp = n_mag[p_idx, m]
    nph = n_phon[q_idx, lam]
    cuda.atomic.add(gamma_phon, q_idx * n_phon.shape[1] + lam, pref * (1.0 + nk + npp))
    cuda.atomic.add(gamma_mag, k_idx * n_mag.shape[1] + n, pref * (npp - nph))
    cuda.atomic.add(gamma_mag, p_idx * n_mag.shape[1] + m, pref * (nk - nph))


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


# ==========================================
# Dynamics diagnostics (flux, thermal fits, entropy, spectra)
# ==========================================
KB_MEV = 0.08617333262          # meV/K
PHONON_COUPLING_CUTOFF = 1.0    # meV; calc_vertex_V returns 0 below this, such phonons never scatter
MAGNON_MIN_ENERGY = 1e-3        # meV; excludes the Gamma-point magnon


@cuda.jit
def phase_2_flux_diagnostic(chan_indices, chan_weights, num_channels, n_mag, n_phon, w_mag, w_phon,
                            N_points, inv_bin_width, n_bins, acc, flux_branch, flux_spec):
    """
    Instantaneous energy bookkeeping of the unified channels M(q) <-> M(k) + Ph(p), evaluated with
    exactly the rate of phase_2_time_step (run only at output steps, so the hot loop stays free of
    contended atomics). Sums over channels (divide by N_points for per-cell values):
        acc[0] = dE_mag/dt   = sum r (w_q - w_k)
        acc[1] = dE_phon/dt  = - sum r w_p
        acc[2] = dN_phon/dt  = - sum r
    so acc[0] + acc[1] = sum r dE is the energy-conservation violation of the broadened delta.
    flux_branch[lam] and flux_spec[bin(w_p)] hold the magnon -> phonon energy flux -sum r (w_q - w_k)
    resolved by phonon branch and by phonon energy.
    """
    idx = cuda.grid(1)
    if idx >= num_channels[0] or idx >= chan_weights.shape[0]:
        return

    q_idx = chan_indices[1, idx]
    k_idx = chan_indices[2, idx]
    p_idx = chan_indices[3, idx]
    n = chan_indices[4, idx]
    m = chan_indices[5, idx]
    lam = chan_indices[6, idx]
    V_sq = chan_weights[idx]

    hbar = 0.6582119569  # meV * ps
    fgr_prefactor = (2.0 * math.pi / hbar) / N_points

    nk_mag = n_mag[k_idx, m]
    nq_mag = n_mag[q_idx, n]
    np_phon = n_phon[p_idx, lam]
    r = fgr_prefactor * V_sq * ((nq_mag + 1.0) * nk_mag * np_phon - nq_mag * (nk_mag + 1.0) * (np_phon + 1.0))

    wq = w_mag[q_idx, n]
    wk = w_mag[k_idx, m]
    wp = w_phon[p_idx, lam]
    f_m = r * (wq - wk)

    cuda.atomic.add(acc, 0, f_m)
    cuda.atomic.add(acc, 1, -r * wp)
    cuda.atomic.add(acc, 2, -r)
    cuda.atomic.add(flux_branch, lam, -f_m)
    b = int(wp * inv_bin_width)
    if b < 0:
        b = 0
    if b >= n_bins:
        b = n_bins - 1
    cuda.atomic.add(flux_spec, b, -f_m)


def bose_mu(w, T, mu=0.0):
    """n = 1 / (exp((w - mu)/kT) - 1); zero where w <= mu or T <= 0."""
    if T <= 0.0:
        return np.zeros_like(w, dtype=np.float64)
    x = (w - mu) / (KB_MEV * T)
    out = np.zeros_like(w, dtype=np.float64)
    ok = x > 0.0
    out[ok] = 1.0 / np.expm1(np.minimum(x[ok], 700.0))
    return out


def bose_entropy(n):
    """S = sum[(n+1) ln(n+1) - n ln n]  (in units of k_B)."""
    n = np.asarray(n, dtype=np.float64)
    pos = n > 0.0
    return float(np.sum((n[pos] + 1.0) * np.log1p(n[pos]) - n[pos] * np.log(n[pos])))


def _bisect(f, lo, hi, iters=60):
    """Root of an increasing f on [lo, hi]; nan if there is no sign change."""
    flo, fhi = f(lo), f(hi)
    if not (np.isfinite(flo) and np.isfinite(fhi)) or flo > 0.0 or fhi < 0.0:
        return float("nan")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if not np.isfinite(fm):
            return float("nan")
        if fm < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fit_thermal(w, n, free_mu):
    """
    Thermal distribution carrying the same energy (and, for free_mu, the same particle number)
    as the populations n on the modes w. Returns (T [K], mu [meV]) (nan, nan on failure).
      free_mu=False: n_BE(w; T, mu=0)            - phonons (number not conserved)
      free_mu=True : n_BE(w; T, mu), mu < w_min  - magnons (number conserved -> mu != 0)
    """
    w = np.asarray(w, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    N0 = float(np.sum(n))
    E0 = float(np.sum(w * n))
    if N0 <= 0.0 or E0 <= 0.0 or w.size == 0:
        return float("nan"), float("nan")

    lnT_lo, lnT_hi = np.log(0.05), np.log(1.0e5)

    if not free_mu:
        lnT = _bisect(lambda x: float(np.sum(w * bose_mu(w, np.exp(x)))) - E0, lnT_lo, lnT_hi)
        return (float(np.exp(lnT)), 0.0) if np.isfinite(lnT) else (float("nan"), float("nan"))

    wmin = float(w.min())

    def mu_of_T(T):
        mu_hi = wmin - 1e-9 * KB_MEV * T      # ground-state occupation ~ 1e9: always above N0
        lo = wmin - 20.0 * KB_MEV * T
        for _ in range(60):
            if float(np.sum(bose_mu(w, T, lo))) < N0:
                break
            lo -= 20.0 * KB_MEV * T
        return _bisect(lambda mu: float(np.sum(bose_mu(w, T, mu))) - N0, lo, mu_hi)

    def energy_gap(x):
        T = np.exp(x)
        mu = mu_of_T(T)
        return float("nan") if not np.isfinite(mu) else float(np.sum(w * bose_mu(w, T, mu))) - E0

    lnT = _bisect(energy_gap, lnT_lo, lnT_hi, iters=45)
    if not np.isfinite(lnT):
        return float("nan"), float("nan")
    T = float(np.exp(lnT))
    return T, float(mu_of_T(T))


def mode_temperature_average(n, w, w_min):
    """Legacy T_eff: energy-weighted mean of w / (kB ln(1 + 1/n)); assumes mu = 0, so NOT a temperature
    once the distribution carries a chemical potential."""
    ok = (n > 1e-10) & (w > w_min)
    if not np.any(ok):
        return 0.0
    mt = w[ok] / (KB_MEV * np.log1p(1.0 / n[ok]))
    return float(np.average(mt, weights=n[ok] * w[ok]))


class DynamicsDiagnostics:
    """
    Writes, from the populations and the flux kernel output at every observation step:

    observables_dynamics.txt  (tab separated; first 9 columns as before, then)
        S_mag, S_phon, S_tot          entropies [k_B], over the interacting modes
        T_mag_fit_K, mu_mag_meV       thermal (T, mu) with the magnons' N and E
        T_phon_fit_K                  thermal T (mu = 0) with the phonons' E
        dS_mag, dS_phon               S[thermal fit] - S[actual] >= 0: entropy still to be gained to thermalise
        eta_mag, eta_phon             sum w|n - n_fit| / sum w n: energy-weighted non-thermal fraction
        P_mp_cell                     magnon -> phonon energy flux [meV/ps per cell] = -dE_mag/dt
        dEtot_rate_cell               d(E_mag + E_phon)/dt [meV/ps per cell]; must vanish (broadening error)
        dNphon_rate_cell              dN_phon/dt per cell [1/ps]
        G_eff_cell                    P_mp / (T_mag_fit - T_phon_fit)  [meV/(K ps) per cell]
        dNmag_rate_cell               dN_mag/dt per cell [1/ps]; non-zero only with --relativistic (V++/V-- change the magnon number by -+2)
    spectral_dynamics.csv   (every spec_every-th observation)
        step,time_ps,subsystem,E_meV,n_mean,n_fit_mean,T_mode_K,n_modes   binned in energy
    flux_dynamics.csv
        step,time_ps,kind,index,E_meV,P_meV_per_ps_cell   kind = branch | energy (phonon-resolved flux)
    All energies are sums over the BZ mesh except the *_cell columns (divided by N_points).
    """

    def __init__(self, out_dir, w_mag, w_phon, N_points, n_bins=60, spec_every=10, exclude_idx=None):
        self.w_mag = np.asarray(w_mag, dtype=np.float64)
        self.w_phon = np.asarray(w_phon, dtype=np.float64)
        self.N = int(N_points)
        self.n_bins = int(n_bins)
        self.spec_every = max(1, int(spec_every))
        self.count = 0

        self.mask_mag = self.w_mag > MAGNON_MIN_ENERGY
        self.mask_phon = self.w_phon > PHONON_COUPLING_CUTOFF
        if exclude_idx is not None:          # Gamma: occupations are pinned to zero, not part of the bath
            self.mask_mag[exclude_idx, :] = False
            self.mask_phon[exclude_idx, :] = False
        w_max = float(max(self.w_mag.max(), self.w_phon.max()))
        self.edges = np.linspace(0.0, w_max * (1.0 + 1e-9), self.n_bins + 1)
        self.inv_bin_width = self.n_bins / self.edges[-1]
        self.bin_mag = np.minimum((self.w_mag * self.inv_bin_width).astype(np.int64), self.n_bins - 1)
        self.bin_phon = np.minimum((self.w_phon * self.inv_bin_width).astype(np.int64), self.n_bins - 1)
        self.branch_E = [float(self.w_phon[:, b][self.mask_phon[:, b]].mean()) if self.mask_phon[:, b].any() else 0.0
                         for b in range(self.w_phon.shape[1])]

        self.obs = open(f"{out_dir}/observables_dynamics.txt", "w")
        self.obs.write("Step\tTime_ps\tE_tot_meV\tE_mag_meV\tE_phon_meV\tN_mag\tN_phon\tT_eff_mag_K\tT_eff_phon_K\t"
                       "S_mag\tS_phon\tS_tot\tT_mag_fit_K\tmu_mag_meV\tT_phon_fit_K\tdS_mag\tdS_phon\t"
                       "eta_mag\teta_phon\tP_mp_cell\tdEtot_rate_cell\tdNphon_rate_cell\tG_eff_cell\tdNmag_rate_cell\n")
        self.spec = open(f"{out_dir}/spectral_dynamics.csv", "w")
        self.spec.write("step,time_ps,subsystem,E_meV,n_mean,n_fit_mean,T_mode_K,n_modes\n")
        self.flux = open(f"{out_dir}/flux_dynamics.csv", "w")
        self.flux.write("step,time_ps,kind,index,E_meV,P_meV_per_ps_cell\n")

    def _binned(self, subsystem, step, t, w, n, n_fit, bins, mask):
        cnt = np.bincount(bins[mask], minlength=self.n_bins)
        sw = np.bincount(bins[mask], weights=w[mask], minlength=self.n_bins)
        sn = np.bincount(bins[mask], weights=n[mask], minlength=self.n_bins)
        sf = np.bincount(bins[mask], weights=n_fit[mask], minlength=self.n_bins)
        for b in np.nonzero(cnt)[0]:
            Ebar, nbar, fbar = sw[b] / cnt[b], sn[b] / cnt[b], sf[b] / cnt[b]
            T_mode = Ebar / (KB_MEV * np.log1p(1.0 / nbar)) if nbar > 1e-14 else 0.0
            self.spec.write(f"{step},{t:.8e},{subsystem},{Ebar:.6f},{nbar:.8e},{fbar:.8e},{T_mode:.4f},{cnt[b]}\n")

    def record(self, step, t, n_mag, n_phon, flux_acc, flux_branch, flux_spec):
        n_mag = np.asarray(n_mag, dtype=np.float64)
        n_phon = np.asarray(n_phon, dtype=np.float64)
        N = self.N

        # Legacy block: unchanged definitions (sums over all modes)
        N_mag, N_phon = float(n_mag.sum()), float(n_phon.sum())
        E_mag = float((n_mag * self.w_mag).sum())
        E_phon = float((n_phon * self.w_phon).sum())
        T_eff_mag = mode_temperature_average(n_mag, self.w_mag, 1e-5)
        T_eff_phon = mode_temperature_average(n_phon, self.w_phon, 1e-5)

        # Interacting modes only (phonons below the vertex cutoff never scatter)
        wm, nm = self.w_mag[self.mask_mag], n_mag[self.mask_mag]
        wp, npn = self.w_phon[self.mask_phon], n_phon[self.mask_phon]
        T_m, mu_m = fit_thermal(wm, nm, free_mu=True)
        T_p, _ = fit_thermal(wp, npn, free_mu=False)
        nfit_m = bose_mu(self.w_mag, T_m, mu_m) if np.isfinite(T_m) else np.zeros_like(self.w_mag)
        nfit_p = bose_mu(self.w_phon, T_p, 0.0) if np.isfinite(T_p) else np.zeros_like(self.w_phon)
        nfit_m[~self.mask_mag] = 0.0
        nfit_p[~self.mask_phon] = 0.0

        S_m, S_p = bose_entropy(nm), bose_entropy(npn)
        dS_m = bose_entropy(nfit_m[self.mask_mag]) - S_m if np.isfinite(T_m) else float("nan")
        dS_p = bose_entropy(nfit_p[self.mask_phon]) - S_p if np.isfinite(T_p) else float("nan")
        eta_m = float(np.sum(wm * np.abs(nm - nfit_m[self.mask_mag])) / max(np.sum(wm * nm), 1e-300))
        eta_p = float(np.sum(wp * np.abs(npn - nfit_p[self.mask_phon])) / max(np.sum(wp * npn), 1e-300))

        acc = np.asarray(flux_acc, dtype=np.float64)
        P_mp = -acc[0] / N
        dE_rate = (acc[0] + acc[1]) / N
        dN_rate = acc[2] / N
        dNmag_rate = acc[3] / N if acc.shape[0] > 3 else 0.0      # nonzero only with the relativistic channels
        dT = T_m - T_p
        G_eff = P_mp / dT if np.isfinite(dT) and abs(dT) > 1e-9 else float("nan")

        self.obs.write(f"{step}\t{t:.8e}\t{E_mag + E_phon:.8e}\t{E_mag:.8e}\t{E_phon:.8e}\t{N_mag:.8e}\t{N_phon:.8e}\t"
                       f"{T_eff_mag:.4f}\t{T_eff_phon:.4f}\t{S_m:.8e}\t{S_p:.8e}\t{S_m + S_p:.8e}\t"
                       f"{T_m:.4f}\t{mu_m:.6f}\t{T_p:.4f}\t{dS_m:.6e}\t{dS_p:.6e}\t{eta_m:.6e}\t{eta_p:.6e}\t"
                       f"{P_mp:.8e}\t{dE_rate:.8e}\t{dN_rate:.8e}\t{G_eff:.8e}\t{dNmag_rate:.8e}\n")
        self.obs.flush()

        if self.count % self.spec_every == 0:
            self._binned("magnon", step, t, self.w_mag, n_mag, nfit_m, self.bin_mag, self.mask_mag)
            self._binned("phonon", step, t, self.w_phon, n_phon, nfit_p, self.bin_phon, self.mask_phon)
            for b, val in enumerate(np.asarray(flux_branch, dtype=np.float64) / N):
                self.flux.write(f"{step},{t:.8e},branch,{b},{self.branch_E[b]:.6f},{val:.8e}\n")
            centers = 0.5 * (self.edges[:-1] + self.edges[1:])
            for b, val in enumerate(np.asarray(flux_spec, dtype=np.float64) / N):
                self.flux.write(f"{step},{t:.8e},energy,{b},{centers[b]:.6f},{val:.8e}\n")
            self.spec.flush()
            self.flux.flush()
        self.count += 1

    def close(self):
        for f in (self.obs, self.spec, self.flux):
            f.close()


# ==========================================
# Hybrid (magnon-polaron) Boltzmann dynamics            [--hybrid_dynamics, needs --full_bz]
# ==========================================
# Theory: theory/mag_polaron_dyn.tex, "Collision Integrals". For occupations n_{k mu} of the hybrid modes,
#   dn_{k mu}/dt =  4pi/hbar  sum_{q,nu,eta} |G~|^2 delta(e_k - e_q - e_{k-q}) [(n_k+1) n_q n_{k-q} - n_k (n_q+1)(n_{k-q}+1)]
#                 + 8pi/hbar  sum_{q,nu,eta} |G~|^2 delta(e_k + e_q - e_{k+q}) [(n_k+1)(n_q+1) n_{k+q} - n_k n_q (n_{k+q}+1)]
# Both sums are generated by ONE set of microscopic events, the splittings P -> A + B, which is exactly the
# c_type 0 channel list that phase_1_scan_hybrid_path stores (with --full_bz the "path" is the whole grid, so every
# (parent, child, child) triple of the BZ is in the list, each unordered split twice - the ordered sum of the theory):
#   R_c = 4pi/(hbar N) |G~|^2 delta_c [ n_P (n_A+1)(n_B+1) - (n_P+1) n_A n_B ]
#   dn_P -= R_c,  dn_A += R_c,  dn_B += R_c
# A parent collects its sum over ordered pairs (prefactor 4pi/hbar); a child appears once as A and once as B for every
# partner (2 x 4pi/hbar = 8pi/hbar) - the coalescence term - so the c_type 1 (coalescence) channels are NOT used here
# (adding them would count every coalescence twice). Polariton number is not conserved (1 -> 2), hence mu = 0 at
# equilibrium; the conserved quantities are energy (up to the delta broadening) and, in the bare limit, magnon number.
HYB_MIN_ENERGY = 0.2      # meV, same cutoff as phase_1_scan_hybrid_path: lower modes never scatter


@cuda.jit
def phase_2_time_step_hybrid(chan_indices, chan_weights, num_channels, n_hyb, dn_hyb, N_points):
    idx = cuda.grid(1)
    if idx >= num_channels[0] or idx >= chan_weights.shape[0]:
        return
    if chan_indices[0, idx] != 0:          # splitting events only (see above)
        return

    q = chan_indices[1, idx]
    k = chan_indices[2, idx]
    p = chan_indices[3, idx]
    bq = chan_indices[4, idx]
    bk = chan_indices[5, idx]
    bp = chan_indices[6, idx]
    V_sq = chan_weights[idx]

    hbar = 0.6582119569  # meV * ps
    pref = (4.0 * math.pi / hbar) / N_points
    nb = n_hyb.shape[1]

    nq = n_hyb[q, bq]
    nk = n_hyb[k, bk]
    npp = n_hyb[p, bp]
    R = pref * V_sq * (nq * (nk + 1.0) * (npp + 1.0) - (nq + 1.0) * nk * npp)

    cuda.atomic.add(dn_hyb, q * nb + bq, -R)
    cuda.atomic.add(dn_hyb, k * nb + bk, R)
    cuda.atomic.add(dn_hyb, p * nb + bp, R)


@cuda.jit
def apply_euler_hybrid(n_hyb, dn_hyb, dt):
    idx = cuda.grid(1)
    if idx < dn_hyb.shape[0]:
        q = idx // n_hyb.shape[1]
        b = idx % n_hyb.shape[1]
        new_n = n_hyb[q, b] + dn_hyb[idx] * dt
        n_hyb[q, b] = new_n if new_n > 1e-15 else 1e-15
        dn_hyb[idx] = 0.0


@cuda.jit
def phase_lifetime_hybrid_dyn(chan_indices, chan_weights, num_channels, n_hyb, gamma, N_points):
    """
    SMRTA rate of every mode, linearised from exactly the events the dynamics integrates:
      parent  : 4pi/(hbar N) V (1 + n_A + n_B)
      child A : 4pi/(hbar N) V (n_B - n_P)       child B : 4pi/(hbar N) V (n_A - n_P)
    Compared at start-up with phase_lifetime_hybrid_path, which uses the c_type 1 channels for the coalescence term.
    """
    idx = cuda.grid(1)
    if idx >= num_channels[0] or idx >= chan_weights.shape[0]:
        return
    if chan_indices[0, idx] != 0:
        return

    q = chan_indices[1, idx]
    k = chan_indices[2, idx]
    p = chan_indices[3, idx]
    bq = chan_indices[4, idx]
    bk = chan_indices[5, idx]
    bp = chan_indices[6, idx]
    V_sq = chan_weights[idx]

    hbar = 0.6582119569
    pref = (4.0 * math.pi / hbar) / N_points
    nb = n_hyb.shape[1]
    nq = n_hyb[q, bq]
    nk = n_hyb[k, bk]
    npp = n_hyb[p, bp]

    cuda.atomic.add(gamma, q * nb + bq, pref * V_sq * (nk + npp + 1.0))
    cuda.atomic.add(gamma, k * nb + bk, pref * V_sq * (npp - nq))
    cuda.atomic.add(gamma, p * nb + bp, pref * V_sq * (nk - nq))


HYB_ACC_NAMES = ("dE_tot", "dE_mag", "dE_ph", "dN_mag", "dN_ph", "dN_hyb", "dSz", "dLz", "dS_entropy")


@cuda.jit
def phase_2_flux_diagnostic_hybrid(chan_indices, chan_weights, num_channels, n_hyb, eps, w_mag, w_ph, sz, lz,
                                   N_points, inv_bin_width, n_bins, acc, flux_band, flux_spec):
    """
    Instantaneous rates of the conserved-or-not quantities from the same events as phase_2_time_step_hybrid
    (run at output steps only). With dn_P = -R, dn_A = dn_B = +R every additive per-mode quantity X changes by
    R (X_A + X_B - X_P). acc (sums over the BZ mesh; divide by N for per-cell values), see HYB_ACC_NAMES:
      dE_tot  energy                     -> equals -sum R dE_c: the energy error of the broadened delta
      dE_mag/dE_ph  character-weighted subsystem energies (w_mag eps, w_ph eps)
      dN_mag/dN_ph  character-weighted particle numbers  (w_mag,  w_ph)  -> magnetization change
      dN_hyb  polariton number,  dSz/dLz  spin / phonon angular momentum,  dS_entropy  entropy production (>= 0)
    flux_band[b], flux_spec[bin] : magnon -> phonon energy flux -dE_mag, resolved by the parent's hybrid band index
    and by the parent's energy.
    """
    idx = cuda.grid(1)
    if idx >= num_channels[0] or idx >= chan_weights.shape[0]:
        return
    if chan_indices[0, idx] != 0:
        return

    q = chan_indices[1, idx]
    k = chan_indices[2, idx]
    p = chan_indices[3, idx]
    bq = chan_indices[4, idx]
    bk = chan_indices[5, idx]
    bp = chan_indices[6, idx]
    V_sq = chan_weights[idx]

    hbar = 0.6582119569
    pref = (4.0 * math.pi / hbar) / N_points

    nq = n_hyb[q, bq]
    nk = n_hyb[k, bk]
    npp = n_hyb[p, bp]
    R = pref * V_sq * (nq * (nk + 1.0) * (npp + 1.0) - (nq + 1.0) * nk * npp)

    eq = eps[q, bq]
    ek = eps[k, bk]
    ep = eps[p, bp]
    dEm = R * (w_mag[k, bk] * ek + w_mag[p, bp] * ep - w_mag[q, bq] * eq)

    cuda.atomic.add(acc, 0, R * (ek + ep - eq))
    cuda.atomic.add(acc, 1, dEm)
    cuda.atomic.add(acc, 2, R * (w_ph[k, bk] * ek + w_ph[p, bp] * ep - w_ph[q, bq] * eq))
    cuda.atomic.add(acc, 3, R * (w_mag[k, bk] + w_mag[p, bp] - w_mag[q, bq]))
    cuda.atomic.add(acc, 4, R * (w_ph[k, bk] + w_ph[p, bp] - w_ph[q, bq]))
    cuda.atomic.add(acc, 5, R)
    cuda.atomic.add(acc, 6, R * (sz[k, bk] + sz[p, bp] - sz[q, bq]))
    cuda.atomic.add(acc, 7, R * (lz[k, bk] + lz[p, bp] - lz[q, bq]))
    if nq > 0.0 and nk > 0.0 and npp > 0.0:
        cuda.atomic.add(acc, 8, R * (math.log((nk + 1.0) / nk) + math.log((npp + 1.0) / npp)
                                     - math.log((nq + 1.0) / nq)))

    cuda.atomic.add(flux_band, bq, -dEm)
    b = int(eq * inv_bin_width)
    if b < 0:
        b = 0
    if b >= n_bins:
        b = n_bins - 1
    cuda.atomic.add(flux_spec, b, -dEm)


def _bose_entropy_per_mode(n):
    n = np.asarray(n, dtype=np.float64)
    s = np.zeros_like(n)
    pos = n > 0.0
    s[pos] = (n[pos] + 1.0) * np.log1p(n[pos]) - n[pos] * np.log(n[pos])
    return s


def fit_temperature_weighted(weight, eps, E_target):
    """T (K) with  sum weight * eps * n_BE(eps; T) = E_target  (mu = 0). The theory's definition of an effective
    subsystem temperature: E_m(t) = E_m(T_m(t)) with E_m(T) = sum w_mag eps n_BE(eps; T)."""
    if not (E_target > 0.0):
        return float("nan")
    lnT = _bisect(lambda x: float(np.sum(weight * eps * bose_mu(eps, np.exp(x)))) - E_target,
                  np.log(0.05), np.log(1.0e5))
    return float(np.exp(lnT)) if np.isfinite(lnT) else float("nan")


class HybridDynamicsDiagnostics:
    """
    Metrics of the hybrid dynamics, in the file layout of DynamicsDiagnostics (so plot_dynamics.py reads both),
    all over the ACTIVE modes (eps >= HYB_MIN_ENERGY; softer modes, incl. Gamma, never scatter):

    observables_dynamics.txt  columns as for the bare dynamics, with these meanings
        E_tot, E_mag, E_phon         sum eps n, sum w_mag eps n, sum w_ph eps n          (sums over the mesh)
        N_mag, N_phon                sum w_mag n,  sum w_ph n   (character-weighted particle numbers)
        T_eff_mag/phon               mode-temperature average over magnon-like (w_mag>0.5) / phonon-like modes
        S_mag, S_phon, S_tot         entropy, split by character weights
        T_mag_fit_K, T_phon_fit_K    E_mag = E_m(T_mag), E_phon = E_p(T_phon) with E_m(T) = sum w_mag eps n_BE(T)
        mu_mag_meV                   0 (polariton number is not conserved)
        dS_mag, dS_phon              entropy still to gain towards the common equilibrium T_eq (character split)
        eta_mag, eta_phon            sum w eps |n - n_BE(T_eq)| / sum w eps n
        P_mp_cell                    -dE_mag/dt per cell,  dEtot_rate_cell = d(sum eps n)/dt per cell (broadening error)
        dNphon_rate_cell             d N_phon / dt per cell,  G_eff_cell = P_mp / (T_mag_fit - T_phon_fit)
      and the hybrid extras
        N_hyb                        polariton number
        Sz_cell, Lz_cell, Jz_cell    (1/N) sum S^z n, (1/N) sum L^z n, their sum
        dSz_rate_cell, dLz_rate_cell, dJz_rate_cell    their rates (J^z is not conserved by the hybridization)
        dN_mag_rate_cell             d N_mag / dt per cell (magnetization change)
        dSdt_cell                    entropy production rate per cell (>= 0)
        M_over_Meq                   M(t) / M(T_eq),  M = 1 - sum_active w_mag n / (N N_mag_branches)
        T_eq_K                       common final temperature from energy conservation
    spectral_dynamics.csv   subsystem 'magnon' / 'phonon' = modes with w_mag > 0.5 / <= 0.5; n_fit_mean = n_BE(T_eq)
    flux_dynamics.csv       kind 'energy' (parent energy bins) and 'branch' (parent hybrid band index)
    """

    def __init__(self, out_dir, eps, w_mag, w_ph, sz, lz, n_mag_branches, n_init, spec_every=10, n_bins=60,
                 pump_flat=None, n_base=None, n_snapshots=0, n_records=0):
        # pump_flat : flat indices (q * n_bands + band) of the pumped modes (empty/None without --pump)
        # n_base    : occupations before the pump was added (reference for the pump excess energy)
        # n_snapshots, n_records : number of full-occupation snapshots to keep (log-spaced in the record index) and
        #                          the total number of record() calls of the run
        self.pump_flat = np.zeros(0, dtype=np.int64) if pump_flat is None else np.asarray(pump_flat, dtype=np.int64)
        self.n_base = None if n_base is None else np.asarray(n_base, dtype=np.float64)
        self.snap_idx = set()
        if n_snapshots and n_records > 1:
            k = np.unique(np.round(np.geomspace(1, n_records - 1, max(1, int(n_snapshots) - 1))).astype(np.int64))
            self.snap_idx = {0, n_records - 1} | set(int(x) for x in k)
        self.snap_n, self.snap_t = [], []
        self.M0 = None
        self.pump_E0 = None
        self.eps = np.asarray(eps, dtype=np.float64)
        self.w_mag = np.asarray(w_mag, dtype=np.float64)
        self.w_ph = np.asarray(w_ph, dtype=np.float64)
        self.sz = np.asarray(sz, dtype=np.float64)
        self.lz = np.asarray(lz, dtype=np.float64)
        self.N = self.eps.shape[0]
        self.n_bins = int(n_bins)
        self.spec_every = max(1, int(spec_every))
        self.count = 0
        self.t_list, self.dT_list = [], []

        self.act = self.eps >= HYB_MIN_ENERGY
        e_max = float(self.eps.max())
        self.edges = np.linspace(0.0, e_max * (1.0 + 1e-9), self.n_bins + 1)
        self.inv_bin_width = self.n_bins / self.edges[-1]
        self.bins = np.minimum((self.eps * self.inv_bin_width).astype(np.int64), self.n_bins - 1)
        self.magnon_like = self.w_mag > 0.5
        self.cap_mag = float(self.N * n_mag_branches)
        self.band_E = [float(self.eps[:, b][self.act[:, b]].mean()) if self.act[:, b].any() else 0.0
                       for b in range(self.eps.shape[1])]

        # Equilibrium reference: the common temperature that carries the initial energy
        a = self.act
        self.E0_tot = float(np.sum(self.eps[a] * n_init[a]))
        lnT = _bisect(lambda x: float(np.sum(self.eps[a] * bose_mu(self.eps[a], np.exp(x)))) - self.E0_tot,
                      np.log(0.05), np.log(1.0e5))
        self.T_eq = float(np.exp(lnT)) if np.isfinite(lnT) else float("nan")
        self.n_eq = np.zeros_like(self.eps)
        if np.isfinite(self.T_eq):
            self.n_eq[a] = bose_mu(self.eps[a], self.T_eq)
        self.s_eq = _bose_entropy_per_mode(self.n_eq)
        self.Nmag_eq = float(np.sum(self.w_mag[a] * self.n_eq[a]))
        self.M_eq = 1.0 - self.Nmag_eq / self.cap_mag
        self.Sz_eq = float(np.sum(self.sz[a] * self.n_eq[a])) / self.N
        self.Lz_eq = float(np.sum(self.lz[a] * self.n_eq[a])) / self.N

        self.obs = open(f"{out_dir}/observables_dynamics.txt", "w")
        self.obs.write("Step\tTime_ps\tE_tot_meV\tE_mag_meV\tE_phon_meV\tN_mag\tN_phon\tT_eff_mag_K\tT_eff_phon_K\t"
                       "S_mag\tS_phon\tS_tot\tT_mag_fit_K\tmu_mag_meV\tT_phon_fit_K\tdS_mag\tdS_phon\t"
                       "eta_mag\teta_phon\tP_mp_cell\tdEtot_rate_cell\tdNphon_rate_cell\tG_eff_cell\t"
                       "N_hyb\tSz_cell\tLz_cell\tJz_cell\tdSz_rate_cell\tdLz_rate_cell\tdJz_rate_cell\t"
                       "dN_mag_rate_cell\tdSdt_cell\tM_over_Meq\tT_eq_K\t"
                       "M_over_M0\tn_pump_mean\tT_pump_K\tE_pump_excess_frac\n")
        self.spec = open(f"{out_dir}/spectral_dynamics.csv", "w")
        self.spec.write("step,time_ps,subsystem,E_meV,n_mean,n_fit_mean,T_mode_K,n_modes\n")
        self.flux = open(f"{out_dir}/flux_dynamics.csv", "w")
        self.flux.write("step,time_ps,kind,index,E_meV,P_meV_per_ps_cell\n")
        self.last = None

    def _binned(self, name, step, t, sel, n):
        m = sel & self.act
        b = self.bins[m]
        cnt = np.bincount(b, minlength=self.n_bins)
        sw = np.bincount(b, weights=self.eps[m], minlength=self.n_bins)
        sn = np.bincount(b, weights=n[m], minlength=self.n_bins)
        sf = np.bincount(b, weights=self.n_eq[m], minlength=self.n_bins)
        for i in np.nonzero(cnt)[0]:
            Ebar, nbar, fbar = sw[i] / cnt[i], sn[i] / cnt[i], sf[i] / cnt[i]
            T_mode = Ebar / (KB_MEV * np.log1p(1.0 / nbar)) if nbar > 1e-14 else 0.0
            self.spec.write(f"{step},{t:.8e},{name},{Ebar:.6f},{nbar:.8e},{fbar:.8e},{T_mode:.4f},{cnt[i]}\n")

    def record(self, step, t, n, acc, flux_band, flux_spec):
        n = np.asarray(n, dtype=np.float64)
        N = self.N
        a = self.act
        eps, wm, wp = self.eps[a], self.w_mag[a], self.w_ph[a]
        na = n[a]

        E_tot = float(np.sum(eps * na))
        E_mag = float(np.sum(wm * eps * na))
        E_ph = float(np.sum(wp * eps * na))
        N_mag, N_ph, N_hyb = float(np.sum(wm * na)), float(np.sum(wp * na)), float(np.sum(na))
        ml, pl = self.magnon_like[a], ~self.magnon_like[a]
        T_eff_m = mode_temperature_average(na[ml], eps[ml], 1e-5) if ml.any() else 0.0
        T_eff_p = mode_temperature_average(na[pl], eps[pl], 1e-5) if pl.any() else 0.0

        s = _bose_entropy_per_mode(na)
        S_tot = float(s.sum()); S_m = float(np.sum(wm * s)); S_p = float(np.sum(wp * s))
        ds = self.s_eq[a] - s
        dS_m, dS_p = float(np.sum(wm * ds)), float(np.sum(wp * ds))
        neq = self.n_eq[a]
        eta_m = float(np.sum(wm * eps * np.abs(na - neq)) / max(np.sum(wm * eps * na), 1e-300))
        eta_p = float(np.sum(wp * eps * np.abs(na - neq)) / max(np.sum(wp * eps * na), 1e-300))

        T_m = fit_temperature_weighted(wm, eps, E_mag)
        T_p = fit_temperature_weighted(wp, eps, E_ph)

        acc = np.asarray(acc, dtype=np.float64)
        dE_tot, dE_mag, dE_ph, dN_mag, dN_ph, dN_hyb, dSz, dLz, dSent = acc / N
        P_mp = -dE_mag
        dT = T_m - T_p
        G_eff = P_mp / dT if np.isfinite(dT) and abs(dT) > 1e-9 else float("nan")
        Sz = float(np.sum(self.sz[a] * na)) / N
        Lz = float(np.sum(self.lz[a] * na)) / N
        M_ratio = (1.0 - N_mag / self.cap_mag) / self.M_eq if self.M_eq != 0 else float("nan")

        # --- pump metrics (nan without --pump) and the magnetization relative to the initial state
        M_abs = 1.0 - N_mag / self.cap_mag
        if self.M0 is None:
            self.M0 = M_abs
        M_over_M0 = M_abs / self.M0 if self.M0 != 0 else float("nan")
        n_pump = T_pump = E_pump_frac = float("nan")
        if self.pump_flat.size:
            nf = n.reshape(-1)
            n_pump = float(np.mean(nf[self.pump_flat]))
            e_p = float(np.mean(self.eps.reshape(-1)[self.pump_flat]))
            T_pump = e_p / (KB_MEV * np.log1p(1.0 / n_pump)) if n_pump > 1e-14 else 0.0
            E_exc = float(np.sum(self.eps.reshape(-1)[self.pump_flat] * (nf[self.pump_flat] - self.n_base.reshape(-1)[self.pump_flat])))
            if self.pump_E0 is None:
                self.pump_E0 = E_exc
            E_pump_frac = E_exc / self.pump_E0 if self.pump_E0 != 0 else float("nan")
        if self.count in self.snap_idx:
            self.snap_n.append(n.copy()); self.snap_t.append(t)

        self.obs.write(f"{step}\t{t:.8e}\t{E_tot:.8e}\t{E_mag:.8e}\t{E_ph:.8e}\t{N_mag:.8e}\t{N_ph:.8e}\t"
                       f"{T_eff_m:.4f}\t{T_eff_p:.4f}\t{S_m:.8e}\t{S_p:.8e}\t{S_tot:.8e}\t{T_m:.4f}\t0.000000\t{T_p:.4f}\t"
                       f"{dS_m:.6e}\t{dS_p:.6e}\t{eta_m:.6e}\t{eta_p:.6e}\t{P_mp:.8e}\t{dE_tot:.8e}\t{dN_ph:.8e}\t{G_eff:.8e}\t"
                       f"{N_hyb:.8e}\t{Sz:.8e}\t{Lz:.8e}\t{Sz + Lz:.8e}\t{dSz:.8e}\t{dLz:.8e}\t{dSz + dLz:.8e}\t"
                       f"{dN_mag:.8e}\t{dSent:.8e}\t{M_ratio:.8f}\t{self.T_eq:.4f}\t"
                       f"{M_over_M0:.8f}\t{n_pump:.8e}\t{T_pump:.4f}\t{E_pump_frac:.8e}\n")
        self.obs.flush()

        if self.count % self.spec_every == 0:
            self._binned("magnon", step, t, self.magnon_like, n)
            self._binned("phonon", step, t, ~self.magnon_like, n)
            for b, val in enumerate(np.asarray(flux_band, dtype=np.float64) / N):
                self.flux.write(f"{step},{t:.8e},branch,{b},{self.band_E[b]:.6f},{val:.8e}\n")
            centers = 0.5 * (self.edges[:-1] + self.edges[1:])
            for b, val in enumerate(np.asarray(flux_spec, dtype=np.float64) / N):
                self.flux.write(f"{step},{t:.8e},energy,{b},{centers[b]:.6f},{val:.8e}\n")
            self.spec.flush(); self.flux.flush()
        self.count += 1

        self.t_list.append(t); self.dT_list.append(dT)
        row = dict(step=step, t=t, T_m=T_m, T_p=T_p, E_tot=E_tot, S_tot=S_tot, Sz=Sz, Lz=Lz, M=M_ratio, P=P_mp,
                   dE=dE_tot, dJ=dSz + dLz, dSent=dSent, eta_m=eta_m, eta_p=eta_p, N_hyb=N_hyb, N_mag=N_mag,
                   M0r=M_over_M0, n_pump=n_pump, T_pump=T_pump, E_pump_frac=E_pump_frac)
        if self.last is None:
            self.first = row
        self.last = row
        return row

    def save_snapshots(self, path, extra=None):
        """Full occupation n[q, band] at the kept record indices (log-spaced in time) + everything needed to plot
        the effective temperature T_eff(eps) of every mode (python/plot_occupation.py)."""
        if not self.snap_n:
            return False
        np.savez(path, t_ps=np.array(self.snap_t), n=np.array(self.snap_n), eps=self.eps, w_mag=self.w_mag,
                 w_ph=self.w_ph, n_eq=self.n_eq, T_eq=self.T_eq, pump_flat=self.pump_flat,
                 n_base=self.n_base if self.n_base is not None else np.zeros(0), min_energy=HYB_MIN_ENERGY,
                 **(extra or {}))
        return True

    def close(self):
        for f in (self.obs, self.spec, self.flux):
            f.close()


def run_hybrid_dynamics(out_dir, crystal_data, d_chan_indices, d_chan_weights, d_channel_count,
                        num_channels, max_channels, total_loops, N_points, T_mag_init, T_phon_init, dt, steps,
                        obs_every, spec_every, gamma_lifetime_ref, n_ref_equilibrium, threads_per_block=256,
                        pump=None, n_snapshots=40):
    """
    Time integration of the hybrid collision integral (see the header of this section) on the c_type 0 channels
    of the full-BZ hybrid-path scan. Initial state: the quasi-equilibrium ansatz of the theory,
    n*_{k mu} = w_mag n_BE(eps; T_mag) + w_ph n_BE(eps; T_phon).
    pump : None, or dict(modes=[resolved modes of mode_id.parse_pump], dn=[quanta per mode], base='thermal'|'none').
           The quanta are ADDED to the named modes on top of that background ('none': the background is zero, i.e. a
           vacuum plus pump).  See python/mode_id.py for how modes are named.
    n_snapshots : full-occupation snapshots kept for python/plot_occupation.py (log-spaced in time; 0 = none).
    gamma_lifetime_ref / n_ref_equilibrium : rates and occupations of the (unchanged) hybrid-path lifetime run,
    used for a start-up consistency check of the linearised event set against it.
    """
    print("\n" + "=" * 70)
    print(" HYBRID (magnon-polaron) BOLTZMANN DYNAMICS")
    print("=" * 70)
    if num_channels == 0:
        raise SystemExit("No hybrid scattering channels were found (check smearing/min_sigma and the mesh); "
                         "nothing to integrate.")
    if num_channels > max_channels:
        raise SystemExit(f"Hybrid channel buffer overflow: found {num_channels:,} > max {max_channels:,}; the event "
                         f"list is truncated. Increase anticipated_fraction to >= "
                         f"{1.05 * num_channels / total_loops:.4f} or use a smaller mesh.")

    nb = crystal_data.phon_branches + crystal_data.n_mag_branches
    eps = crystal_data.w_hyb
    phon_ch, mag_ch, phon_am, spin_am = crystal_data.extract_full_grid_hybrid_properties()
    d_eps = cuda.to_device(np.ascontiguousarray(eps))
    d_wm = cuda.to_device(np.ascontiguousarray(mag_ch))
    d_wp = cuda.to_device(np.ascontiguousarray(phon_ch))
    d_sz = cuda.to_device(np.ascontiguousarray(spin_am))
    d_lz = cuda.to_device(np.ascontiguousarray(phon_am))

    n_cpu = init_bose_einstein_two_temp(eps, mag_ch, T_mag_init, T_phon_init)
    n_cpu[~(eps >= HYB_MIN_ENERGY)] = 0.0          # inactive modes (incl. Gamma) carry no population in the dynamics
    if (n_cpu < 0.0).any():                        # w_mag outside [0, 1] (squeezed modes) can make the ansatz negative
        print(f" WARNING: quasi-equilibrium ansatz negative for {int((n_cpu < 0).sum())} modes "
              f"(w_mag range {mag_ch.min():.3f}..{mag_ch.max():.3f}); clipped to 0.")
        n_cpu = np.maximum(n_cpu, 0.0)

    # ---- named-mode pump: quanta added on top of the background (or of the vacuum with base = 'none') ----------
    pump_flat = np.zeros(0, dtype=np.int64)
    if pump is not None:
        if pump.get("base", "thermal") == "none":
            n_cpu[:] = 0.0
        n_base = n_cpu.copy()
        pump_flat = np.array([m["q_idx"] * nb + m["band"] for m in pump["modes"]], dtype=np.int64)
        for m, dn_i in zip(pump["modes"], pump["dn"]):
            if not eps[m["q_idx"], m["band"]] >= HYB_MIN_ENERGY:
                raise SystemExit(f"Pumped mode {m['label']} (q_idx {m['q_idx']}) is below the scattering cutoff "
                                 f"{HYB_MIN_ENERGY} meV and never scatters.")
            n_cpu[m["q_idx"], m["band"]] += dn_i
    else:
        n_base = n_cpu.copy()
    d_n = cuda.to_device(n_cpu)
    d_dn = cuda.to_device(np.zeros(N_points * nb, dtype=np.float64))

    blocks_eval = math.ceil(num_channels / threads_per_block)
    blocks_euler = math.ceil(N_points * nb / threads_per_block)

    # ---- start-up check 1: linearised rates of the event set vs the lifetime kernel -------------------------
    d_gam = cuda.to_device(np.zeros(N_points * nb, dtype=np.float64))
    d_nref = cuda.to_device(np.ascontiguousarray(n_ref_equilibrium))
    phase_lifetime_hybrid_dyn[blocks_eval, threads_per_block](
        d_chan_indices, d_chan_weights, d_channel_count, d_nref, d_gam, N_points)
    cuda.synchronize()
    gam_dyn = d_gam.copy_to_host().reshape(N_points, nb)
    gam_ref = np.asarray(gamma_lifetime_ref, dtype=np.float64).reshape(N_points, nb)
    act = eps >= HYB_MIN_ENERGY
    ok = act & (gam_ref > 1e-8) & (gam_dyn > 1e-8)
    print(" Start-up check: linearised dynamics rates vs phase_lifetime_hybrid_path (same channels, same n_BE):")
    if ok.any():
        r = gam_dyn[ok] / gam_ref[ok]
        print(f"   gamma_dyn / gamma_lifetime : median {np.median(r):.3f}, quartiles {np.quantile(r, .25):.3f}..{np.quantile(r, .75):.3f},"
              f"  ratio of sums {gam_dyn[ok].sum() / gam_ref[ok].sum():.3f}")
        for name, sel in (("magnon-like", mag_ch > 0.5), ("phonon-like", mag_ch <= 0.5)):
            s = ok & sel
            if s.any():
                print(f"   {name:12s}: ratio of sums {gam_dyn[s].sum() / gam_ref[s].sum():.3f}  ({s.sum():,} modes)")
        print("   (1 expected; deviations flag coalescence/splitting vertex asymmetry or snapping on the grid)")
    else:
        print("   no mode with a finite rate in both - check that the channel list is populated")
    g_max = float(gam_dyn[act].max()) if act.any() else float("nan")
    print(f" Largest linearised rate: {g_max:.4g} ps^-1  ->  Euler stable for dt << {1.0 / g_max:.3g} ps; "
          f"configured dt = {dt:g} ps (dt*gamma_max = {dt * g_max:.3g}); suggested dt ~ {0.05 / g_max:.3g} ps")
    if dt * g_max > 0.2:
        print("   WARNING: dt * gamma_max > 0.2 - the explicit Euler step is likely unstable / inaccurate.")

    n_records = int(math.ceil(steps / max(1, obs_every))) + 1          # record() calls: every obs_every steps + the last
    diag = HybridDynamicsDiagnostics(out_dir, eps, mag_ch, phon_ch, spin_am, phon_am,
                                     crystal_data.n_mag_branches, n_cpu, spec_every=spec_every,
                                     pump_flat=pump_flat, n_base=n_base, n_snapshots=n_snapshots, n_records=n_records)
    print(f" Initial state: T_mag = {T_mag_init} K, T_phon = {T_phon_init} K (quasi-equilibrium ansatz); "
          f"{int(act.sum()):,} of {act.size:,} modes active (eps >= {HYB_MIN_ENERGY} meV)")
    if pump is not None:
        print(f" PUMP (background: {'thermal as above' if pump.get('base', 'thermal') != 'none' else 'none (vacuum)'}):")
        E_in = 0.0
        for m, dn_i in zip(pump["modes"], pump["dn"]):
            e_m = dn_i * m["eps"] / N_points
            E_in += e_m
            n0 = n_cpu[m["q_idx"], m["band"]]
            T0 = m["eps"] / (KB_MEV * math.log1p(1.0 / n0)) if n0 > 1e-14 else 0.0
            print(f"   {m['label']:18s} q_idx {m['q_idx']:5d} band {m['band']:2d}  dn = {dn_i:10.4g} quanta  "
                  f"-> n = {n0:.4g} (T_eff = {T0:.4g} K)   energy {e_m:.4g} meV/cell   w_mag = {m['w_mag']:.3f}")
        E_bg = float(np.sum(eps[act] * n_base[act])) / N_points
        print(f"   absorbed {E_in:.4g} meV/cell on top of {E_bg:.4g} meV/cell of background "
              f"({E_in / max(E_bg, 1e-300):.3g} x)")
    print(f" Energy-conserving common temperature T_eq = {diag.T_eq:.3f} K;  M(T_eq) = {diag.M_eq:.5f} (capacity-normalised)")
    print(f" Equilibrium reference per cell: S^z_eq = {diag.Sz_eq:.5f}, L^z_eq = {diag.Lz_eq:.5f} (hbar)")
    # ---- start-up check 2: is the equilibrium state stationary under the broadened events? ----------------------
    # For n = n_BE(T_eq) the exact-delta collision integral vanishes; with a Gaussian of width sigma each event has
    # R ~ (e^{-dE/kT} - 1) != 0 and the energy rate -sum R dE is sign-definite (heating). Reported per cell.
    d_neq = cuda.to_device(np.ascontiguousarray(diag.n_eq))
    d_acc0 = cuda.to_device(np.zeros(len(HYB_ACC_NAMES), dtype=np.float64))
    d_fb0 = cuda.to_device(np.zeros(nb, dtype=np.float64))
    d_fs0 = cuda.to_device(np.zeros(diag.n_bins, dtype=np.float64))
    phase_2_flux_diagnostic_hybrid[blocks_eval, threads_per_block](
        d_chan_indices, d_chan_weights, d_channel_count, d_neq, d_eps, d_wm, d_wp, d_sz, d_lz,
        N_points, diag.inv_bin_width, diag.n_bins, d_acc0, d_fb0, d_fs0)
    cuda.synchronize()
    acc0 = d_acc0.copy_to_host() / N_points
    E0_cell = diag.E0_tot / N_points
    print(f" Equilibrium-state check (n = n_BE(T_eq)): dE_tot/dt = {acc0[0]:+.3e} meV/ps per cell "
          f"({acc0[0] / E0_cell:+.3e} /ps of E_tot/cell), dN_hyb/dt = {acc0[5]:+.3e} /ps, dS/dt = {acc0[8]:+.3e} k_B/ps per cell")
    print("   (should be ~0; a positive dE_tot/dt is the heating of the Gaussian-broadened delta - it shrinks with "
          "min_sigma/smearing. Test: T_mag = T_phon must keep E_tot and M/M_eq flat.)")
    print(f" Integrating {steps:,} steps of {dt:g} ps = {steps * dt:.4g} ps; diagnostics every {obs_every} steps")

    n_obs_total = max(1, steps // max(1, obs_every))
    print_every = max(1, n_obs_total // 40)
    nbins = diag.n_bins

    def run_diag(step, t):
        d_acc = cuda.to_device(np.zeros(len(HYB_ACC_NAMES), dtype=np.float64))
        d_fb = cuda.to_device(np.zeros(nb, dtype=np.float64))
        d_fs = cuda.to_device(np.zeros(nbins, dtype=np.float64))
        phase_2_flux_diagnostic_hybrid[blocks_eval, threads_per_block](
            d_chan_indices, d_chan_weights, d_channel_count, d_n, d_eps, d_wm, d_wp, d_sz, d_lz,
            N_points, diag.inv_bin_width, nbins, d_acc, d_fb, d_fs)
        cuda.synchronize()
        row = diag.record(step, t, d_n.copy_to_host(), d_acc.copy_to_host(), d_fb.copy_to_host(), d_fs.copy_to_host())
        if (diag.count - 1) % print_every == 0:
            e0 = diag.first["E_tot"]
            print(f"  step {step:>10,d}  t = {t:9.4f} ps | T_mag {row['T_m']:8.2f}  T_phon {row['T_p']:8.2f} K"
                  f" | dE/E0 {(row['E_tot'] - e0) / e0:+.2e} | S {row['S_tot']:.5e}"
                  f" | M/M_eq {row['M']:.5f} | J^z {row['Sz'] + row['Lz']:+.5e}"
                  f" | P {row['P']:+.3e}"
                  + (f" | n_pump {row['n_pump']:.4g}  M/M0 {row['M0r']:.5f}" if pump is not None else ""), flush=True)

    t_now = 0.0
    for step in range(steps):
        phase_2_time_step_hybrid[blocks_eval, threads_per_block](
            d_chan_indices, d_chan_weights, d_channel_count, d_n, d_dn, N_points)
        cuda.synchronize()
        if step % obs_every == 0:
            run_diag(step, t_now)
        apply_euler_hybrid[blocks_euler, threads_per_block](d_n, d_dn, dt)
        t_now += dt
    run_diag(steps, t_now)
    snap_saved = diag.save_snapshots(
        f"{out_dir}/occupation_snapshots.npz",
        extra=dict(q_frac=crystal_data.q_grid.astype(np.float64) / crystal_data.mesh,
                   pump_labels=np.array([m["label"] for m in pump["modes"]]) if pump else np.array([], dtype=str),
                   pump_dn=np.array(pump["dn"]) if pump else np.zeros(0)))
    diag.close()

    n_final = d_n.copy_to_host()
    np.savez(f"{out_dir}/final_populations_hybrid.npz", n_hyb=n_final, w_hyb=eps, mag_char=mag_ch, phon_char=phon_ch,
             spin_am=spin_am, phon_am=phon_am, T_eq=diag.T_eq, time_ps=t_now)

    # ---- summary ------------------------------------------------------------------------------------------
    f0, f1 = diag.first, diag.last
    dT = np.abs(np.array(diag.dT_list))
    t_arr = np.array(diag.t_list)
    t_efold = float("nan")
    if len(dT) > 2 and np.isfinite(dT[0]) and dT[0] > 0:
        hit = np.nonzero(dT <= dT[0] / math.e)[0]
        t_efold = float(t_arr[hit[0]]) if len(hit) else float("nan")
    obs_table = np.genfromtxt(f"{out_dir}/observables_dynamics.txt", names=True, delimiter="\t")
    dS_series = np.atleast_1d(obs_table["S_tot"])
    n_dec = int(np.sum(np.diff(dS_series) < -1e-9 * np.abs(dS_series[:-1]))) if dS_series.size > 1 else 0
    e0 = f0["E_tot"]
    print("\n" + "-" * 70)
    print(" HYBRID DYNAMICS SUMMARY")
    print("-" * 70)
    print(f" T_mag  : {f0['T_m']:.2f} -> {f1['T_m']:.2f} K     T_phon : {f0['T_p']:.2f} -> {f1['T_p']:.2f} K     T_eq = {diag.T_eq:.2f} K")
    print(f" e-folding time of |T_mag - T_phon| : {t_efold:.4g} ps")
    print(f" Energy drift (E_tot(end) - E_tot(0)) / E_tot(0)      : {(f1['E_tot'] - e0) / e0:+.3e}")
    print(f" Entropy gain S(end) - S(0)                            : {f1['S_tot'] - f0['S_tot']:+.5e} k_B   (decreasing output intervals: {n_dec})")
    print(f" Magnetization M/M(T_eq)                               : {f0['M']:.5f} -> {f1['M']:.5f}")
    print(f" Effective magnon number change  N_mag(end)/N_mag(0)-1 : {f1['N_mag'] / f0['N_mag'] - 1.0:+.3e}")
    print(f" Spin AM  S^z per cell: {f0['Sz']:+.5e} -> {f1['Sz']:+.5e}   phonon AM L^z: {f0['Lz']:+.5e} -> {f1['Lz']:+.5e}"
          f"   J^z: {f0['Sz'] + f0['Lz']:+.5e} -> {f1['Sz'] + f1['Lz']:+.5e} (hbar)")
    print(f" Non-thermal fraction eta (end): magnon-like {f1['eta_m']:.3e}, phonon-like {f1['eta_p']:.3e}")
    print(f" Polariton number N_hyb: {f0['N_hyb']:.6e} -> {f1['N_hyb']:.6e} (not conserved by 1 -> 2 processes)")
    print(f" Magnetization relative to the initial state M(t)/M(0)  : {f0['M0r']:.5f} -> {f1['M0r']:.5f}  (min {np.nanmin(np.atleast_1d(obs_table['M_over_M0'])):.5f})")
    if pump is not None:
        npm = np.atleast_1d(obs_table["n_pump_mean"]); tt = np.atleast_1d(obs_table["Time_ps"])
        half = np.nonzero(npm <= npm[0] / 2)[0]
        print(f" Pumped modes: n_pump {f0['n_pump']:.4g} -> {f1['n_pump']:.4g}; excess energy left in them "
              f"{f1['E_pump_frac']:.3e} of the initial; time to halve n_pump: "
              f"{tt[half[0]]:.4g} ps" if len(half) else
              f" Pumped modes: n_pump {f0['n_pump']:.4g} -> {f1['n_pump']:.4g}; excess energy left in them "
              f"{f1['E_pump_frac']:.3e} of the initial (n_pump did not halve within the run)")
    print(f" Outputs: observables_dynamics.txt, spectral_dynamics.csv, flux_dynamics.csv, final_populations_hybrid.npz"
          f"{', occupation_snapshots.npz' if snap_saved else ''} in {out_dir}")
    if snap_saved:
        print(f" Plot the occupation with:  python python/plot_occupation.py --dir {out_dir}")
    print("-" * 70)


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


def init_bose_einstein_two_temp(w_distribution, mag_character, T_m, T_p):
    """
    Quasi-equilibrium occupation n*_k = w_mag_k * n_BE(eps_k; T_m)
    + (1 - w_mag_k) * n_BE(eps_k; T_p), reusing init_bose_einstein for each
    term. Reduces exactly to init_bose_einstein(w_distribution, T_m) when
    T_m == T_p, since the character weights then cancel out of the mix.
    """
    n_at_Tm = init_bose_einstein(w_distribution, T_m)
    n_at_Tp = init_bose_einstein(w_distribution, T_p)
    return mag_character * n_at_Tm + (1.0 - mag_character) * n_at_Tp


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
    parser.add_argument("--dt", type=float, help="Override the time step (ps)")
    parser.add_argument("--full_bz", action="store_true",help="Evaluate hybrid lifetimes over the whole Brillouin zone.")
    parser.add_argument("--analyse_modes", type=str, default=None,
                        help="Per-channel scattering analysis for the given hybrid modes, "
                             "as a comma-separated list of path_idx:band (e.g. \"0:5,0:6\"). "
                             "Prints the dominant channels, where their partners sit in the "
                             "BZ, the partners' character / PAM / spin AM, and the J_z balance.")
    parser.add_argument("--no_slc", action="store_true",
                        help="Disable the spin-lattice coupling blocks in the BdG Hamiltonian. "
                             "The 'hybrid' modes then reduce to bare magnons and bare phonons "
                             "(the scattering vertex is unaffected). SLC is ON by default.")
    parser.add_argument("--Gmp_grid", action="store_true",
                        help="After the hybrid path/BZ channel scan, additionally sweep G_mp "
                             "over a 2D grid of (T_m, T_p) by cheaply re-weighting the already-"
                             "scanned channels (no new vertex evaluation). Only physically "
                             "meaningful together with --full_bz, since G_mp is a BZ-integrated "
                             "quantity. Writes Gmp_temperature_grid.csv.")
    parser.add_argument("--Gmp_T_min", type=float, default=5.0,
                        help="Lower bound (K) of the T_m/T_p sweep for --Gmp_grid.")
    parser.add_argument("--Gmp_T_max", type=float, default=100.0,
                        help="Upper bound (K) of the T_m/T_p sweep for --Gmp_grid.")
    parser.add_argument("--Gmp_T_step", type=float, default=5.0,
                        help="Step size (K) of the T_m/T_p sweep for --Gmp_grid.")
    parser.add_argument("--obs_every", type=int, default=1000,
                        help="Write the dynamics observables every this many time steps.")
    parser.add_argument("--spec_every", type=int, default=10,
                        help="Write the binned spectra / spectral flux every this many observations.")
    parser.add_argument("--hybrid_dynamics", action="store_true",
                        help="Run the time-resolved Boltzmann dynamics in the hybridized magnon-polaron basis "
                             "(theory/mag_polaron_dyn.tex) on the full-BZ hybrid-path channels, instead of the "
                             "bare magnon/phonon dynamics. Requires --full_bz. The lifetime calculations run unchanged.")
    parser.add_argument("--pump", action="append", default=None, metavar="SPEC",
                        help="Hybrid dynamics only: add quanta to a NAMED hybrid mode on top of the initial occupation. "
                             "SPEC = '<k>:<selectors>', e.g. \"G:Eu,L-,E=25.3\", \"G:b=22\", \"0.5,0,0:ph,E=12\"; "
                             "repeat the flag or separate with ';' for several modes. Modes are named by k-point, "
                             "irrep (Gamma, needs `structure` in the config), phonon helicity (L+/L-) and energy "
                             "(E=meV); list them with --pump_list. See python/mode_id.py.")
    parser.add_argument("--pump_E", type=float, default=None,
                        help="Energy absorbed by the pumped modes, meV per unit cell (shared equally between the "
                             "modes): dn = N_cells * E / eps for each mode.")
    parser.add_argument("--pump_dn", type=float, default=None,
                        help="Alternative to --pump_E: number of quanta added to EACH pumped mode.")
    parser.add_argument("--pump_base", choices=["thermal", "none"], default="thermal",
                        help="Background the pump is added to: the usual quasi-equilibrium state at --tmag/--tphon "
                             "('thermal', default) or empty modes ('none').")
    parser.add_argument("--pump_list", nargs="?", const="G", default=None, metavar="K",
                        help="Print the catalogue of named hybrid modes at k-point K (default G = Gamma; same syntax "
                             "as the k part of --pump) and exit. Needs no GPU and no scan.")
    parser.add_argument("--snapshots", type=int, default=40,
                        help="Hybrid dynamics: number of full-occupation snapshots (log-spaced in time) written to "
                             "occupation_snapshots.npz for python/plot_occupation.py (0 = none).")
    parser.add_argument("--legacy_bare", action="store_true",
                        help="reproduce the bare magnon-phonon kernels of before the audit of theory/bare_vertex_audit.md (old vertex arguments, "
                             "no magnon absorption term in the path rates, zero phonon gradients in the adaptive width)")
    parser.add_argument("--relativistic", action="store_true",
                        help="Bare basis: add the RELATIVISTIC magnon-number non-conserving channels ph(q) <-> mag(k) + "
                             "mag(q-k) (vertices V^{++}, V^{--} of theory/bare_scattering_rates_plus_relativistic.tex; they "
                             "need J^xx != J^yy or J^xy + J^yx != 0 in the spin-lattice tensor, and vanish for an isotropic "
                             "(exchange-striction) coupling). Adds them to the bare path lifetimes (path_lifetimes.csv gets "
                             "gamma_exch / gamma_rel columns and gamma_ps-1 becomes the total) and to the bare dynamics. Not used by "
                             "--hybrid_dynamics. Gaussian width of these channels: max(adaptive, --min_sigma), whereas the "
                             "number-conserving path channels floor it at 1e-5 meV. Partner modes at Gamma are excluded.")
    parser.add_argument("--phph", action="store_true",
                        help="Bare basis: add three-phonon (phonon-phonon) scattering from the third-order force constants "
                             "(theory/ph_ph_scattering.tex; one-atom cells, bcc Fe). Needs the config key `fc3` of the material "
                             "(phono3py fc3.hdf5; phono3py.yaml must lie next to it) or --phph_fc3. Adds the phonon RTA rates to the bare "
                             "path lifetimes (path_lifetimes.csv gets a gamma_phph_ps-1 column and gamma_ps-1 becomes the total) and the "
                             "phonon-phonon collision term to the bare dynamics. Same Gaussian width rule (adaptive, floored at --min_sigma) "
                             "and cut-off (modes below %g meV and the Gamma partner are excluded) as the other bare channels. "
                             "Not available with --hybrid_dynamics." % BARE_MIN_SCATTERING_ENERGY)
    parser.add_argument("--phph_fc3", type=str, default=None, metavar="FILE",
                        help="fc3.hdf5 for --phph (overrides the `fc3` entry of the material in the config).")
    parser.add_argument("--test_slc_aniso", type=str, default=None, metavar="A,B",
                        help="TEST ONLY: add a synthetic anisotropic part to every spin-lattice row between magnetic atoms: "
                             "J^xx += A r, J^yy -= A r, J^xy += B r, J^yx += B r (r = reproducible random 0.5..1.5), i.e. "
                             "J^xx - J^yy = 2A r and J^xy + J^yx = 2B r. Makes the --relativistic channels non-zero for "
                             "tensors that have none (the real CrI3 tensor has J^xx = J^yy, J^xy = -J^yx exactly).")
    parser.add_argument("--dynamics_only", action="store_true",
                        help="Skip all path / hybrid-path lifetime work and the G_mp grid; run only "
                             "the bare magnon-phonon channel scan and the time integration.")

    args = parser.parse_args()
    BARE_LEGACY = args.legacy_bare          # module global, read by the kernels at their first compilation

    if args.relativistic and args.hybrid_dynamics:
        print("NOTE: --relativistic enters the bare path lifetimes only; the hybrid dynamics does not include V++ / V--.")
    if args.hybrid_dynamics and not args.full_bz:
        raise SystemExit("--hybrid_dynamics integrates the full-BZ hybrid-path channels: add --full_bz.")
    if args.phph and args.hybrid_dynamics:
        raise SystemExit("--phph acts on the bare phonon populations only; it is not available with --hybrid_dynamics.")
    # The hybrid dynamics needs the hybrid-path scan, so --dynamics_only must not skip it.
    run_lifetimes = (not args.dynamics_only) or args.hybrid_dynamics

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
    dt = args.dt if args.dt is not None else sim_config["dt"]

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

    print(f" SLC       : {'ON' if not args.no_slc else 'OFF (--no_slc)'}")
    print(f"==================================================")

    crystal_data = CrystalDataSoA(
        mesh,
        Jijs,
        slc_files=slc_files,
        lattice_constant=lattice_constant,
        anisotropy=anisotropy,
        include_slc=not args.no_slc,
    )
    
    if args.test_slc_aniso:
        _a, _b = (float(x) for x in args.test_slc_aniso.split(","))
        _r = np.random.default_rng(3).uniform(0.5, 1.5, len(crystal_data.slc_J))
        _m = (crystal_data.slc_types[:, 0] <= crystal_data.n_mag_branches) & (crystal_data.slc_types[:, 1] <= crystal_data.n_mag_branches)
        crystal_data.slc_J[_m, 0, 0] += _a * _r[_m]; crystal_data.slc_J[_m, 1, 1] -= _a * _r[_m]
        crystal_data.slc_J[_m, 0, 1] += _b * _r[_m]; crystal_data.slc_J[_m, 1, 0] += _b * _r[_m]
        print(f" TEST ONLY: injected J^xx-J^yy = {2 * _a} r and J^xy+J^yx = {2 * _b} r into {int(_m.sum())} spin-lattice rows")

    crystal_data.print_summary()
    #crystal_data.check_slc_asr()

    # ---- named modes: catalogue and pump resolution BEFORE any scan (a bad name must fail early) ----------------
    structure_file = mat_config.get("structure")
    pump = None
    if args.pump_list is not None or args.pump:
        mode_props = crystal_data.extract_full_grid_hybrid_properties()
        if args.pump_list is not None:
            qi = mode_id.parse_k(args.pump_list, crystal_data)
            mode_id.print_catalogue(crystal_data, qi, mode_id.mode_catalogue(
                crystal_data, qi, mode_props, structure_file, min_energy=HYB_MIN_ENERGY))
            sys.exit(0)
        if not args.hybrid_dynamics:
            raise SystemExit("--pump sets the initial occupation of the hybrid dynamics: add --hybrid_dynamics --full_bz.")
        pump_modes = mode_id.parse_pump(args.pump, crystal_data, mode_props, structure_file, min_energy=HYB_MIN_ENERGY)
        pump = dict(modes=pump_modes, base=args.pump_base,
                    dn=mode_id.pump_occupation(pump_modes, crystal_data.N, E_abs=args.pump_E, dn=args.pump_dn))
    elif args.pump_E is not None or args.pump_dn is not None:
        raise SystemExit("--pump_E / --pump_dn need --pump (name the mode(s) to pump; list them with --pump_list).")

    gpu_data = crystal_data.push_to_gpu()

    gamma_idx = int(crystal_data.grid_map[0, 0, 0])

    # ---- three-phonon scattering (--phph): lattice-Fourier tables of the third-order force constants on the device ----
    pp_gpu = None
    pp_cache = {}                    # grid event list (d_idx, d_w, d_count, n, n_resonant), built once and reused
    pp_threads = (8, 8)
    pp_blocks_grid = (math.ceil(crystal_data.N / pp_threads[0]), math.ceil(crystal_data.N / pp_threads[1]))

    def pp_get_grid_events():
        if "ev" not in pp_cache:
            print(" -> phph: scanning the three-phonon phase space on the BZ grid ...", flush=True)
            pp_cache["ev"] = pp_gpu.scan_grid(gpu_data, smearing, min_sigma, gamma_idx, pp_blocks_grid, pp_threads)
            print(f"    resonant branch triples (upper bound) {pp_cache['ev'][4]:,}; events with a non-zero vertex {pp_cache['ev'][3]:,} "
                  f"({pp_cache['ev'][3] * 36 / 1e9:.2f} GB)", flush=True)
        return pp_cache["ev"]

    if args.phph:
        import phph_gpu
        fc3_file = args.phph_fc3 or mat_config.get("fc3")
        if not fc3_file:
            raise SystemExit(f"--phph: no third-order force constants given: add `fc3 = \"path/to/fc3.hdf5\"` to [materials.{args.material}] "
                             f"in {args.config} or pass --phph_fc3.")
        if not os.path.exists(fc3_file):
            raise SystemExit(f"--phph: fc3 file '{fc3_file}' not found.")
        pp_gpu = phph_gpu.PhPhGPU(fc3_file, crystal_data, BARE_MIN_SCATTERING_ENERGY)

    if run_lifetimes:
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


    if run_lifetimes:
        # ========================== Path Lifetime Evaluation ==========================
        print("\nStarting Path Lifetime Evaluation...")
    
        # 1. Setup fractional grid mapping
        grid_q_frac_cpu = crystal_data.q_grid.astype(np.float64) / crystal_data.mesh
        d_grid_q_frac = cuda.to_device(grid_q_frac_cpu)

        # 2. Setup Phase 1 memory for path scanning
        N_path = crystal_data.N_path
        total_path_loops = N_path * N_points * crystal_data.n_mag_branches**2 * crystal_data.phon_branches * (5 if args.relativistic else 3)
        max_path_channels = int(total_path_loops * anticipated_fraction)
    
        d_path_chan_indices = cuda.device_array((7, max_path_channels), dtype=np.int32)
        d_path_chan_weights = cuda.device_array(max_path_channels, dtype=np.float64)
        d_path_channel_count = cuda.to_device(np.zeros(1, dtype=np.int64))   # int64: the counter keeps counting past the buffer

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
        if args.relativistic:
            # relativistic processes (c_type 3/4) in their own kernel, appended to the same channel buffers
            phase_1_scan_path_rel[(blocks_x_path, blocks_y_path), threads_2d_path](
                gpu_data["mesh"], d_grid_q_frac, gpu_data["q_grid_cart"], gpu_data["grid_map"],
                d_path_q_frac, d_path_q_cart, d_path_w_phon, d_path_w_mag, d_path_eig_phon,
                gpu_data["w_phon"], gpu_data["w_mag"], gpu_data["eig_phon"],
                gpu_data["grad_f_phon"], gpu_data["grad_f_mag"],
                gpu_data["slc_axis"], gpu_data["slc_rij"], gpu_data["slc_rik"], gpu_data["slc_J"], gpu_data["slc_types"],
                smearing, min_sigma, d_path_chan_indices, d_path_chan_weights, d_path_channel_count,
                gpu_data["atom_masses"], gpu_data["mag_moments"], gamma_idx
            )
    
        cuda.synchronize()
    
        path_num_channels = d_path_channel_count.copy_to_host()[0]
        print(f" -> Path Channels found: {path_num_channels:,}")
        if path_num_channels > max_path_channels:
            msg = (f"path channel buffer overflow: found {path_num_channels:,} > max {max_path_channels:,}; the event list is "
                   f"truncated and the lifetimes are too small. Increase anticipated_fraction to >= "
                   f"{1.05 * path_num_channels / total_path_loops:.4f}.")
            if args.relativistic:
                raise SystemExit("Relativistic run: " + msg)
            print(" WARNING: " + msg)

        # Slice active channels
        d_path_chan_indices_active = d_path_chan_indices[:, :path_num_channels]
        d_path_chan_weights_active = d_path_chan_weights[:path_num_channels]

        # 4. Allocate Path Scattering Rate Arrays
        d_gamma_mag_path = cuda.to_device(np.zeros(N_path * crystal_data.n_mag_branches, dtype=np.float64))
        d_gamma_phon_path = cuda.to_device(np.zeros(N_path * crystal_data.phon_branches, dtype=np.float64))
        # relativistic (number non-conserving) contributions, accumulated separately (all zero without --relativistic)
        d_gamma_mag_rel_path = cuda.to_device(np.zeros(N_path * crystal_data.n_mag_branches, dtype=np.float64))
        d_gamma_phon_rel_path = cuda.to_device(np.zeros(N_path * crystal_data.phon_branches, dtype=np.float64))

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
            d_gamma_mag_rel_path,
            d_gamma_phon_rel_path,
            N_points  # Density of states normalization relies on BZ grid resolution
        )
        cuda.synchronize()

        # 6. Extract and Save
        gamma_mag_path_cpu = d_gamma_mag_path.copy_to_host().reshape((N_path, crystal_data.n_mag_branches))
        gamma_phon_path_cpu = d_gamma_phon_path.copy_to_host().reshape((N_path, crystal_data.phon_branches))
        gamma_mag_rel_cpu = d_gamma_mag_rel_path.copy_to_host().reshape((N_path, crystal_data.n_mag_branches))
        gamma_phon_rel_cpu = d_gamma_phon_rel_path.copy_to_host().reshape((N_path, crystal_data.phon_branches))
        gamma_mag_exch_cpu = gamma_mag_path_cpu.copy()       # number-conserving (exchange striction + antisym. xy)
        gamma_phon_exch_cpu = gamma_phon_path_cpu.copy()
        if args.relativistic:                                  # gamma_ps-1 becomes the TOTAL rate
            gamma_mag_path_cpu = gamma_mag_exch_cpu + gamma_mag_rel_cpu
            gamma_phon_path_cpu = gamma_phon_exch_cpu + gamma_phon_rel_cpu
            print(f" -> relativistic channels: sum gamma_rel / sum gamma_exch = magnons "
                  f"{gamma_mag_rel_cpu.sum() / max(gamma_mag_exch_cpu.sum(), 1e-300):.3e}, phonons "
                  f"{gamma_phon_rel_cpu.sum() / max(gamma_phon_exch_cpu.sum(), 1e-300):.3e}")

        # three-phonon rates (--phph): phonons only
        gamma_phph_cpu = np.zeros((N_path, crystal_data.phon_branches))
        if args.phph:
            print(" -> three-phonon (phph) lifetimes ...", flush=True)
            d_gp = cuda.to_device(np.zeros(N_path * crystal_data.phon_branches, dtype=np.float64))
            if args.full_bz:                  # path = grid: the linearised rates of the very event list used by the dynamics
                ev_idx, ev_w, ev_count, ev_n, _ = pp_get_grid_events()
                if ev_n > 0:
                    phph_gpu.phph_gamma_grid[math.ceil(ev_n / threads_per_block), threads_per_block](
                        ev_idx, ev_w, ev_count, d_n_phon, d_gp, N_points)
            else:
                blocks_pp_path = (math.ceil(N_path / pp_threads[0]), math.ceil(N_points / pp_threads[1]))
                ev_idx, ev_w, ev_count, ev_n, ev_res = pp_gpu.scan_path(
                    gpu_data, d_path_q_frac, d_path_w_phon, d_path_eig_phon, smearing, min_sigma, gamma_idx, blocks_pp_path, pp_threads)
                print(f"    phph path events: {ev_n:,}", flush=True)
                if ev_n > 0:
                    phph_gpu.phph_gamma_path[math.ceil(ev_n / threads_per_block), threads_per_block](
                        ev_idx, ev_w, ev_count, d_n_phon, d_gp, N_points)
            cuda.synchronize()
            gamma_phph_cpu = d_gp.copy_to_host().reshape((N_path, crystal_data.phon_branches))
            gamma_phon_path_cpu = gamma_phon_path_cpu + gamma_phph_cpu         # gamma_ps-1 becomes the total
            print(f" -> phph: mean phonon rate {gamma_phph_cpu.mean():.4e} ps^-1 (max {gamma_phph_cpu.max():.4e}); "
                  f"sum gamma_phph / sum gamma_(magnon-phonon) = {gamma_phph_cpu.sum() / max(gamma_phon_exch_cpu.sum() + gamma_phon_rel_cpu.sum(), 1e-300):.3e}")

        path_dist, label_comment = crystal_data.get_path_distance_info()

        # --- Observables for the BARE modes, same conventions as
        #     save_hybrid_path_properties (reuses get_nambu_angular_momentum so the
        #     two files are directly comparable). For unhybridized modes the
        #     subsystem weights are exact by construction: a bare magnon is 100%
        #     magnon and carries no phonon AM, a bare phonon is 100% phonon and
        #     carries no spin AM.
        _n_ph = crystal_data.phon_branches
        _n_mg = crystal_data.n_mag_branches
        _dim_block = _n_ph + _n_mg

        _L_z_total = crystal_data.get_nambu_angular_momentum(axis=2)

        # Phonon orbital AM: particle block, Cartesian index p = 3*atom + mu.
        _L_ph = _L_z_total[:_n_ph, :_n_ph]

        # Magnon spin AM in the magnon-only Nambu space (2*n_mag), pulled out of
        # the full matrix so the magnon BdG eigenvectors can be applied directly.
        _S_mag = np.zeros((2 * _n_mg, 2 * _n_mg), dtype=np.complex128)
        _S_mag[:_n_mg, :_n_mg] = _L_z_total[_n_ph:_dim_block, _n_ph:_dim_block]
        _S_mag[_n_mg:, _n_mg:] = _L_z_total[_dim_block + _n_ph:, _dim_block + _n_ph:]

        with open(f"{out_dir}/path_lifetimes.csv", "w") as f:
            f.write(label_comment + "\n")
            f.write("q_idx,qx,qy,qz,particle,branch,energy_meV,gamma_ps-1,tau_ps,"
                    "phon_character,mag_character,phon_AM_z_hbar,spin_AM_z_hbar,path_dist"
                    + (",gamma_exch_ps-1,gamma_rel_ps-1" if (args.relativistic or args.phph) else "")
                    + (",gamma_phph_ps-1" if args.phph else "") + "\n")

            # Magnons
            for i in range(N_path):
                qx, qy, qz = crystal_data.path_q_frac[i]
                U_mag = crystal_data.path_eig_mag[i]
                S_q = U_mag.conj().T @ _S_mag @ U_mag
                for b in range(crystal_data.n_mag_branches):
                    energy = crystal_data.path_w_mag[i, b]
                    gamma = gamma_mag_path_cpu[i, b]
                    tau = 1.0 / gamma if gamma > 1e-12 else float('inf')
                    spin_AM = S_q[b, b].real
                    extra = (f",{gamma_mag_exch_cpu[i, b]:.6e},{gamma_mag_rel_cpu[i, b]:.6e}" if (args.relativistic or args.phph) else "")
                    extra += (f",{0.0:.6e}" if args.phph else "")
                    f.write(f"{i},{qx:.6f},{qy:.6f},{qz:.6f},magnon,{b},{energy:.6f},{gamma:.6e},{tau:.6e},"
                            f"{0.0:.6f},{1.0:.6f},{0.0:.6e},{spin_AM:.6e},{path_dist[i]:.6f}{extra}\n")

            # Phonons
            for i in range(N_path):
                qx, qy, qz = crystal_data.path_q_frac[i]
                for b in range(crystal_data.phon_branches):
                    energy = crystal_data.path_w_phon[i, b]
                    gamma = gamma_phon_path_cpu[i, b]
                    tau = 1.0 / gamma if gamma > 1e-12 else float('inf')
                    # e is the mass-weighted polarisation vector, flattened to the
                    # same Cartesian index p = 3*atom + mu used by _L_ph.
                    e = crystal_data.path_eig_phon[i, b].reshape(-1)
                    phon_AM = np.vdot(e, _L_ph @ e).real
                    extra = (f",{gamma_phon_exch_cpu[i, b]:.6e},{gamma_phon_rel_cpu[i, b]:.6e}" if (args.relativistic or args.phph) else "")
                    extra += (f",{gamma_phph_cpu[i, b]:.6e}" if args.phph else "")
                    f.write(f"{i},{qx:.6f},{qy:.6f},{qz:.6f},phonon,{b},{energy:.6f},{gamma:.6e},{tau:.6e},"
                            f"{1.0:.6f},{0.0:.6f},{phon_AM:.6e},{0.0:.6e},{path_dist[i]:.6f}{extra}\n")

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

        # ---------------- Per-channel scattering analysis --------------------
        # Which modes does a given hybrid mode scatter into, where in the BZ do
        # they sit, are they hybrid, and is J_z conserved? Runs off the same
        # channel buffers the lifetime kernel consumed, so nothing is recomputed
        # on the GPU. Controlled by --analyse_modes "path_idx:band,...".
        if args.analyse_modes:
            from diagnose_channels import channel_table, path_observables
            print("\n -> Extracting grid/path mode observables for channel analysis...")
            _grid_props = crystal_data.extract_full_grid_hybrid_properties()
            _path_props = path_observables(crystal_data)

            for _spec in args.analyse_modes.split(","):
                _spec = _spec.strip()
                if not _spec:
                    continue
                try:
                    _pi, _bb = (int(x) for x in _spec.split(":"))
                except ValueError:
                    print(f"    skipping malformed --analyse_modes entry '{_spec}' "
                          "(expected path_idx:band)")
                    continue
                if _pi >= N_path or _bb >= num_hyb_branches:
                    print(f"    skipping out-of-range entry '{_spec}' "
                          f"(N_path={N_path}, branches={num_hyb_branches})")
                    continue
                channel_table(
                    crystal_data,
                    d_path_hyb_chan_indices, d_path_hyb_chan_weights,
                    path_hyb_num_channels, n_hyb_cpu, N_points,
                    path_idx=_pi, band=_bb,
                    grid_props=_grid_props, path_props=_path_props,
                )

        path_dist, label_comment = crystal_data.get_path_distance_info()

        # Phonon angular momentum of the hybrid modes. Identical construction to
        # save_hybrid_path_properties: L^z is intrinsically CARTESIAN (Levi-Civita
        # on x,y,z), so with the Hamiltonian in the bare-phonon-branch basis it must
        # be rotated per q-point before contracting with the hybrid eigenvectors.
        _L_z_tot = crystal_data.get_nambu_angular_momentum(axis=2)
        _L_cart = _L_z_tot[:crystal_data.phon_branches, :crystal_data.phon_branches]
        _n_ph = crystal_data.phon_branches
        _blk = num_hyb_branches
        _dim_tot = 2 * _blk

        # Subsystem weights from the para-unitarity metric J = diag(1_N, -1_N),
        # split into its phonon and magnon projections (same as
        # save_hybrid_path_properties). These are basis-independent - an identity
        # block is invariant under a unitary rotation inside its own subspace - so
        # unlike L^z they need no per-q rotation and are built once here.
        _J_phon = np.zeros((_dim_tot, _dim_tot), dtype=np.float64)
        _J_spin = np.zeros((_dim_tot, _dim_tot), dtype=np.float64)

        np.fill_diagonal(_J_phon[:_n_ph, :_n_ph], 1.0)
        np.fill_diagonal(_J_phon[_blk:_blk + _n_ph, _blk:_blk + _n_ph], -1.0)

        np.fill_diagonal(_J_spin[_n_ph:_blk, _n_ph:_blk], 1.0)
        np.fill_diagonal(_J_spin[_blk + _n_ph:_dim_tot, _blk + _n_ph:_dim_tot], -1.0)

        # Spin angular momentum operator (magnon block of L^z), same as
        # save_hybrid_path_properties. Unlike the phonon PAM this is diagonal in
        # the magnon subspace and basis-independent w.r.t. the phonon rotation, so
        # it is built once here rather than per q-point.
        _L_z_spin = np.zeros((_dim_tot, _dim_tot), dtype=np.complex128)
        _L_z_spin[_n_ph:_blk, _n_ph:_blk] = _L_z_tot[_n_ph:_blk, _n_ph:_blk]
        _L_z_spin[_blk + _n_ph:_dim_tot, _blk + _n_ph:_dim_tot] = \
            _L_z_tot[_blk + _n_ph:_dim_tot, _blk + _n_ph:_dim_tot]

        # Equilibrium Bose-Einstein occupation of the PATH modes at T_mag_init.
        # NOTE: n_hyb_cpu (built above from crystal_data.w_hyb) is the GRID
        # occupation used as the scattering partners' bath population inside the
        # kernel - it is indexed by grid q-point, not path index, so it cannot be
        # reused here. This is a separate array over path_w_hyb.
        n_hyb_occ = init_bose_einstein(crystal_data.path_w_hyb, T_mag_init)

        # Cartesian group velocities in Angstrom/ps (columns vx,vy,vz), from the grid gradient.
        v_hyb_path = crystal_data.get_path_group_velocities()

        with open(f"{out_dir}/hybrid_path_lifetimes.csv", "w") as f:
            f.write(label_comment + "\n")
            f.write("q_idx,qx,qy,qz,branch,energy_meV,gamma_ps-1,tau_ps,"
                    "phon_character,mag_character,phon_AM_z_hbar,spin_AM_z_hbar,n_occ,path_dist,vx,vy,vz\n")
            for i in range(N_path):
                qx, qy, qz = crystal_data.path_q_frac[i]
                T = crystal_data.path_eig_hyb[i]
                T_dag = T.conj().T

                # E_q[p, lambda] with p = 3*atom + mu
                E_q = crystal_data.path_eig_phon[i].reshape(_n_ph, _n_ph).T
                L_branch = E_q.conj().T @ _L_cart @ E_q

                L_z_phon = np.zeros((_dim_tot, _dim_tot), dtype=np.complex128)
                L_z_phon[:_n_ph, :_n_ph] = L_branch
                # Hole block: O_hole = -O_particle^*. The Cartesian shortcut
                # (hole == particle) does not survive the rotation.
                L_z_phon[_blk:_blk + _n_ph, _blk:_blk + _n_ph] = -L_branch.conj()

                AM_phon_q = T_dag @ L_z_phon @ T
                AM_spin_q = T_dag @ _L_z_spin @ T
                char_phon_q = T_dag @ _J_phon @ T
                char_spin_q = T_dag @ _J_spin @ T

                for b in range(num_hyb_branches):
                    energy = crystal_data.path_w_hyb[i, b]
                    gamma = gamma_hyb_path_cpu[i, b]
                    tau = 1.0 / gamma if gamma > 1e-12 else float('inf')
                    phon_char = char_phon_q[b, b].real
                    mag_char = char_spin_q[b, b].real
                    phon_AM = AM_phon_q[b, b].real
                    spin_AM = AM_spin_q[b, b].real
                    n_occ = n_hyb_occ[i, b]
                    f.write(f"{i},{qx:.6f},{qy:.6f},{qz:.6f},{b},{energy:.6f},{gamma:.6e},{tau:.6e},"
                            f"{phon_char:.6f},{mag_char:.6f},{phon_AM:.6e},{spin_AM:.6e},{n_occ:.6e},"
                            f"{path_dist[i]:.6f},"
                            f"{v_hyb_path[i, b, 0]:.6f},{v_hyb_path[i, b, 1]:.6f},{v_hyb_path[i, b, 2]:.6f}\n")

        print(f" -> Hybrid Path Channels: found_channels / max_channels : {path_hyb_num_channels / max_path_hyb_channels * 100:.2f}%")

        print(f"-> Saved hybrid path lifetimes to {out_dir}/hybrid_path_lifetimes.csv")


    if args.hybrid_dynamics:
        run_hybrid_dynamics(
            out_dir, crystal_data,
            d_path_hyb_chan_indices_active, d_path_hyb_chan_weights_active, d_path_hyb_channel_count,
            path_hyb_num_channels, max_path_hyb_channels, total_path_hyb_loops, N_points,
            T_mag_init, T_phon_init, dt, steps, args.obs_every, args.spec_every,
            gamma_hyb_path_cpu, n_hyb_cpu, threads_per_block=threads_per_block,
            pump=pump, n_snapshots=args.snapshots
        )
        print("Hybrid dynamics complete; the bare-basis scan and dynamics are skipped.")
        sys.exit(0)

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
    if num_channels > max_channels:
        raise SystemExit(f"Channel buffer overflow: found {num_channels:,} > max {max_channels:,}. "
                         f"Increase anticipated_fraction to at least "
                         f"{1.05 * num_channels / total_loops:.4f} (now {anticipated_fraction}).")

    # Slice the device arrays so Phase 2 ONLY iterates over valid channels
    # Slicing along the 2nd axis preserves the C-contiguous layout
    d_chan_indices_active = d_chan_indices[:, :num_channels]
    d_chan_weights_active = d_chan_weights[:num_channels]
    
    blocks_eval = math.ceil(num_channels / threads_per_block)

    # ====== Relativistic (number non-conserving) events: ph(q) <-> mag(k) + mag(q-k) ======
    num_nc = 0
    d_nc_indices_active = d_nc_weights_active = d_nc_count = None
    if args.relativistic:
        total_nc_loops = N_points**2 * crystal_data.n_mag_branches**2 * crystal_data.phon_branches
        max_nc = max(int(total_nc_loops * anticipated_fraction), 1)
        d_nc_indices = cuda.device_array((7, max_nc), dtype=np.int32)
        d_nc_weights = cuda.device_array(max_nc, dtype=np.float64)
        d_nc_count = cuda.to_device(np.zeros(1, dtype=np.int64))
        print("\nScanning the relativistic (V^{++}, V^{--}) phase space ...", flush=True)
        phase_1_scan_nc[blocks_per_grid_2d, threads_per_block_2d](
            gpu_data["mesh"], gpu_data["q_grid"], gpu_data["q_grid_cart"], gpu_data["grid_map"],
            gpu_data["w_phon"], gpu_data["w_mag"], gpu_data["eig_phon"],
            gpu_data["grad_f_phon"], gpu_data["grad_f_mag"],
            gpu_data["slc_axis"], gpu_data["slc_rij"], gpu_data["slc_rik"], gpu_data["slc_J"], gpu_data["slc_types"],
            smearing, min_sigma, d_nc_indices, d_nc_weights, d_nc_count,
            gpu_data["atom_masses"], gpu_data["mag_moments"], gamma_idx)
        cuda.synchronize()
        num_nc = int(d_nc_count.copy_to_host()[0])
        print(f"Relativistic events with a non-zero vertex: {num_nc:,}  ({num_nc / total_nc_loops:.2%} of the scanned phase space)", flush=True)
        if num_nc > max_nc:
            raise SystemExit(f"Relativistic event buffer overflow: found {num_nc:,} > max {max_nc:,}. Increase "
                             f"anticipated_fraction to at least {1.05 * num_nc / total_nc_loops:.4f} (now {anticipated_fraction}).")
        if num_nc == 0:
            _J = crystal_data.slc_J
            _aniso = max(float(np.abs(_J[:, 0, 0] - _J[:, 1, 1]).max()), float(np.abs(_J[:, 0, 1] + _J[:, 1, 0]).max()))
            if _aniso < 1e-12:
                print("  NOTE: V^{++} and V^{--} vanish identically for this spin-lattice tensor (J^xx = J^yy and J^xy = -J^yx in every "
                      "row, i.e. no anisotropic / symmetric-off-diagonal part): --relativistic changes nothing.")
            else:
                print(f"  NOTE: the tensor has anisotropic parts (max |J^xx-J^yy| or |J^xy+J^yx| = {_aniso:.3g}) but no event is "
                      f"energy-resonant on this mesh at sigma >= {min_sigma} meV (phonons below {PHONON_COUPLING_CUTOFF} meV and "
                      f"the Gamma point are excluded): try a larger --min_sigma or a finer mesh.")
        d_nc_indices_active = d_nc_indices[:, :num_nc]
        d_nc_weights_active = d_nc_weights[:num_nc]

        # start-up check: linearised rates of the event list vs the path-lifetime kernel (same events, same n_BE)
        if num_nc > 0 and run_lifetimes and args.full_bz and 'gamma_mag_rel_cpu' in globals():
            d_gm = cuda.to_device(np.zeros(N_points * crystal_data.n_mag_branches))
            d_gp = cuda.to_device(np.zeros(N_points * crystal_data.phon_branches))
            phase_lifetime_nc_dyn[math.ceil(num_nc / threads_per_block), threads_per_block](
                d_nc_indices_active, d_nc_weights_active, d_nc_count, d_n_mag, d_n_phon, d_gm, d_gp, N_points)
            cuda.synchronize()
            g_dyn_m = d_gm.copy_to_host().reshape(N_points, -1); g_dyn_p = d_gp.copy_to_host().reshape(N_points, -1)
            print(" Start-up check: linearised relativistic dynamics rates vs phase_lifetime_path (c_type 3/4), same n_BE:")
            for name, a, b in (("magnons", g_dyn_m, gamma_mag_rel_cpu), ("phonons", g_dyn_p, gamma_phon_rel_cpu)):
                ok = (a > 1e-10) & (b > 1e-10)
                if ok.any():
                    r = a[ok] / b[ok]
                    print(f"   {name}: gamma_dyn/gamma_lifetime median {np.median(r):.3f}, quartiles {np.quantile(r, .25):.3f}.."
                          f"{np.quantile(r, .75):.3f}, ratio of sums {a[ok].sum() / b[ok].sum():.3f}  ({int(ok.sum()):,} modes)")
                else:
                    print(f"   {name}: no mode with a finite rate in both")
            print("   (1 expected; the path scan and the event list use the same sigma floor (min_sigma), cut-offs and canonical labels, so any deviation is a bug; note the Gamma point is excluded in both)")

    # ====== Three-phonon events (--phph): a(q) -> b(k) + c(q-k) ======
    num_pp = 0
    d_pp_idx = d_pp_w = d_pp_count = None
    if args.phph:
        d_pp_idx, d_pp_w, d_pp_count, num_pp, _ = pp_get_grid_events()
        print(f"Three-phonon events in the dynamics: {num_pp:,}", flush=True)
        if num_pp > 0 and run_lifetimes and args.full_bz:
            # start-up check: the dynamics' linearised phonon rates vs the path-lifetime phph rates (same events: must agree)
            d_gm = cuda.to_device(np.zeros(N_points * crystal_data.phon_branches))
            phph_gpu.phph_gamma_grid[math.ceil(num_pp / threads_per_block), threads_per_block](
                d_pp_idx, d_pp_w, d_pp_count, d_n_phon, d_gm, N_points)
            cuda.synchronize()
            g_dyn = d_gm.copy_to_host().reshape(N_points, -1)
            ok = (g_dyn > 1e-10) & (gamma_phph_cpu > 1e-10)
            if ok.any():
                print(f" Start-up check phph: gamma_dyn / gamma_lifetime median {np.median(g_dyn[ok] / gamma_phph_cpu[ok]):.4f} "
                      f"(1 expected; thermal n_BE at T_phon)")

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
    """
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
    """
    cuda.synchronize()

    # Pull back and reshape
    gamma_mag_cpu = d_gamma_mag.copy_to_host().reshape((N_points, crystal_data.n_mag_branches))
    gamma_phon_cpu = d_gamma_phon.copy_to_host().reshape((N_points, crystal_data.phon_branches))

    if not args.dynamics_only:
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

    # Save
    np.savetxt(f"{out_dir}/w_mag_grid.csv", crystal_data.w_mag, delimiter=",")
    np.savetxt(f"{out_dir}/w_phon_grid.csv", crystal_data.w_phon, delimiter=",")

    # ========================== Time-evolution Phase ==========================
    
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
    
    blocks_nc = max(1, math.ceil(num_nc / threads_per_block))
    blocks_pp = max(1, math.ceil(num_pp / threads_per_block))
    diag = DynamicsDiagnostics(out_dir, crystal_data.w_mag, crystal_data.w_phon, N_points,
                               spec_every=args.spec_every, exclude_idx=gamma_idx)
    n_phon_branches = crystal_data.phon_branches

    def run_diagnostics(step, t):
        """Copy populations, run the flux kernel on the current state, append all diagnostics."""
        d_acc = cuda.to_device(np.zeros(4, dtype=np.float64))
        d_flux_branch = cuda.to_device(np.zeros(n_phon_branches, dtype=np.float64))
        d_flux_spec = cuda.to_device(np.zeros(diag.n_bins, dtype=np.float64))
        phase_2_flux_diagnostic[blocks_eval, threads_per_block](
            d_chan_indices_active, d_chan_weights_active, d_channel_count,
            d_n_mag, d_n_phon, gpu_data["w_mag"], gpu_data["w_phon"],
            N_points, diag.inv_bin_width, diag.n_bins, d_acc, d_flux_branch, d_flux_spec
        )
        if num_nc > 0:
            phase_2_flux_diagnostic_nc[blocks_nc, threads_per_block](
                d_nc_indices_active, d_nc_weights_active, d_nc_count,
                d_n_mag, d_n_phon, gpu_data["w_mag"], gpu_data["w_phon"],
                N_points, diag.inv_bin_width, diag.n_bins, d_acc, d_flux_branch, d_flux_spec
            )
        cuda.synchronize()
        diag.record(step, t, d_n_mag.copy_to_host(), d_n_phon.copy_to_host(),
                    d_acc.copy_to_host(), d_flux_branch.copy_to_host(), d_flux_spec.copy_to_host())
    
    print(f"\nStarting Phase 2: Time Integration ({steps} steps)...")

    current_time = 0.0

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
        if num_nc > 0:
            phase_2_time_step_nc[blocks_nc, threads_per_block](
                d_nc_indices_active, d_nc_weights_active, d_nc_count,
                d_n_mag, d_n_phon, d_dn_mag, d_dn_phon, N_points)
        if num_pp > 0:
            phph_gpu.phph_time_step[blocks_pp, threads_per_block](d_pp_idx, d_pp_w, d_pp_count, d_n_phon, d_dn_phon, N_points)
        cuda.synchronize()

            
        if step % args.obs_every == 0:
            run_diagnostics(step, current_time)

        # 4. Apply the safe_dt globally using the pure Euler kernel
        apply_euler_and_reset[blocks_euler, threads_per_block](
            d_n_mag, d_n_phon, d_dn_mag, d_dn_phon, dt
        )
        
        current_time += dt

    run_diagnostics(steps, current_time)       # final state
    diag.close()
    np.savez(f"{out_dir}/final_populations.npz", n_mag=d_n_mag.copy_to_host(), n_phon=d_n_phon.copy_to_host(),
             w_mag=crystal_data.w_mag, w_phon=crystal_data.w_phon, time_ps=current_time)
    print("Simulation Complete.")


