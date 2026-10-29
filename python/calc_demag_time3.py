import numpy as np
import pandas as pd

# File path
file_path = (
    "Outputs/CrI3_Path_Hyrbid_Full_sig_0.2_15K/hybrid_path_lifetimes.csv"
)

# Load CSV (skipping comment lines starting with '#')
df = pd.read_csv(file_path, comment="#")

# Physical constants and parameters
T = 15.0  # Temperature in Kelvin
k_B = 0.08617333262  # Boltzmann constant in meV / K

# Extract raw columns
energy_raw = df["energy_meV"].to_numpy()
gamma_raw = df["gamma_ps-1"].to_numpy()
n_occ_raw = df["n_occ"].to_numpy()
L_z_raw = df["phon_AM_z_hbar"].to_numpy()

# Filter: keep only finite modes with strictly positive scattering rates
valid = (
    np.isfinite(gamma_raw)
    & (gamma_raw > 0)
    & np.isfinite(energy_raw)
    & np.isfinite(L_z_raw)
)

energy = energy_raw[valid]
gamma = gamma_raw[valid]
n_occ = n_occ_raw[valid]
L_z = L_z_raw[valid]

print(f"Loaded {len(energy):,} active modes (gamma > 0) out of {len(energy_raw):,} total modes.")

# Calculate dn_BE / dT
dn_dT = np.where(
    energy > 1e-6,
    (energy / (k_B * (T**2))) * n_occ * (n_occ + 1.0),
    k_B / np.maximum(energy, 1e-12),
)


def compute_inverse_tau(L_values, gamma_values, weights):
    """Calculates 1 / tau_PAM = sum(gamma * L * dn/dT) / sum(L * dn/dT)"""
    w = L_values * weights
    numerator = np.sum(gamma_values * w)
    denominator = np.sum(w)

    if denominator == 0:
        return float("nan"), float("nan")

    inv_tau = numerator / denominator
    tau = 1.0 / inv_tau if inv_tau != 0 else np.inf
    return inv_tau, tau


# 1. Signed PAM (L_z)
inv_tau_signed, tau_signed = compute_inverse_tau(L_z, gamma, dn_dT)

# 2. Unsigned PAM (|L_z|)
inv_tau_unsigned, tau_unsigned = compute_inverse_tau(
    np.abs(L_z), gamma, dn_dT
)

# Display results
print("=" * 45)
print("Signed PAM (L_z):")
print(f"  1 / tau_PAM : {inv_tau_signed:14.6e} ps^-1")
print(f"  tau_PAM     : {tau_signed:14.6e} ps")
print("-" * 45)
print("Unsigned PAM (|L_z|):")
print(f"  1 / tau_PAM : {inv_tau_unsigned:14.6e} ps^-1")
print(f"  tau_PAM     : {tau_unsigned:14.6e} ps")
print("=" * 45)