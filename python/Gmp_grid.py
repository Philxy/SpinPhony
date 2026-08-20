import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# -------------------------------------------------------------------------
# 1. Physical Parameters & Unit Conversion
# -------------------------------------------------------------------------
# File paths
file_path_bccFe = "Outputs/bccFe_Full_NonHyrbid_sig_30.0_300K/Gmp_temperature_grid.csv"
file_path_CrI3 = "Outputs/CrI3/Gmp_temperature_grid.csv"

# Active material selection
material = "bccFe"  # Options: "bccFe" or "CrI3"
file_path = file_path_bccFe

# Energy conversion: 1 meV/(ps*K) -> W/K
EV_TO_JOULE = 1.602176634e-19
CONV_W_PER_K = (1e-3 * EV_TO_JOULE) / 1e-12  # 1.602176634e-10 W/K

# --- Unit Cell Volumes (in m^3) ---
# bcc-Fe: cubic cell, a ≈ 2.8665 Å (or 2.8 Å)
a_bccFe = 2.8665 * 1e-10  # meters
V_bccFe = a_bccFe**3  # ~ 2.355e-29 m^3

# CrI3: a = 6.8 Å
# (For a cubic/effective box volume V = a^3; for hexagonal unit cell: V = (sqrt(3)/2) * a^2 * c)
a_CrI3 = 6.8 * 1e-10  # meters
V_CrI3 = a_CrI3**3  # ~ 3.144e-28 m^3

# Assign volume and Curie temperature based on selected material
if material == "bccFe":
    V_total = V_bccFe
    T_C = 1043.0  # Curie temperature for bcc Fe (K)
    title_material = r"\mathrm{bcc\text{-}Fe}"
elif material == "CrI3":
    V_total = V_CrI3
    T_C = 45.0  # Curie temperature for monolayer CrI3 (K) (~61 K for bulk)
    title_material = r"\mathrm{CrI}_3"

# Volumetric conversion factor: meV/(ps*K) -> W/(m^3*K)
VOLUMETRIC_FACTOR = CONV_W_PER_K / V_total

# -------------------------------------------------------------------------
# 2. Data Loading and Processing
# -------------------------------------------------------------------------
df = pd.read_csv(file_path)

# Convert G_mp to volumetric SI units (W/m^3/K)
df["G_mp_vol"] = df["G_mp"] * VOLUMETRIC_FACTOR

# -------------------------------------------------------------------------
# 3. Plot 1: 2D Heatmap across Tm and Tp
# -------------------------------------------------------------------------
grid = df.pivot(index="T_m_K", columns="T_p_K", values="G_mp_vol")
grid = grid.sort_index(ascending=False)  # Lower Tm at the bottom

plt.figure(figsize=(8, 6.5))
sns.heatmap(
    grid,
    cmap="viridis",
    cbar_kws={"label": r"$G_{mp}$ ($\mathrm{W\cdot m^{-3}\cdot K^{-1}}$)"},
)
plt.title(
    rf"Volumetric $G_{{mp}}$ Temperature Grid (${title_material}$)",
    fontsize=13,
    pad=12,
)
plt.xlabel(r"$T_p$ (K)", fontsize=11)
plt.ylabel(r"$T_m$ (K)", fontsize=11)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------
# 4. Plot 2: 1D Equilibrium Curve (Tm = Tp) vs T / T_C
# -------------------------------------------------------------------------
# Filter for Tm == Tp
df_eq = df[np.isclose(df["T_m_K"], df["T_p_K"])].copy()
df_eq = df_eq.sort_values(by="T_m_K")

df_eq["T_norm"] = df_eq["T_m_K"] / T_C

plt.figure(figsize=(7, 5))
plt.plot(
    df_eq["T_norm"],
    df_eq["G_mp_vol"],
    marker="o",
    linestyle="-",
    linewidth=1.8,
    markersize=5,
    color="navy",
    label=r"$T_m = T_p$",
)

# Reference vertical line at T/T_C = 1
plt.axvline(
    x=1.0, color="firebrick", linestyle="--", alpha=0.7, label=r"$T = T_C$"
)

plt.yscale("log")

plt.xlabel(r"$T / T_C$", fontsize=12)
plt.ylabel(r"$G_{mp}$ ($\mathrm{W\cdot m^{-3}\cdot K^{-1}}$)", fontsize=12)
plt.title(
    rf"Equilibrium Volumetric Coupling ($T_m = T_p$) - ${title_material}$",
    fontsize=13,
    pad=10,
)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(frameon=True)
plt.tight_layout()
plt.show()