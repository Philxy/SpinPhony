import os
import matplotlib.pyplot as plt
import pandas as pd

# Define file path
filename = "observables_dynamics.txt"

# Check if the file exists
if not os.path.exists(filename):
    print(f"Error: {filename} not found. Please ensure the simulation has run and generated this file.")
    exit()

# Load the data (handles variable whitespace/tabs automatically)
df = pd.read_csv(filename, sep=r'\s+')

# Calculate numerical correlation coefficient
energy_corr = df['E_mag_meV'].corr(df['E_tot_meV'])
print(f"Pearson correlation coefficient (E_mag vs E_tot): {energy_corr:.4f}")

# Initialize a 3x2 multi-panel plot layout without using plt.figure()
fig, axes = plt.subplots(3, 2, figsize=(13, 14))

# ------------------------------------------------------------
# Row 1, Column 1: Absolute Subsystem & Total Energies
# ------------------------------------------------------------
axes[0, 0].plot(df['Time_ps'], df['E_tot_meV'], label='Total Energy', color='black', linestyle='--', linewidth=1.5)
axes[0, 0].plot(df['Time_ps'], df['E_mag_meV'], label='Magnon Subsystem', color='#d62728', linewidth=2)
axes[0, 0].plot(df['Time_ps'], df['E_phon_meV'], label='Phonon Subsystem', color='#1f77b4', linewidth=2)
axes[0, 0].set_ylabel('Energy (meV)', fontsize=11)
axes[0, 0].set_xlabel('Time (ps)', fontsize=11)
axes[0, 0].legend(loc='best', frameon=True)
axes[0, 0].grid(True, linestyle=':', alpha=0.6)
axes[0, 0].set_title('Absolute Energy Evolution', fontsize=12, fontweight='bold')

# ------------------------------------------------------------
# Row 1, Column 2: Energy Fractions / Ratios (New)
# ------------------------------------------------------------
df['Ratio_mag'] = df['E_mag_meV'] / df['E_tot_meV']
df['Ratio_phon'] = df['E_phon_meV'] / df['E_tot_meV']

axes[0, 1].plot(df['Time_ps'], df['Ratio_mag'], label='Magnon Fraction', color='#d62728', linewidth=2)
axes[0, 1].plot(df['Time_ps'], df['Ratio_phon'], label='Phonon Fraction', color='#1f77b4', linewidth=2)
axes[0, 1].set_ylabel('Energy Fraction ($E_i / E_{\\mathrm{tot}}$)', fontsize=11)
axes[0, 1].set_xlabel('Time (ps)', fontsize=11)
axes[0, 1].legend(loc='best', frameon=True)
axes[0, 1].grid(True, linestyle=':', alpha=0.6)
axes[0, 1].set_title('Normalized Energy Distribution Ratio', fontsize=12, fontweight='bold')

# ------------------------------------------------------------
# Row 2, Column 1: Particle Numbers
# ------------------------------------------------------------
axes[1, 0].plot(df['Time_ps'], df['N_mag'], label='Magnon Population', color='#d62728', linewidth=2)
axes[1, 0].plot(df['Time_ps'], df['N_phon'], label='Phonon Population', color='#1f77b4', linewidth=2)
axes[1, 0].set_ylabel('Particle Number', fontsize=11)
axes[1, 0].set_xlabel('Time (ps)', fontsize=11)
axes[1, 0].legend(loc='best', frameon=True)
axes[1, 0].grid(True, linestyle=':', alpha=0.6)
axes[1, 0].set_title('Particle Populations', fontsize=12, fontweight='bold')

# ------------------------------------------------------------
# Row 2, Column 2: Effective Temperatures
# ------------------------------------------------------------
axes[1, 1].plot(df['Time_ps'], df['T_eff_mag_K'], label='Magnon $T_{\\mathrm{eff}}$', color='#d62728', linewidth=2)
axes[1, 1].plot(df['Time_ps'], df['T_eff_phon_K'], label='Phonon $T_{\\mathrm{eff}}$', color='#1f77b4', linewidth=2)
axes[1, 1].set_ylabel('Effective Temperature (K)', fontsize=11)
axes[1, 1].set_xlabel('Time (ps)', fontsize=11)
axes[1, 1].legend(loc='best', frameon=True)
axes[1, 1].grid(True, linestyle=':', alpha=0.6)
axes[1, 1].set_title('Effective Temperature Equilibration', fontsize=12, fontweight='bold')

# ------------------------------------------------------------
# Row 3, Column 1: Parametric Energy Space Correlation (New)
# ------------------------------------------------------------
sc1 = axes[2, 0].scatter(df['E_tot_meV'], df['E_mag_meV'], c=df['Time_ps'], cmap='viridis', s=15, alpha=0.8)
axes[2, 0].set_xlabel('Total Energy $E_{\\mathrm{tot}}$ (meV)', fontsize=11)
axes[2, 0].set_ylabel('Magnon Energy $E_{\\mathrm{mag}}$ (meV)', fontsize=11)
axes[2, 0].grid(True, linestyle=':', alpha=0.6)
axes[2, 0].set_title('Parametric Trajectory: $E_{\\mathrm{mag}}$ vs $E_{\\mathrm{tot}}$', fontsize=12, fontweight='bold')
cbar1 = fig.colorbar(sc1, ax=axes[2, 0], orientation='horizontal', pad=0.15)
cbar1.set_label('Time Progression (ps)', fontsize=10)

# ------------------------------------------------------------
# Row 3, Column 2: Parametric Temperature Pathway (New)
# ------------------------------------------------------------
sc2 = axes[2, 1].scatter(df['T_eff_phon_K'], df['T_eff_mag_K'], c=df['Time_ps'], cmap='plasma', s=15, alpha=0.8)
axes[2, 1].set_xlabel('Phonon Temperature $T_{\\mathrm{phon}}$ (K)', fontsize=11)
axes[2, 1].set_ylabel('Magnon Temperature $T_{\\mathrm{mag}}$ (K)', fontsize=11)

# Superimpose an equilibrium diagonal line where T_mag == T_phon
min_t = min(df['T_eff_phon_K'].min(), df['T_eff_mag_K'].min())
max_t = max(df['T_eff_phon_K'].max(), df['T_eff_mag_K'].max())
axes[2, 1].plot([min_t, max_t], [min_t, max_t], color='gray', linestyle=':', label='Thermal Equilibrium Line')

axes[2, 1].legend(loc='best')
axes[2, 1].grid(True, linestyle=':', alpha=0.6)
axes[2, 1].set_title('Thermalization Profile: $T_{\\mathrm{mag}}$ vs $T_{\\mathrm{phon}}$', fontsize=12, fontweight='bold')
cbar2 = fig.colorbar(sc2, ax=axes[2, 1], orientation='horizontal', pad=0.15)
cbar2.set_label('Time Progression (ps)', fontsize=10)

# Optimize plot formatting and save
plt.tight_layout()
output_image = "observables_extended_analysis.png"
plt.savefig(output_image, dpi=300, bbox_inches='tight')
print(f"Comprehensive analysis plot successfully saved to {output_image}")