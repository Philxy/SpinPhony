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

# Initialize a multi-panel plot without using plt.figure()
fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# 1. Subsystem & Total Energies
axes[0].plot(df['Time_ps'], df['E_tot_meV'], label='Total Energy', color='black', linestyle='--', linewidth=1.5)
axes[0].plot(df['Time_ps'], df['E_mag_meV'], label='Magnon Subsystem', color='#d62728', linewidth=2)
axes[0].plot(df['Time_ps'], df['E_phon_meV'], label='Phonon Subsystem', color='#1f77b4', linewidth=2)
axes[0].set_ylabel('Energy (meV)', fontsize=11)
axes[0].legend(loc='best', frameon=True)
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].set_title('Relaxation Dynamics & Thermalization', fontsize=14, fontweight='bold', pad=15)

# 2. Particle Numbers
axes[1].plot(df['Time_ps'], df['N_mag'], label='Magnon Population', color='#d62728', linewidth=2)
axes[1].plot(df['Time_ps'], df['N_phon'], label='Phonon Population', color='#1f77b4', linewidth=2)
axes[1].set_ylabel('Particle Number', fontsize=11)
axes[1].legend(loc='best', frameon=True)
axes[1].grid(True, linestyle=':', alpha=0.6)

# 3. Effective Temperatures
axes[2].plot(df['Time_ps'], df['T_eff_mag_K'], label='Magnon $T_{\\mathrm{eff}}$', color='#d62728', linewidth=2)
axes[2].plot(df['Time_ps'], df['T_eff_phon_K'], label='Phonon $T_{\\mathrm{eff}}$', color='#1f77b4', linewidth=2)
axes[2].set_ylabel('Effective Temperature (K)', fontsize=11)
axes[2].set_xlabel('Time (ps)', fontsize=12, fontweight='bold')
axes[2].legend(loc='best', frameon=True)
axes[2].grid(True, linestyle=':', alpha=0.6)

# Fine-tune layout to avoid any label truncation or overlap
plt.tight_layout()

# Save the plot high resolution
output_image = "observables_dynamics_plot.png"
plt.savefig(output_image, dpi=300, bbox_inches='tight')
print(f"Plot successfully saved to {output_image}")
plt.show()