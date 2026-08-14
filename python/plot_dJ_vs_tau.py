import matplotlib.pyplot as plt
import numpy as np

# Extracted data
m_eff = np.array([2138, 2204, 2484, 2468, 2592, 2117])
inv_tau = np.array([7.97e-6, 7.37e-6, 6.63e-3, 1.19e-4, 1.31e-4, 4.53e-5])  # in ps^-1

# Calculate lifetime tau in ps
tau = 1.0 / inv_tau

# Sort by M_eff for a continuous line plot
sorted_indices = np.argsort(m_eff)
x_sorted = m_eff[sorted_indices]
y_sorted = tau[sorted_indices]

# Create plot
plt.figure(figsize=(8, 5))
plt.plot(x_sorted, y_sorted, linestyle='-', color='tab:blue')  # Markers omitted

# Formatting
plt.yscale('log')
plt.xlabel(r'$M_{\mathrm{eff}}$')
plt.ylabel(r'Lifetime $\tau$ (ps)')
plt.title(r'Effective Mass ($M_{\mathrm{eff}}$) vs Lifetime ($\tau$)')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()

# Display plot
plt.show()