import matplotlib.pyplot as plt
import numpy as np

# Extracted data
tau = np.array([
    1.26e5, 1.36e5, 151, 8390, 7640, 2.2e4, 
    8320, 1534, 826, 754, 280, 818
])

mean_dj_z = np.array([
    0.53, 1.85, -0.08, 0.91, 0.70, 0.57, 
    0.88, 0.90, 0.79, 1.28, 0.66, 0.96
])

# Calculate absolute values for mean |dJ|
mean_abs_dj = np.abs(mean_dj_z)

# Sort data by mean |dJ| to ensure the line plots smoothly
sorted_indices = np.argsort(mean_abs_dj)
x_sorted = mean_abs_dj[sorted_indices]
y_sorted = tau[sorted_indices]

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(x_sorted, y_sorted, linestyle='-', color='tab:blue') # Markers explicitly omitted

# Formatting
plt.yscale('log')
plt.xlabel('Mean |ΔJ_z|')
plt.ylabel('τ (ps)')
plt.title('Relaxation Time (τ) vs Mean |ΔJ_z|')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()

# Display the plot
plt.show()