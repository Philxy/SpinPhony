import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
df_disp = pd.read_csv('Outputs/Hybrid/hybrid_path_properties.csv')
df_life = pd.read_csv('Outputs/Hybrid_band32/hybrid_path_lifetimes.csv')

# ---------------------------------------------------------
# 1.5 Handle Infinite and Non-Positive Lifetimes for Log Scale
# ---------------------------------------------------------
# Replace inf values with the maximum finite lifetime
max_finite_tau = df_life.loc[np.isfinite(df_life['tau_ps']), 'tau_ps'].max()
df_life['tau_ps'] = df_life['tau_ps'].replace([np.inf, -np.inf], max_finite_tau)

# Ensure all values are strictly positive (>0) for log scale
min_pos_tau = df_life.loc[df_life['tau_ps'] > 0, 'tau_ps'].min()
df_life['tau_ps'] = df_life['tau_ps'].clip(lower=min_pos_tau)

# ---------------------------------------------------------
# 2. Compute 1D Path Coordinates
# ---------------------------------------------------------
q_unique = df_disp[['q_idx', 'qx', 'qy', 'qz']].drop_duplicates().sort_values('q_idx')

dq_dense = np.diff(q_unique[['qx', 'qy', 'qz']].values, axis=0)
dist_dense = np.linalg.norm(dq_dense, axis=1)
q_unique['q_path'] = np.concatenate(([0], np.cumsum(dist_dense)))

df_disp = df_disp.merge(q_unique[['q_idx', 'q_path']], on='q_idx')

tree = cKDTree(q_unique[['qx', 'qy', 'qz']].values)
distances, idx = tree.query(df_life[['qx', 'qy', 'qz']].values)
df_life['q_path'] = q_unique['q_path'].values[idx]

# Set logarithmic color normalization
vmin = df_life['tau_ps'].min()
vmax = df_life['tau_ps'].max()
norm = LogNorm(vmin=vmin, vmax=vmax)
cmap = plt.get_cmap('viridis')

# ---------------------------------------------------------
# 3. Figure 1: Dense Lines + Sparse Color-Coded Scatter (Log Scale)
# ---------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(8, 6))

# Plot dense background bands
for band in df_disp['band'].unique():
    subset = df_disp[df_disp['band'] == band].sort_values('q_path')
    ax1.plot(subset['q_path'], subset['energy_meV'], color='lightgrey', lw=1, zorder=1)

# Scatter plot with LogNorm
sc = ax1.scatter(df_life['q_path'], df_life['energy_meV'], 
                 c=df_life['tau_ps'], cmap=cmap, norm=norm, 
                 s=30, zorder=2, edgecolors='k', linewidth=0.5)

cbar1 = fig1.colorbar(sc, ax=ax1)
cbar1.set_label(r'Lifetime $\tau$ (ps) [Log Scale]')
ax1.set_xlim(df_disp['q_path'].min(), df_disp['q_path'].max())
ax1.set_xlabel('Path distance')
ax1.set_ylabel('Energy (meV)')
ax1.set_title('Figure 1: Dispersion with Sparse Lifetime Scatter (Log Scale)')
fig1.tight_layout()

# ---------------------------------------------------------
# 4. Figure 2: Interpolated Color-Coded Lines (Log-Space Interp)
# ---------------------------------------------------------
def plot_colored_line(x, y, c, ax, cmap, norm, lw=2.5):
    """Helper to plot a line whose color changes smoothly on a log scale."""
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    # Geometric mean / log-average of segment endpoints for accurate log color mapping
    lc.set_array(np.sqrt(c[:-1] * c[1:]))
    lc.set_linewidth(lw)
    ax.add_collection(lc)

fig2, ax2 = plt.subplots(figsize=(8, 6))

for branch in df_life['branch'].unique():
    mask_life = df_life['branch'] == branch
    subset_life = df_life[mask_life].sort_values('q_path')
    
    if len(subset_life) < 2:
        continue
        
    mask_disp = df_disp['band'] == branch
    subset_disp = df_disp[mask_disp].sort_values('q_path')
    
    if subset_disp.empty:
        continue
        
    # Interpolate log10(tau) to get linear behavior in log-space across orders of magnitude
    log_tau_sparse = np.log10(subset_life['tau_ps'])
    f_interp = interp1d(subset_life['q_path'], log_tau_sparse, 
                        kind='linear', fill_value='extrapolate')
    
    # Convert back to tau space
    tau_dense = 10 ** f_interp(subset_disp['q_path'])
    
    plot_colored_line(subset_disp['q_path'], subset_disp['energy_meV'], tau_dense, 
                      ax2, cmap, norm)

# Add log-scaled colorbar for Figure 2
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar2 = fig2.colorbar(sm, ax=ax2)
cbar2.set_label(r'Interpolated Lifetime $\tau$ (ps) [Log Scale]')

ax2.set_xlim(df_disp['q_path'].min(), df_disp['q_path'].max())
ax2.set_ylim(df_disp['energy_meV'].min(), df_disp['energy_meV'].max())
ax2.set_xlabel('Path distance')
ax2.set_ylabel('Energy (meV)')
ax2.set_title('Figure 2: Dispersion Colored by Interpolated Lifetime (Log Scale)')
fig2.tight_layout()

plt.show()